# SuperMimic HOI 特权 teacher：设计与实施计划

目标：在官方 SuperMimic（`deaf759`）上训练一个特权 teacher，在仿真里完成 OMOMO 八类物体
（largebox、smallbox、plasticbox、suitcase、whitechair、woodchair、largetable、smalltable）的
人-物交互跟踪。学习算法、超参和网络结构沿用官方，改动集中在任务层。

## 1. 原则

1. 想象里的奖励和终止都是 WM 状态、参考动作和物体几何的解析函数（可微），不学奖励头或接触头。
2. 仿真和想象调用同一套公式（`supermimic/hoi/terms.py`），权重和阈值只在 `HoiTaskCfg` 里定义一次。
3. 新代码放在 `src/supermimic/hoi/` 和 `tools/hoi/`，核心文件只加钩子，方便以后 rebase 官方更新。
4. 单轮更新时间优先：任何会让每一步仿真或每一轮更新变慢的设计都要有理由。

## 2. 已定的决定

| 项 | 决定 |
|---|---|
| 起点 | 官方 `deaf759`，分支 `hoi/g1-collision-hulls` 之上；不加回采集噪声 |
| 数据 | 椅子、桌子：HOI-Retarget G1（OMOMO，物体统一缩放 0.83），用户目测筛选的表格；箱子类：ULTRA `omomo_aug_v4` 中物体 100% 尺寸的 clip，经质量检查后按源动作轮流抽取，配平到与椅子/桌子每类相近的帧数 |
| 物体名义质量 (kg) | whitechair 2.5、woodchair 3、smalltable 2.5、largetable 4、smallbox 0.5、largebox 2.0、plasticbox 1.0、suitcase 2.0 |
| 随机化 | 官方三项（脚摩擦、躯干质心、编码器偏置）+ 物体质量 ×0.5–1.5（惯量同比缩放）；物体摩擦固定 1.0 |
| sim_ctx | 官方三项 + log 物体质量 + 物体类别 one-hot + 物体尺寸（log 缩放） |
| Actor | 特权 `rep_state`：看到物体真实状态和几何特征；另加不归一化的物体上下文（one-hot + 尺寸），不看质量 |
| WM / critic | 自动拿到 sim_ctx（含 one-hot、尺寸、质量） |
| WM 预热 | 前 500 轮更新只训 WM |
| 梯度累积 | 策略默认分 2 块（4096 环境下 L40 峰值约 39.5 GB；1 块显存不够） |
| 手脚碰撞 | 用 `assets/g1_collision` 的凸块：手替换 mjlab 手部胶囊；脚凸块只和物体碰，地面接触仍走官方脚底胶囊 |

## 3. 每个 world 放一个物体（为速度设计）

mjwarp 所有 world 共用一个模型。做法：

- 只有一个物体刚体（一个 free joint，nv 只多 6）。它的几何体是所有"物体变体"凸块的并集。
- **启动时**给每个 world 固定分配一个变体（按各变体的数据时长比例分配，至少 16 个 world），
  同时抽一次质量缩放。用 mjlab 的按 world 模型字段写入：
  - `geom_pos`：不属于本 world 变体的凸块整体上移 50 m，宽相碰撞直接剔除；
  - `body_mass / body_inertia / body_ipos / body_iquat`：本变体的惯量，乘以质量缩放。
  这些只在启动时写一次，`set_const` 也只跑一次。训练中重置不改模型，单步仿真时间和单物体环境相同量级。
- 重置时，每个 world 只从**自己变体**的 clip 里抽起始帧（回报采样分布在变体内重新归一化）。
- 碰撞过滤（contype / conaffinity 位掩码）：

| | 地面 | 机器人自身 | 物体 |
|---|---|---|---|
| 脚底胶囊（官方） | ✓ | ✓ | ✗ |
| 脚凸块 | ✗ | ✗ | ✓ |
| 手凸块 | ✓ | ✓ | ✓ |
| 其他身体胶囊 | ✓ | ✓ | ✓ |
| 物体凸块 | ✓ | — | — |

- 物体凸块：CoACD（阈值 0.06、不删顶点、最多 12 块），每块再包一个外包多面体：从块的主轴盒子开始，
  每次加入当前多面体越界最多的那个真实表面平面，直到越界 ≤ 1 mm 或满 38 个面（≤ 72 个顶点）。
  所有平面都是块的支撑面，所以外包保守、并贴合任意朝向的表面（固定方向的 k-DOP 在 ULTRA 箱子上
  因网格在局部系里是斜的，角上会鼓出 4–7 cm）。接触 condim 3、solref 0.01、摩擦 1.0。全局椭圆摩擦锥、impratio 10。
- 求解器迭代上限 20（官方 10）：椭圆锥 + impratio 10 在 10 次上限时约 4% 的子步没收敛（拿着物体时更多），
  单个子步物体速度误差 p99 约 3 cm/s；20 次时全部收敛，单步时间只多约 2%（`tools/hoi/solver_check.py`）。

## 4. 数据（阶段 1）

`tools/hoi/build_hoi_dataset.py` 读用户清单和数据管线的输出（每段 clip 的 `labels.npz` 已经是 50 Hz：
机器人根位姿、关节、物体位姿，以及 40 个通道的几何接触标签），写一个打包目录：

- mjlab 打包格式：`root_pos_w / root_quat_w / root_lin_vel_w / root_ang_vel_w / joint_pos / joint_vel`、
  `motion_ends.npy`、`joint_names.json`、`motion_names.json`；速度由 50 Hz 中心差分得到。
- HOI 附加表（同一目录 `hoi/`）：物体位姿与速度、左右手接触标签（-1 未知 / 0 / 1）、每段 clip 的变体编号、
  `variants.json`（类别、尺寸缩放、名义质量、几何来源哈希）。
- 每个变体的物体资产（`hoi/objects/<variant>/`）：凸块 OBJ、SDF 网格（1 cm，含法向）、
  最低点用的凸块顶点、由体素算出的质心和惯量（按名义质量）。

几何来源：管线为每个物体存的 object-local 碰撞面（与轨迹同一坐标系，已按数据源缩放）。

- 用户清单（`--table`，xlsx/tsv/csv）：第一列 clip，中间列是类别（carry / grasp / push / kick / pull），
  `notes` 写了 trim 的 clip 只保留第一段不含"只有一只手有接触标签"的连续片段（≥ 1 s），其余全部丢掉；
  一段都留不下的 clip 整段不用。
- 接触标签：手（wrist_yaw / rubber_hand 通道）和脚（ankle_roll 通道）各存一份；管线把穿模帧标成未知，
  这里把穿模也算作接触。
- `ref_intersection`：管线判定"机器人某个身体和物体相交"的帧（只有是否相交，没有深度），仅作参考。
- RSI 起点：任务启动时用仿真里的碰撞几何（手脚凸块、身体胶囊）在每个参考帧上查物体 SDF，
  算出插进物体的最大深度；只从深度 ≤ 1 cm（`start_penetration`）的帧起步。
  一帧都达不到的 clip，退而用它穿透最浅的 10% 帧。
- 重置时若物体最低点在地面以下，把物体抬到地面上方 1 mm。

## 5. 世界模型（阶段 3）

`HoiGenCoordRepresentation`：

- 全状态 = 官方 `[root 13, qj, vj]` + 物体 `[pos 3, quat 4, lin_vel 3, ang_vel 3]` + 变体编号 1。
- aux 状态 = `[root_pos, root_quat, obj_pos, obj_quat, variant]`。
- rep_state = 官方 68 维 + 物体在 root 系下 `[位置 3, rot6d 6, 线速度 3, 角速度 3, 高度 1]`
  + 手-物几何特征 24（每只手 3 个探针点 × (SDF 值, 表面法向 3)）+ 物体最低点离地 1。
- 网络多预测物体的 `Δv, Δω`（root 系），用和 root 相同的半隐式积分；积分后用 FK 和 SDF 重算几何特征。
- loss 加 `object_pos`、`object_ori` 两块（名义误差 0.01 m、0.03 rad）。

## 6. 奖励与终止

奖励（每项在 [0,1]，乘权重；仿真和想象同一公式）：

| 项 | 权重 | 尺度 |
|---|---|---|
| 官方 6 项跟踪 + 动作变化率 + 关节限位 | 不变 | 不变 |
| 物体位置（世界系） | 1.0 | 0.15 m |
| 物体姿态 | 1.0 | 0.4 rad |
| 物体线速度 / 角速度 | 0.5 / 0.5 | 0.5 m/s / 1.5 rad/s |
| 手-物相对位置（物体系） | 1.0 | 0.1 m |
| 接触接近（参考接触=1 的手） | 1.0 | 0.03 m |
| 穿透惩罚 | −20 /m | — |

终止（仿真硬阈值，想象里同一误差过 sigmoid）：锚点高度 0.25 m、倾斜 0.8（官方）；
骨盆系下物体位置误差（参考接触时 0.25 m，否则 0.5 m）；物体姿态误差 0.8 rad（参考接触时）。

## 7. 策略输入与训练（阶段 4）

- 目标窗口：官方 10 帧 × 间隔 5。每帧：关节位置和速度、锚点相对位置和姿态、物体参考位姿（当前 root 系）、
  物体参考速度、左右手接触标签和有效位；整窗口再加一次当前物体误差。actor 与 critic 用同一份。
- 隐变量 32 → 64。actor 额外上下文（one-hot + 尺寸）直接拼进 decoder，不归一化。
- 回报采样：每段 clip 内按 1 s 分箱，用"帧 → 箱"查找表；参数不变。
- EMA actor（衰减 0.999），只用于评估和导出。

## 8. 阶段与验收

| 阶段 | 内容 | 验收 |
|---|---|---|
| 0 | clone mjlab_research；手脚凸块 | 已完成（`assets/g1_collision`） |
| 1 | 数据打包和物体资产 | 打包目录能被 mjlab `MotionLoader` 读出；每个变体有凸块、SDF、惯量 |
| 2 | HOI 仿真任务 | 运动学回放奖励接近满分、不触发终止；重置不炸 |
| 3 | HOI 表示 | 表示/还原一致；零预测积分正确；想象奖励与仿真奖励在同一状态上相等 |
| 4 | 训练接线 | 单物体 pilot 能稳定训练，单轮时间可接受 |

## 9. 待验证

- solref 0.01 在 5 ms 步长、真实机器人负载下的稳定性。