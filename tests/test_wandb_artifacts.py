"""Checkpoint artifact logging: evaluated checkpoints (aliased) survive the pruning."""

from __future__ import annotations

from types import SimpleNamespace

from supermimic.utils import wandb_artifacts


class _Version:
  def __init__(self, index: int, aliases: list[str]):
    self.version, self.aliases, self.deleted = f"v{index}", aliases, False

  def delete(self):
    self.deleted = True


class _FakeWandb:
  """Stands in for the wandb module: a run that logs versions, an Api that lists them."""

  def __init__(self):
    self.versions: list[_Version] = []
    self.run = SimpleNamespace(
      id="abc", entity="e", project="p", log_artifact=self._log
    )
    fake = self

    class Artifact:
      def __init__(self, name, type):
        self.name, self.files = name, []

      def add_file(self, path):
        self.files.append(path)

    class Api:
      def artifacts(self, type_name, collection):
        return [v for v in fake.versions if not v.deleted]

    self.Artifact, self.Api = Artifact, Api

  def _log(self, artifact, aliases=None):
    self.versions.append(_Version(len(self.versions), list(aliases or [])))


def test_aliased_versions_survive_pruning(monkeypatch):
  fake = _FakeWandb()
  monkeypatch.setattr(wandb_artifacts, "wandb", fake)
  for i in range(40):
    aliases = (
      ["eval-2000"] if i == 8 else None
    )  # the evaluated checkpoint of iteration 2000
    wandb_artifacts.log_model_artifact([f"ckpt_{i}.pt"], aliases=aliases)
  kept = [int(v.version[1:]) for v in fake.versions if not v.deleted]
  assert 8 in kept and fake.versions[8].aliases == ["eval-2000"]
  assert kept == [0, 8, 25, 35, 36, 37, 38, 39]


def test_plain_versions_are_pruned_as_before(monkeypatch):
  fake = _FakeWandb()
  monkeypatch.setattr(wandb_artifacts, "wandb", fake)
  for i in range(12):
    wandb_artifacts.log_model_artifact([f"ckpt_{i}.pt"])
  assert [int(v.version[1:]) for v in fake.versions if not v.deleted] == [
    0,
    7,
    8,
    9,
    10,
    11,
  ]
