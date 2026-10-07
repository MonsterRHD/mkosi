# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Build generations for multi-image dependency graphs.

A generation freezes the selected dependency graph (its post-configure Config objects and its
repository metadata / package inputs) after the configure scripts have run. Every node builds in
topological order and publishes its image, manifest, checksums and associated outputs into an
isolated per-generation view below the output directory, so that nodes only ever consume candidate
outputs of the generation they belong to. Only once every node of the graph has been built
successfully are the candidate outputs committed to the real output directory in one journaled
step, followed by publishing the pending build history and running the auto-bump.

If any node fails, is cancelled or the generation is cleaned, the outputs of the previous complete
graph remain in place in the real output directory: candidate outputs are never picked up as
default inputs. The persisted generation state allows the next invocation to resume the nodes that
were not finished (or to discard the candidates when the frozen inputs changed) without mistaking
candidate outputs in the isolated view for an independently successful build based on "output
exists" checks.
"""

import contextlib
import dataclasses
import errno
import fcntl
import hashlib
import json
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional, cast

from mkosi.config import (
    Args,
    Config,
    ConfigTree,
    JsonEncoder,
    finalize_historydirs,
    promote_pending_history,
)
from mkosi.log import die
from mkosi.run import SandboxProtocol, nosandbox
from mkosi.tree import move_tree, rmtree
from mkosi.util import flock

STATE_VERSION = 1

NODE_PENDING = "pending"
NODE_BUILDING = "building"
NODE_DONE = "done"

PHASE_BUILDING = "building"
PHASE_COMMITTING = "committing"

_PRIVATE = ".mkosi-private"
_GENERATIONS = "generations"


def generations_dir(output_dir: Path) -> Path:
    return output_dir / _PRIVATE / _GENERATIONS


def finalize_generations_dirs(args: Args, output_dir: Optional[Path] = None) -> list[Path]:
    # Mirrors finalize_historydirs(): generations live next to the history in both the config
    # directory and, when configured, the output directory.
    return [hd.parent / _GENERATIONS for hd in finalize_historydirs(args, output_dir)]


def any_pending_generation(output_dirs: Sequence[Path]) -> bool:
    for d in output_dirs:
        pending = generations_dir(d) / "pending"
        if pending.is_dir() and any(pending.iterdir()):
            return True
    return False


def pending_generation_for_images(output_dirs: Sequence[Path], names: set[str]) -> bool:
    # Whether any pending (uncommitted) generation shares a node with the given image set.
    if not names:
        return False

    for d in output_dirs:
        pending = generations_dir(d) / "pending"
        if not pending.is_dir():
            continue
        for link in pending.iterdir():
            other = _pending_generation_dir(link)
            if other is not None and bool(_state_node_names(other / "state.json") & names):
                return True
    return False


def discard_all_generations(args: Args, output_dir: Optional[Path] = None) -> None:
    # Used by "mkosi clean": remove every generation (including candidate outputs and state) in both
    # possible locations.
    for d in finalize_generations_dirs(args, output_dir):
        if d.exists():
            rmtree(d)


@dataclasses.dataclass(frozen=True)
class NodeSpec:
    # A node of the dependency graph as frozen right after the configure scripts ran.
    name: str
    deps: tuple[str, ...]
    config: Config
    # Whether the output removal / clean script semantics of run_clean() apply to this node.
    clean_scripts: bool

    @property
    def output_names(self) -> set[str]:
        return set(self.config.outputs)


@dataclasses.dataclass
class NodeState:
    name: str
    deps: list[str]
    config: dict[str, Any]
    status: str = NODE_PENDING
    # The actual file names published from the staging directory once the node is done.
    outputs: list[str] = dataclasses.field(default_factory=list)
    clean_scripts: bool = False


class Generation:
    def __init__(self, output_dir: Path, gid: str, *, sandbox: SandboxProtocol = nosandbox) -> None:
        # Resolve the output directory so that paths finalized during configuration parsing (which
        # are resolved themselves) compare correctly against the generation root.
        self.root = output_dir.absolute().resolve()
        self.gid = gid
        self.sandbox = sandbox
        self.phase = PHASE_BUILDING
        self.order: list[str] = []
        self.nodes: dict[str, NodeState] = {}
        self.inputs: Optional[dict[str, str]] = None
        self.journal: set[tuple[str, str, str]] = set()
        self._lock_fd: Optional[int] = None
        self._closed = False

    @property
    def base(self) -> Path:
        return generations_dir(self.root)

    @property
    def dir(self) -> Path:
        return self.base / f"gen-{self.gid}"

    @property
    def view(self) -> Path:
        return self.dir / "view"

    @property
    def packages(self) -> Path:
        return self.dir / "packages"

    @property
    def backup(self) -> Path:
        return self.dir / "backup"

    @property
    def state_path(self) -> Path:
        return self.dir / "state.json"

    @property
    def lock_path(self) -> Path:
        return self.dir / "lock"

    @property
    def pending_link(self) -> Path:
        return self.base / "pending" / self.gid

    @property
    def committing(self) -> bool:
        return self.phase == PHASE_COMMITTING

    def config(self, name: str) -> Config:
        return Config.from_json(self.nodes[name].config)

    def is_done(self, name: str) -> bool:
        return self.nodes[name].status == NODE_DONE and self._candidates_present(name)

    def needs_build(self, name: str) -> bool:
        return not self.is_done(name)

    def _candidates_present(self, name: str) -> bool:
        node = self.nodes[name]
        return all((self.view / o).exists() or (self.view / o).is_symlink() for o in node.outputs)

    # ------------------------------------------------------------------ state

    def _load(self) -> None:
        j = json.loads(self.state_path.read_text())
        if j.get("version") != STATE_VERSION:
            die(
                f"Unknown build generation state version {j.get('version')} in {self.state_path}",
                hint="Run 'mkosi clean' to remove stale build generation state",
            )
        if j.get("id") != self.gid:
            die(f"Build generation state in {self.dir} has mismatched id")

        self.phase = j["phase"]
        self.order = list(j["order"])
        self.inputs = j.get("inputs")
        self.nodes = {
            n["name"]: NodeState(
                name=n["name"],
                deps=list(n["deps"]),
                config=n["config"],
                status=n["status"],
                outputs=list(n.get("outputs", ())),
                clean_scripts=bool(n.get("clean_scripts", False)),
            )
            for n in j["nodes"]
        }
        self.journal = {tuple(e) for e in j.get("journal", ())}

    def _save(self) -> None:
        tmp = self.state_path.with_name(".state.json.tmp")
        payload = {
            "version": STATE_VERSION,
            "id": self.gid,
            "phase": self.phase,
            "order": self.order,
            "nodes": [dataclasses.asdict(self.nodes[name]) for name in self.order],
            "inputs": self.inputs,
            "journal": sorted(self.journal),
        }
        tmp.write_text(json.dumps(payload, indent=4, sort_keys=True))
        os.replace(tmp, self.state_path)

        fd = os.open(self.dir, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    # ------------------------------------------------------------------ locks

    def _acquire_build_lock(self) -> None:
        self.lock_path.touch(mode=0o644, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(fd)
            if e.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise e
            die(
                f"Build generation {self.gid} is already being built by another process",
                hint="Wait for the other mkosi process to finish or run 'mkosi clean' to remove it",
            )
        self._lock_fd = fd

    def _release_build_lock(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    def close(self) -> None:
        if not self._closed:
            self._release_build_lock()
            self._closed = True

    def __enter__(self) -> "Generation":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------ begin

    @classmethod
    def begin(
        cls,
        output_dir: Path,
        specs: Sequence[NodeSpec],
        *,
        sandbox: SandboxProtocol = nosandbox,
    ) -> "Generation":
        root = output_dir.absolute().resolve()
        gid = _graph_id(specs)
        base = generations_dir(root)
        (base / "pending").mkdir(parents=True, exist_ok=True)
        (base / "lock").touch(mode=0o644, exist_ok=True)

        with flock(base / "lock"):
            gendir = base / f"gen-{gid}"

            if (gendir / "state.json").exists():
                gen = cls(root, gid, sandbox=sandbox)
                gen._load()
                gen._acquire_build_lock()
                gen._resume(specs)
                return gen

            # The directory exists but has no state (e.g. a previous process was killed during
            # creation or during the final cleanup): discard and recreate.
            if gendir.exists():
                rmtree(gendir, sandbox=sandbox)
                with contextlib.suppress(FileNotFoundError):
                    (base / "pending" / gid).unlink()

            # Retire pending generations of a previous, interrupted attempt that share nodes with
            # the graph we're about to build. Only do so if their build lock can be acquired (i.e.
            # no other mkosi process is still building them); disjoint graphs are left untouched so
            # they can be built and committed independently.
            names = {spec.name for spec in specs}
            for link in sorted((base / "pending").iterdir()):
                other = _pending_generation_dir(link)
                if other is None or not (other / "state.json").exists():
                    with contextlib.suppress(OSError):
                        link.unlink()
                    if other is not None and other.exists():
                        rmtree(other, sandbox=sandbox)
                    continue

                other_names = _state_node_names(other / "state.json")
                if not (other_names & names):
                    continue

                try:
                    other_lock = os.open(other / "lock", os.O_RDONLY | os.O_CLOEXEC | os.O_CREAT, 0o644)
                    try:
                        fcntl.flock(other_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    finally:
                        os.close(other_lock)
                except OSError as e:
                    if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                        shared = ",".join(sorted(other_names & names))
                        die(
                            f"Another mkosi process is still building one of these images: {shared}",
                            hint=(
                                "Wait for the other mkosi process to finish or run 'mkosi clean' "
                                "in its directory"
                            ),
                        )
                    raise e

                with contextlib.suppress(FileNotFoundError):
                    link.unlink()
                rmtree(other, sandbox=sandbox)

            gen = cls(root, gid, sandbox=sandbox)
            gen._init_fresh(specs)

        gen._acquire_build_lock()
        return gen

    def _init_fresh(self, specs: Sequence[NodeSpec]) -> None:
        self.view.mkdir(parents=True)
        self.packages.mkdir()
        self.backup.mkdir()

        seen: dict[str, str] = {}
        for spec in specs:
            for output in spec.output_names:
                if output in seen:
                    die(
                        f"Images {seen[output]} and {spec.name} both generate output '{output}'",
                        hint=(
                            "Configure distinct Output= names for images built in the same output directory"
                        ),
                    )
                seen[output] = spec.name

            self.order.append(spec.name)
            self.nodes[spec.name] = NodeState(
                name=spec.name,
                deps=list(spec.deps),
                config=_config_json(spec.config),
                clean_scripts=spec.clean_scripts,
            )

        self._project_view()
        self._save()
        os.symlink(self.dir, self.pending_link)

    def _resume(self, specs: Sequence[NodeSpec]) -> None:
        names = [spec.name for spec in specs]
        if set(names) != set(self.order) or len(names) != len(set(names)):
            die(
                f"Cannot resume build generation {self.gid}: the selected image set changed",
                hint="Run 'mkosi clean' or rebuild with --force",
            )

        by_name = {spec.name: spec for spec in specs}
        for name in self.order:
            if _canonical_config(self.nodes[name].config) != _canonical_config(by_name[name].config):
                die(
                    f"Cannot resume generation {self.gid}: frozen configuration of image {name} changed",
                    hint="Rebuild with --force to start a new generation",
                )

        # Use the post-configure topological order of the current invocation.
        self.order = names

        # Drop done nodes whose candidate outputs are no longer complete (e.g. the process was
        # killed while publishing the node).
        for name in self.order:
            node = self.nodes[name]
            if node.status == NODE_DONE and not self._candidates_present(name):
                logging.info(
                    f"Resuming build generation {self.gid}: candidate outputs of image {name} are "
                    "incomplete, rebuilding the image"
                )
                self._discard_node_candidates(name)
                node.status = NODE_PENDING
                node.outputs = []

        self._project_view()
        self._save()

    # ------------------------------------------------------------------ view

    def _node_output_names(self) -> set[str]:
        names: set[str] = set()
        for node in self.nodes.values():
            names |= set(Config.from_json(node.config).outputs)
        return names

    def _project_view(self) -> None:
        # The view is the generation's isolated "output directory": candidate outputs of the
        # generation's nodes appear here as they are built, while every other entry of the real
        # output directory is exposed through a (read-only, following) symlink as an external input.
        self.view.mkdir(parents=True, exist_ok=True)
        owned = self._node_output_names()

        if self.root.exists():
            for entry in self.root.iterdir():
                if entry.name == _PRIVATE or entry.name in owned:
                    continue
                link = self.view / entry.name
                target = entry.absolute()
                if link.is_symlink() and os.readlink(link) == os.fspath(target):
                    continue
                if link.is_symlink() or link.exists():
                    # The view is private to the generation, the only entries we did not create are
                    # leftovers from a previous attempt, remove them.
                    rmtree(link, sandbox=self.sandbox)
                os.symlink(target, link)

        # Prune external symlinks whose source disappeared from the real output directory.
        for link in self.view.iterdir():
            if link.is_symlink() and not link.exists() and link.name not in owned:
                with contextlib.suppress(FileNotFoundError):
                    link.unlink()

    def view_path(self, path: Path) -> Path:
        # Map a path inside the real output directory to the corresponding path in the generation
        # view; paths outside the output directory are returned unchanged. Configuration paths are
        # resolved during parsing, but compare both forms so symlinked path prefixes (e.g. the
        # output directory itself being reached through a symlink) are handled as well.
        path = path.absolute()
        if path.is_relative_to(self.root):
            return self.view / path.relative_to(self.root)

        with contextlib.suppress(OSError):
            resolved = path.resolve()
            if resolved.is_relative_to(self.root):
                return self.view / resolved.relative_to(self.root)

        return path

    def remap_config(self, config: Config) -> Config:
        # Build a Config that publishes into the generation view and reads dependency inputs from
        # it. Only the dependency input surfaces documented in the man page are remapped; the tools
        # tree (which is built and cached independently of the image graph) is left untouched.
        return dataclasses.replace(
            config,
            output_dir=self.view,
            base_trees=[self.view_path(p) for p in config.base_trees],
            initrds=[self.view_path(p) for p in config.initrds],
            extra_trees=[
                ConfigTree(self.view_path(tree.source), tree.target) for tree in config.extra_trees
            ],
        )

    def _discard_node_candidates(self, name: str) -> None:
        stale = {
            self.view / output
            for output in self.config(name).outputs
            if (self.view / output).exists() or (self.view / output).is_symlink()
        }
        if stale:
            rmtree(*stale, sandbox=self.sandbox)

    # ------------------------------------------------------------------ inputs

    def freeze_inputs(self, inputs: dict[str, str]) -> None:
        if self.inputs is None:
            self.inputs = dict(inputs)
            self._save()
            return

        if self.inputs != inputs:
            logging.info(
                f"Repository metadata and package inputs of build generation {self.gid} changed since "
                "the interrupted build, discarding candidate outputs and rebuilding the graph"
            )
            for name in self.order:
                node = self.nodes[name]
                if node.status != NODE_PENDING:
                    self._discard_node_candidates(name)
                node.status = NODE_PENDING
                node.outputs = []
            self.inputs = dict(inputs)
            self._save()

    # ------------------------------------------------------------------ build

    def mark_building(self, name: str) -> None:
        # Drop candidate leftovers of a previous (interrupted) attempt to build this node so that
        # they cannot be mistaken for freshly published outputs.
        self._discard_node_candidates(name)
        self.nodes[name].status = NODE_BUILDING
        self.nodes[name].outputs = []
        self._save()

    def mark_done(self, name: str, outputs: Sequence[str]) -> None:
        node = self.nodes[name]
        missing = [o for o in outputs if not (self.view / o).exists() and not (self.view / o).is_symlink()]
        if missing:
            die(f"Internal error: candidate outputs of image {name} are missing: {', '.join(missing)}")
        node.status = NODE_DONE
        node.outputs = list(outputs)
        self._save()

    def all_done(self) -> bool:
        return all(node.status == NODE_DONE for node in self.nodes.values())

    # ------------------------------------------------------------------ commit

    def commit(
        self,
        *,
        run_clean_scripts: Callable[[Config], None],
        historydirs: Sequence[Path],
    ) -> None:
        # Forward-recovery journaled commit: every step is recorded durably before moving on, so a
        # crash during the commit is resumed to completion by the next invocation instead of leaving
        # the output directory in a mixed state.
        try:
            self.phase = PHASE_COMMITTING
            self._save()

            for name in self.order:
                node = self.nodes[name]
                config = self.config(name)

                clean_key = (name, "clean", "")
                if node.clean_scripts and clean_key not in self.journal:
                    # Clean scripts are run against the real output directory right before the new
                    # candidate outputs take the node's place, replacing the previous "remove all
                    # outputs before the build" behavior for generation builds.
                    run_clean_scripts(config)
                    self.journal.add(clean_key)
                    self._save()

                node_backup = self.backup / name

                old: set[Path] = {
                    self.root / output
                    for output in config.outputs
                    if (self.root / output).exists() or (self.root / output).is_symlink()
                }
                # Resolve the "output" symlink we create ourselves so that a format or compression
                # change does not leave its old target behind, but never follow links pointing
                # outside of the output directory.
                old |= {
                    p.resolve() for p in set(old) if p.is_symlink() and p.resolve().is_relative_to(self.root)
                }

                for path in sorted(old):
                    key = (name, "backup", path.name)
                    target = node_backup / path.name
                    if key not in self.journal:
                        # Tolerate a crash between the rename and journaling the step.
                        already_backed_up = target.exists() or target.is_symlink()
                        still_in_place = path.exists() or path.is_symlink()
                        if not already_backed_up or still_in_place:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            if already_backed_up:
                                rmtree(target, sandbox=self.sandbox)
                            os.rename(path, target)
                        self.journal.add(key)
                        self._save()

                for output in node.outputs:
                    key = (name, "publish", output)
                    if key not in self.journal:
                        candidate = self.view / output
                        target = self.root / output
                        # Tolerate a crash between moving the candidate into place and journaling
                        # the step: rename() removes it from the view, so if the target exists and
                        # the candidate does not, the move already completed.
                        if (target.exists() or target.is_symlink()) and not (
                            candidate.exists() or candidate.is_symlink()
                        ):
                            pass
                        else:
                            move_tree(
                                candidate,
                                self.root,
                                use_subvolumes=config.use_subvolumes,
                                sandbox=self.sandbox,
                            )
                        self.journal.add(key)
                        self._save()

                if node_backup.exists():
                    rmtree(node_backup, sandbox=self.sandbox)

            history_key = ("", "history", "")
            if history_key not in self.journal:
                promote_pending_history(historydirs)
                self.journal.add(history_key)
                self._save()

            with contextlib.suppress(FileNotFoundError):
                self.pending_link.unlink()

            (self.base / "last-committed").write_text(self.gid)

            self._release_build_lock()
            rmtree(self.dir, sandbox=self.sandbox)
            self._closed = True
        except BaseException:
            # Make sure the state is on disk before propagating the exception.
            if not self._closed:
                with contextlib.suppress(OSError):
                    self._save()
            raise

    # ------------------------------------------------------------------ discard

    def discard(self) -> None:
        # Abandon an uncommitted generation. The real output directory is never touched, so the
        # outputs of the previous complete graph remain the default inputs.
        self.close()
        with contextlib.suppress(FileNotFoundError):
            self.pending_link.unlink()
        if self.dir.exists():
            rmtree(self.dir, sandbox=self.sandbox)
        self._closed = True


def _config_json(config: Config) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(json.dumps(config.to_dict(), cls=JsonEncoder)),
    )


def _canonical_config(config: Any) -> str:
    if isinstance(config, Config):
        config = _config_json(config)
    return json.dumps(config, cls=JsonEncoder, sort_keys=True)


def _graph_id(specs: Sequence[NodeSpec]) -> str:
    # The generation identity is the frozen dependency graph (per-node post-configure Config and
    # dependency edges). --force does not participate in the identity: a failed or interrupted
    # build is always resumed from its persisted generation; use 'mkosi clean' to discard candidate
    # outputs and start over.
    payload = [
        {
            "name": spec.name,
            "dependencies": sorted(spec.deps),
            "config": _config_json(spec.config),
        }
        for spec in specs
    ]
    return hashlib.sha256(
        json.dumps(payload, cls=JsonEncoder, sort_keys=True).encode()
    ).hexdigest()[:16]


def _pending_generation_dir(link: Path) -> Optional[Path]:
    if not link.is_symlink():
        return None
    try:
        target = os.readlink(link)
    except OSError:
        return None
    p = Path(target)
    return p if p.is_absolute() else (link.parent / p).resolve()


def _state_node_names(state_path: Path) -> set[str]:
    try:
        j = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    return {node["name"] for node in j.get("nodes", ())}
