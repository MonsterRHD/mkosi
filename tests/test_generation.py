# SPDX-License-Identifier: LGPL-2.1-or-later

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Optional

import pytest

from mkosi.config import Args, Config, ConfigTree, OutputFormat
from mkosi.generation import (
    NODE_DONE,
    NODE_PENDING,
    Generation,
    NodeSpec,
    any_pending_generation,
    discard_all_generations,
    generations_dir,
)


def make_config(name: str, output_dir: Path, *, output: Optional[str] = None, **kw: object) -> Config:
    return dataclasses.replace(
        Config.default(),
        image=name,
        output=output or name,
        output_dir=output_dir,
        output_format=OutputFormat.directory,
        **kw,
    )


def make_spec(config: Config, deps: tuple[str, ...] = (), *, clean_scripts: bool = False) -> NodeSpec:
    return NodeSpec(
        name=config.image,
        deps=deps,
        config=config,
        clean_scripts=clean_scripts,
    )


def write_candidate(gen: Generation, name: str, *, content: str = "new") -> None:
    # Simulate a published directory output for a node in the generation view.
    p = gen.view / name
    p.mkdir(parents=True, exist_ok=True)
    (p / "payload").write_text(content)


def test_generation_begin_projects_view(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    # An unrelated entry of the output directory (external input) and a previous-generation output
    # of the "base" node.
    (root / "external.txt").write_text("external")
    (root / "base").mkdir()
    (root / "base" / "old").write_text("old")

    gen = Generation.begin(root, [make_spec(make_config("base", root))])
    try:
        assert gen.pending_link.is_symlink()
        assert gen.packages.is_dir()

        # External entries are exposed through read-through symlinks...
        external = gen.view / "external.txt"
        assert external.is_symlink()
        assert external.read_text() == "external"

        # ...while node-owned names are hidden so nodes can only consume this generation's
        # candidates.
        assert not (gen.view / "base").exists()
        assert gen.needs_build("base")
    finally:
        gen.close()


def test_generation_resume_reuses_completed_candidates(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()

    specs = [
        make_spec(make_config("base", root)),
        make_spec(make_config("main", root), deps=("base",)),
    ]

    gen = Generation.begin(root, specs)
    try:
        gen.freeze_inputs({"meta": "1"})
        gen.mark_building("base")
        write_candidate(gen, "base")
        gen.mark_done("base", ["base"])
        assert gen.is_done("base")
        assert not gen.is_done("main")
    finally:
        gen.close()

    gen = Generation.begin(root, specs)
    try:
        assert gen.is_done("base")
        assert gen.needs_build("main")
        # The candidate is reused from the isolated view, not from the real output directory.
        assert (gen.view / "base" / "payload").read_text() == "new"
        assert not (root / "base").exists()
    finally:
        gen.close()


def test_generation_resume_rebuilds_incomplete_candidate(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    specs = [make_spec(make_config("base", root))]

    gen = Generation.begin(root, specs)
    try:
        gen.freeze_inputs({"meta": "1"})
        gen.mark_building("base")
        write_candidate(gen, "base")
        gen.mark_done("base", ["base"])
    finally:
        gen.close()

    # Simulate the process dying while the candidate was being published: the whole candidate
    # directory disappeared (finalize_staging publishes atomically, so it is either complete or
    # absent).
    shutil.rmtree(gen.view / "base")

    gen = Generation.begin(root, specs)
    try:
        assert gen.nodes["base"].status == NODE_PENDING
        assert gen.needs_build("base")
    finally:
        gen.close()


def test_generation_changed_graph_discards_intersecting(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    specs = [make_spec(make_config("base", root, image_version="1"))]

    gen = Generation.begin(root, specs)
    old_dir = gen.dir
    gen.mark_building("base")
    write_candidate(gen, "base")
    gen.mark_done("base", ["base"])
    gen.close()

    # Same node, different frozen configuration: the old pending generation is retired even though
    # it had a completed candidate.
    new_specs = [make_spec(make_config("base", root, image_version="2"))]
    gen = Generation.begin(root, new_specs)
    try:
        assert not old_dir.exists()
        assert gen.needs_build("base")
        assert not (gen.view / "base").exists()
    finally:
        gen.close()


def test_generation_disjoint_graph_is_left_alone(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()

    gen = Generation.begin(root, [make_spec(make_config("base", root))])
    base_dir = gen.dir
    gen.close()

    # An unrelated graph shares no nodes with the pending one.
    other = Generation.begin(root, [make_spec(make_config("other", root))])
    try:
        assert base_dir.exists()
        assert any_pending_generation([root])
    finally:
        other.close()


def test_generation_same_graph_resumes_after_force(tmp_path: Path) -> None:
    # --force does not change the generation identity: an interrupted build is resumed from its
    # persisted generation even when the next invocation uses --force. Candidates are only
    # discarded via 'mkosi clean' or by changing the frozen configuration.
    root = tmp_path / "out"
    root.mkdir()
    specs = [make_spec(make_config("base", root))]

    first = Generation.begin(root, specs)
    first_dir = first.dir
    first.close()

    second = Generation.begin(root, specs)
    try:
        assert second.gid == first.gid
        assert second.dir == first_dir
    finally:
        second.close()


def test_generation_inputs_change_resets_candidates(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    specs = [make_spec(make_config("base", root))]

    gen = Generation.begin(root, specs)
    try:
        gen.freeze_inputs({"meta": "1"})
        gen.mark_building("base")
        write_candidate(gen, "base")
        gen.mark_done("base", ["base"])
    finally:
        gen.close()

    gen = Generation.begin(root, specs)
    try:
        gen.freeze_inputs({"meta": "2"})
        assert gen.nodes["base"].status == NODE_PENDING
        assert gen.needs_build("base")
        assert not (gen.view / "base").exists()
    finally:
        gen.close()


def test_generation_duplicate_output_dies(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    specs = [
        make_spec(make_config("a", root, output="shared")),
        make_spec(make_config("b", root, output="shared")),
    ]
    with pytest.raises(SystemExit):
        Generation.begin(root, specs)


def test_generation_remap_config(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    external = tmp_path / "external-tree"
    external.mkdir()

    config = dataclasses.replace(
        Config.default(),
        image="main",
        output="main",
        output_dir=root,
        output_format=OutputFormat.directory,
        base_trees=[root / "base"],
        initrds=[root / "initrd.img"],
        extra_trees=[
            ConfigTree(root / "extra", None),
            ConfigTree(external, None),
        ],
    )
    gen = Generation.begin(root, [make_spec(config)])
    try:
        view_config = gen.remap_config(config)
        assert view_config.output_dir == gen.view
        assert view_config.base_trees == [gen.view / "base"]
        assert view_config.initrds == [gen.view / "initrd.img"]
        assert view_config.extra_trees[0].source == gen.view / "extra"
        assert view_config.extra_trees[1].source == external
        # The tools tree, which is not part of the image graph, is never remapped.
        assert view_config.tools_tree == config.tools_tree
    finally:
        gen.close()


def test_generation_commit_publishes_graph(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()

    # The previous complete graph.
    (root / "base").mkdir()
    (root / "base" / "old").write_text("old base")
    (root / "main").write_text("old main")
    (root / "external.txt").write_text("external")

    historydir = root / ".mkosi-private/history"
    historydir.mkdir(parents=True)
    (historydir / "pending.json").write_text(json.dumps({"Image": "main"}))

    specs = [
        make_spec(make_config("base", root), clean_scripts=True),
        make_spec(make_config("main", root), deps=("base",), clean_scripts=True),
    ]

    cleaned: list[str] = []
    gen = Generation.begin(root, specs)
    try:
        gen.freeze_inputs({"meta": "1"})
        for spec in specs:
            gen.mark_building(spec.name)
            write_candidate(gen, spec.name, content=f"new {spec.name}")
            gen.mark_done(spec.name, [spec.name])

        gen.commit(
            run_clean_scripts=lambda config: cleaned.append(config.image),
            historydirs=[historydir],
        )
    finally:
        gen.close()

    # Clean scripts run in topological order, right before each node's outputs take their place.
    assert cleaned == ["base", "main"]

    # Candidate outputs replaced the previous graph...
    assert (root / "base" / "payload").read_text() == "new base"
    assert not (root / "base" / "old").exists()
    assert (root / "main" / "payload").read_text() == "new main"

    # ...external entries are untouched...
    assert (root / "external.txt").read_text() == "external"

    # ...the history is published and the generation state is gone.
    assert json.loads((historydir / "latest.json").read_text()) == {"Image": "main"}
    assert not (historydir / "pending.json").exists()
    assert not any_pending_generation([root])
    assert (generations_dir(root) / "last-committed").read_text() == gen.gid


def test_generation_commit_is_resumed_after_crash(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    (root / "base").mkdir()
    (root / "main").write_text("old main")

    specs = [
        make_spec(make_config("base", root), clean_scripts=True),
        make_spec(make_config("main", root), deps=("base",), clean_scripts=True),
    ]

    calls = {"n": 0}

    def clean_scripts(config: Config) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("interrupted during commit")

    gen = Generation.begin(root, specs)
    try:
        gen.freeze_inputs({"meta": "1"})
        for spec in specs:
            gen.mark_building(spec.name)
            write_candidate(gen, spec.name)
            gen.mark_done(spec.name, [spec.name])

        with pytest.raises(RuntimeError, match="interrupted during commit"):
            gen.commit(run_clean_scripts=clean_scripts, historydirs=[])
    finally:
        gen.close()

    # The real output directory still contains the previous complete graph.
    assert (root / "base").is_dir()
    assert (root / "main").read_text() == "old main"
    assert any_pending_generation([root])

    resumed: list[str] = []

    gen = Generation.begin(root, specs)
    try:
        # The interrupted commit is rolled forward to completion.
        assert gen.committing
        gen.commit(
            run_clean_scripts=lambda config: resumed.append(config.image),
            historydirs=[],
        )
    finally:
        gen.close()

    assert resumed == ["base", "main"]
    assert (root / "base" / "payload").read_text() == "new base"
    assert (root / "main" / "payload").read_text() == "new main"
    assert not any_pending_generation([root])


def test_generation_node_status_constants() -> None:
    assert NODE_DONE == "done"
    assert NODE_PENDING == "pending"


def test_discard_all_generations(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    specs = [make_spec(make_config("base", root))]

    gen = Generation.begin(root, specs)
    gen.close()
    assert generations_dir(root).exists()

    discard_all_generations(Args.default(), root)
    assert not generations_dir(root).exists()
    assert not any_pending_generation([root])
