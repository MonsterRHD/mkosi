# SPDX-License-Identifier: LGPL-2.1-or-later

# Integration tests for the multi-image build generation logic: a base subimage used as the base
# tree of the main image, a failure halfway through the graph, resuming the interrupted generation,
# and auto-bump gating.

from pathlib import Path

import pytest

from . import Image, ImageConfig

pytestmark = pytest.mark.integration


MAIN_CONF = """\
[Distribution]
Distribution={distribution}
Release={release}

[Config]
Dependencies=base

[Output]
Format=directory
Output=main

[Content]
BaseTrees=%O/base
"""

BASE_CONF = """\
[Output]
Format=directory
Output=base
"""

FINALIZE = """\
#!/bin/sh
set -e
# Fail the main image as long as the marker file exists in the source directory. The script itself
# (and hence the frozen configuration) does not change between the failing and the resumed run.
if [ -e "$SRCDIR/.fail-main" ]; then
    exit 1
fi
"""


def test_build_generation_failure_resume_and_auto_bump(
    config: ImageConfig,
    tmp_path: Path,
) -> None:
    configdir = tmp_path / "config"
    (configdir / "mkosi.images" / "base").mkdir(parents=True)
    (configdir / "mkosi.images" / "base" / "mkosi.conf").write_text(BASE_CONF)
    (configdir / "mkosi.conf").write_text(
        MAIN_CONF.format(distribution=config.distribution, release=config.release)
    )
    (configdir / "mkosi.version").write_text("1\n")

    finalize = configdir / "mkosi.finalize"
    finalize.write_text(FINALIZE)
    finalize.chmod(0o755)

    with Image(config) as image:
        options = [
            "--directory", configdir,
            "--output-directory", image.output_dir,
            "--incremental=no",
            "--auto-bump",
        ]  # fmt: skip

        # Build the first complete generation.
        image.mkosi("build", options)
        base = image.output_dir / "base"
        main = image.output_dir / "main"
        assert base.is_dir()
        assert main.is_dir()
        assert (configdir / "mkosi.version").read_text().strip() == "2"

        # Fail the main node of the second generation. The base node already published its
        # candidate, but it must not become visible in the real output directory.
        (configdir / ".fail-main").touch()
        before = base.stat().st_mtime_ns, main.stat().st_mtime_ns
        result = image.mkosi("build", options, check=False)
        assert result.returncode != 0

        # The previous complete graph is untouched, the version is not bumped and the interrupted
        # generation is persisted for resume.
        assert (base.stat().st_mtime_ns, main.stat().st_mtime_ns) == before
        assert (configdir / "mkosi.version").read_text().strip() == "2"
        pending = image.output_dir / ".mkosi-private" / "generations" / "pending"
        assert pending.is_dir() and any(pending.iterdir())

        # Resume the same generation without --force: the base candidate is reused, the main node
        # is rebuilt and the generation is committed.
        (configdir / ".fail-main").unlink()
        image.mkosi("build", options)

        assert not any(pending.iterdir())
        assert main.stat().st_mtime_ns != before[1]
        assert main.is_dir()
        # The auto-bump is only performed once the whole graph committed successfully.
        assert (configdir / "mkosi.version").read_text().strip() == "3"
