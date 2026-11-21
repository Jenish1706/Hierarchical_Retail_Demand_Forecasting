"""
Documentation structure: every relative link resolves, and the README's repository layout matches the tree.

These checks used to run from a private audit folder; they belong with the project so a moved file or a
renamed script cannot silently break the docs.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
README = ROOT / "README.md"
OPTIONAL_PATHS = {"data"}  # raw M5 data is git-ignored


def _markdown_files() -> list[Path]:
    return [README, *sorted((ROOT / "docs").rglob("*.md"))]


@pytest.mark.parametrize("md", _markdown_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_relative_link_resolves(md: Path) -> None:
    text = md.read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.S)  # code samples are not links
    broken = []
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = target.split("#", 1)[0]
        if path and not (md.parent / path).resolve().exists():
            broken.append(target)
    assert not broken, f"{md.relative_to(ROOT)}: broken links {broken}"


def _layout_block() -> str:
    text = README.read_text(encoding="utf-8")
    section = text.split("## Repository Layout")[1].split("\n---")[0]
    return re.search(r"```text\n(.*?)```", section, flags=re.S).group(1)


def test_readme_layout_lists_every_top_level_entry() -> None:
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    top = {f.split("/")[0] for f in tracked if f}
    block = _layout_block()
    missing = [name for name in sorted(top) if name not in block and name not in {"README.md", ".gitignore", ".dvcignore"}]
    assert not missing, f"top-level entries missing from the README layout: {missing}"


def test_readme_layout_paths_exist() -> None:
    missing = []
    # top-level entries are lines that start at column 0; only the leading token is a path
    for line in _layout_block().splitlines():
        if not line or line.startswith(" "):
            continue
        token = line.split()[0].rstrip("/")
        if "{" in token or token in OPTIONAL_PATHS or not re.fullmatch(r"[\w.]+(/[\w.]+)*", token):
            continue
        if not (ROOT / token).exists():
            missing.append(token)
    assert not missing, missing


def test_script_groups_in_the_layout_match_the_folders() -> None:
    block = _layout_block()
    for group in ("pipeline", "tools", "experiments"):
        folder = ROOT / "scripts" / group
        assert folder.is_dir()
        line = next(ln for ln in block.splitlines() if ln.strip().startswith(f"{group}/"))
        for script in sorted(p.stem for p in folder.glob("*.py")):
            if group != "experiments":  # experiments are described, not enumerated
                assert script in line, f"{script} missing from the layout line for scripts/{group}/"
    assert not list((ROOT / "scripts").glob("*.py")), "scripts must live in pipeline/, tools/ or experiments/"


def test_test_folders_are_packages_with_topic_names() -> None:
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        assert (path.parent / "__init__.py").exists(), f"{path.parent} is not a package"
        assert not re.search(r"test_(m5|g4b)_", path.name), f"{path.name} is named after a work item, not a topic"
