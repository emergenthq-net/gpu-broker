"""Print one version's CHANGELOG.md entry: the GitHub release notes for that version.

    python scripts/changelog_entry.py 0.3.2 > notes.md
    gh release create v0.3.2 --title "gpu-broker 0.3.2" --notes-file notes.md

Exits 1 when CHANGELOG.md has no `## [<version>] - <date>` heading, so a release without an
entry fails before anything is published.
"""
from __future__ import annotations

import pathlib
import re
import sys

CHANGELOG = pathlib.Path(__file__).resolve().parent.parent / "CHANGELOG.md"
HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\]( - \d{4}-\d{2}-\d{2})?$", re.M)
LINK = re.compile(r"^\[[^\]]+\]: ", re.M)   # the compare links at the bottom


def entry(text: str, version: str) -> str | None:
    """The body under `## [version] - date`, up to the next version heading or the link list."""
    heads = list(HEADING.finditer(text))
    for i, h in enumerate(heads):
        if h.group("version") == version and h.group(2):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            body = text[h.end():end]
            if (link := LINK.search(body)) is not None:
                body = body[:link.start()]
            return body.strip() + "\n"
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    version = argv[0].removeprefix("v")
    body = entry(CHANGELOG.read_text(), version)
    if body is None:
        print(f"CHANGELOG.md has no `## [{version}] - YYYY-MM-DD` entry", file=sys.stderr)
        return 1
    sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
