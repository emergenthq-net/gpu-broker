"""CHANGELOG.md: every released version has an entry, and scripts/changelog_entry.py extracts it."""
import importlib.util
import tomllib

from tests.helpers import ROOT

spec = importlib.util.spec_from_file_location("changelog_entry", ROOT / "scripts/changelog_entry.py")
changelog_entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(changelog_entry)

TEXT = """# Changelog

## [Unreleased]

### Added
- next

## [1.1.0] - 2026-01-02

### Fixed
- a fix

## [1.0.0] - 2026-01-01

### Added
- first

[Unreleased]: https://example.com/compare/v1.1.0...HEAD
[1.1.0]: https://example.com/compare/v1.0.0...v1.1.0
"""


def test_an_entry_runs_to_the_next_version_and_stops_before_the_links():
    assert changelog_entry.entry(TEXT, "1.1.0") == "### Fixed\n- a fix\n"
    assert changelog_entry.entry(TEXT, "1.0.0") == "### Added\n- first\n"


def test_unreleased_and_unknown_versions_have_no_entry():
    assert changelog_entry.entry(TEXT, "Unreleased") is None
    assert changelog_entry.entry(TEXT, "2.0.0") is None


def test_the_packaged_version_has_an_entry():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert changelog_entry.entry((ROOT / "CHANGELOG.md").read_text(), version), \
        f"CHANGELOG.md needs a `## [{version}] - YYYY-MM-DD` entry"


def test_the_cli_prints_the_entry_or_fails(capsys):
    assert changelog_entry.main(["v0.3.2"]) == 0
    assert "self-testing" in capsys.readouterr().out
    assert changelog_entry.main(["9.9.9"]) == 1
