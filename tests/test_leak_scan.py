"""scripts/leak_scan.py: finds denylisted text in a tree and in built distributions, reports
where without repeating the text, and never reports clean without a usable denylist."""
import io
import tarfile
import zipfile

import pytest

from tests.helpers import load_leak_scan

scan = load_leak_scan()
SECRET = "Zorblax-Prime"   # a made-up private name; the real list is never in this repository


@pytest.fixture
def denylist(tmp_path):
    p = tmp_path / "deny.txt"
    p.write_text(f"# a comment line\n\n{SECRET.lower()}\n\\b10\\.9\\.8\\.[0-9]\n")
    return p


def tree(root, files):
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


def run(capsys, *argv):
    code = scan.main([str(a) for a in argv])
    out, err = capsys.readouterr()
    return code, out, err


def test_a_clean_tree_passes(tmp_path, denylist, capsys):
    src = tree(tmp_path / "src", {"a.py": "x = 1\n", "docs/b.md": "10.9.80.1 is not 10.9.8.x\n"})
    assert run(capsys, "--denylist", denylist, src)[0] == scan.CLEAN


def test_hits_name_the_file_line_and_entry_but_never_the_text(tmp_path, denylist, capsys):
    src = tree(tmp_path / "src", {"a.py": "ok\nhost = 'ZORBLAX-prime'\n", "n.txt": "see 10.9.8.7\n",
                                  f"{SECRET}/c.txt": "fine\n"})
    code, out, err = run(capsys, "--denylist", denylist, src)
    assert code == scan.HITS
    assert out.splitlines() == ["a.py:2: entry 1", "n.txt:1: entry 2", "Zorblax-Prime/c.txt:0: entry 1 (in the file name)"]
    body = out.replace(f"{SECRET}/c.txt", "")   # the file name is the location, not a repeat of matched text
    assert SECRET.lower() not in (body + err).lower() and "10.9.8.7" not in body + err


def test_built_distributions_are_scanned_member_by_member(tmp_path, denylist, capsys):
    whl = tmp_path / "pkg-1.0-py3-none-any.whl"
    with zipfile.ZipFile(whl, "w") as z:
        z.writestr("pkg/__init__.py", "")
        z.writestr("pkg/conf.py", f"# made for {SECRET}\n")
    sdist = tmp_path / "pkg-1.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as t:
        data = b"ok\naddr = '10.9.8.1'\n"
        info = tarfile.TarInfo("pkg-1.0/tests/t.py")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    code, out, _ = run(capsys, "--denylist", denylist, whl, sdist)
    assert code == scan.HITS
    assert out.splitlines() == [f"{whl.name}!pkg/conf.py:1: entry 1", f"{sdist.name}!pkg-1.0/tests/t.py:2: entry 2"]


def test_vcs_and_cache_folders_and_the_denylist_itself_are_not_scanned(tmp_path, capsys):
    src = tree(tmp_path / "src", {".git/config": SECRET, ".venv/x.py": SECRET, "pkg.egg-info/PKG-INFO": SECRET,
                                  "deny.txt": SECRET})
    assert run(capsys, "--denylist", src / "deny.txt", src)[0] == scan.CLEAN


@pytest.mark.parametrize("text", ["", "# only comments\n\n", "(unclosed\n"])
def test_an_empty_or_broken_denylist_is_never_clean(tmp_path, capsys, text):
    (tmp_path / "deny.txt").write_text(text)
    src = tree(tmp_path / "src", {"a.py": SECRET})
    assert run(capsys, "--denylist", tmp_path / "deny.txt", src)[0] == scan.UNUSABLE


def test_without_a_denylist_it_refuses_rather_than_passing(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(scan.ENV, raising=False)
    code, _, err = run(capsys, tmp_path)
    assert code == scan.UNUSABLE and scan.ENV in err
    monkeypatch.setenv(scan.ENV, str(tmp_path / "missing.txt"))
    assert run(capsys, tmp_path)[0] == scan.UNUSABLE


def test_the_environment_names_the_denylist(tmp_path, denylist, capsys, monkeypatch):
    monkeypatch.setenv(scan.ENV, str(denylist))
    assert scan.from_env() is not None
    assert run(capsys, tree(tmp_path / "src", {"a.py": SECRET}))[0] == scan.HITS
    monkeypatch.delenv(scan.ENV)
    assert scan.from_env() is None
