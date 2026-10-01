"""The input boundary: everything that can reach a subprocess or the filesystem."""
import pytest

from gpu_broker.drivers import check, validate
from gpu_broker.units import unit_ref

TRAVERSAL = ["../etc", "a/../../b", "/abs/path", "-rf", "a b", "a;b", "$(x)", "a\nb", ""]


def test_unit_specs():
    assert unit_ref("llama").key == "llama"
    assert unit_ref({"name": "llama", "target": 156}).key == "156:llama"
    assert unit_ref({"ct": 156, "name": "llama"}).key == "156:llama"
    for bad in [{"ct": 1}, {"name": "x", "target": "1;2"}, {"name": "x", "target": "../1"}, 5, *TRAVERSAL]:
        with pytest.raises(ValueError):
            unit_ref(bad)


def test_check_enforces_verbs_and_the_allowlist():
    assert check("a", "start", frozenset({"a"})).name == "a"
    with pytest.raises(ValueError):
        check("a", "restart", None)
    with pytest.raises(PermissionError):
        check("b", "start", frozenset({"a"}))


@pytest.mark.parametrize("kind,ref", [("hf", "org/repo"), ("gh", "https://github.com/org/repo")])
def test_good_downloads(kind, ref):
    validate.download(kind, ref, "my-model.v2", ["*.safetensors", "sub/*.json"])


@pytest.mark.parametrize("kind,ref,slug,include", [
    ("s3", "org/repo", "m", []),
    ("hf", "org/repo/extra", "m", []), ("hf", "../repo", "m", []), ("hf", "org/..", "m", []),
    ("gh", "http://github.com/o/r", "m", []), ("gh", "https://evil.com/o/r", "m", []),
    ("gh", "https://github.com/o/r --upload-pack=x", "m", []),
    ("hf", "org/repo", "../m", []), ("hf", "org/repo", "M", []), ("hf", "org/repo", "-m", []),
    ("hf", "org/repo", "m", ["../x"]), ("hf", "org/repo", "m", ["--force"]), ("hf", "org/repo", "m", ["a b"]),
])
def test_bad_downloads(kind, ref, slug, include):
    with pytest.raises(ValueError):
        validate.download(kind, ref, slug, include)


def test_links():
    validate.link("model/x.safetensors", "loras")
    for rel, sub in [("../x", "loras"), ("/x", "loras"), ("a/../../x", "loras"), ("x", "custom_nodes"), ("a b", "vae")]:
        with pytest.raises(ValueError):
            validate.link(rel, sub)


def test_inside_resolves_symlinks(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path)
    assert validate.inside(str(root), "a", "b").startswith(str(root.resolve()))
    for parts in (("..", "x"), ("escape", "x")):
        with pytest.raises(ValueError):
            validate.inside(str(root), *parts)
