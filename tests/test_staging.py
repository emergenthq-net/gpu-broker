"""Staged input files: names from the job id, private files, upload, hand-off, cleanup."""
import stat

from gpu_broker.media import InputFile
from gpu_broker.staging import Staging

IMG = InputFile("image", b"png-bytes", "png", "inline")
END = InputFile("end_image", b"jpg-bytes", "jpeg", "url")
F0, F1 = InputFile("frames", b"f0", "png", "inline", 0), InputFile("frames", b"f1", "webp", "inline", 1)
VID = InputFile("video", b"mp4", "mp4", "inline")


def test_put_writes_private_files_named_from_the_job(tmp_path):
    s = Staging(str(tmp_path / "in"))
    s.put("abc123", [IMG, END, F0, VID])
    files = sorted(p.name for p in (tmp_path / "in").iterdir())
    assert files == ["abc123-end_image.jpg", "abc123-frames-00.png", "abc123-image.png", "abc123-video.mp4"]
    assert stat.S_IMODE((tmp_path / "in" / "abc123-image.png").stat().st_mode) == 0o600


def test_put_with_no_files_creates_nothing(tmp_path):
    Staging(str(tmp_path / "in")).put("abc123", [])
    assert not (tmp_path / "in").exists()


def test_upload_sends_only_the_single_images_under_unique_names(tmp_path):
    s, sent = Staging(str(tmp_path)), []
    s.put("j1", [IMG, END, F0, VID])
    s.put("j2", [IMG])
    names = s.upload("j1", lambda name, data, kind: sent.append((name, data, kind)) or "sub/" + name)
    assert names == {"image": "sub/broker-j1-image.png", "end_image": "sub/broker-j1-end_image.jpg"}
    assert sorted(sent) == [("broker-j1-end_image.jpg", b"jpg-bytes", "jpeg"), ("broker-j1-image.png", b"png-bytes", "png")]


def test_files_hands_over_one_jobs_files_in_name_order(tmp_path):
    s = Staging(str(tmp_path))
    s.put("j1", [F1, VID, F0])
    s.put("j2", [IMG])
    assert [(n, p.read_bytes()) for n, p in s.paths("j1")] == [
        ("frames-00.png", b"f0"), ("frames-01.webp", b"f1"), ("video.mp4", b"mp4")]
    assert Staging(str(tmp_path / "none")).paths("j1") == []


J1, J2 = "0123456789ab", "ba9876543210"   # real job ids: 12 hex digits


def test_discard_removes_only_that_jobs_files_and_clear_removes_all_staged(tmp_path):
    s = Staging(str(tmp_path))
    s.put(J1, [IMG, F0])
    s.put(J2, [IMG, END, F0, F1, VID])
    s.discard(J1)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f"{J2}-{f.name}" for f in (IMG, END, F0, F1, VID))
    s.clear()
    assert list(tmp_path.iterdir()) == []
    Staging(str(tmp_path / "missing")).clear()


def test_clear_leaves_files_it_did_not_stage(tmp_path):
    """A staging_dir pointed at the broker's state dir must not cost the database."""
    foreign = ["broker.db", "broker.db-wal", "notes-image.png", f"{J1}-image.gif", f"{J1}-other.png",
               f"{J1[:-1]}-image.png", f"x{J1}-image.png", f"{J1}-image.png.bak", f"{J1}-frames-x.png",
               f"{J1}-video.png.mp4x", "params.json"]
    for n in foreign:
        (tmp_path / n).write_bytes(b"keep")
    (tmp_path / f"{J1}-image.png").mkdir()
    s = Staging(str(tmp_path))
    s.put(J2, [IMG, F0, VID])
    s.clear()
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([*foreign, f"{J1}-image.png"])


def test_received_counts_staged_files_per_slot(tmp_path):
    s = Staging(str(tmp_path))
    s.put(J1, [IMG, F0, F1, VID])
    assert s.received(J1) == {"image": 1, "frames": 2, "video": 1}
    assert s.received(J2) == {}
