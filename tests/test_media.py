"""Input file data: format by magic bytes, size caps, base64 and data URLs (URL fetching: test_media_fetch.py)."""
import base64

import pytest

from gpu_broker import media
from gpu_broker.settings import Inputs

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 32
GIF = b"GIF89a" + b"\x00" * 32
CFG = Inputs(max_bytes=64, video_max_bytes=96, max_frames=3, allow_urls=True)
b64 = lambda data: base64.b64encode(data).decode()  # noqa: E731


@pytest.mark.parametrize(("data", "kind"), [(PNG, "png"), (JPEG, "jpeg"), (WEBP, "webp")])
def test_formats_are_recognised_by_magic_bytes(data, kind):
    assert media.sniff(data, CFG.types) == kind


MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 32
MOV = b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 32
WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 32


@pytest.mark.parametrize(("data", "kind"), [(MP4, "mp4"), (MOV, "mov"), (WEBM, "webm")])
def test_video_containers_are_recognised_by_magic_bytes(data, kind):
    assert media.sniff(data, CFG.video_types) == kind
    with pytest.raises(ValueError, match="not an accepted file"):
        media.sniff(data, CFG.types)            # a video is never accepted where an image is


def test_the_video_slot_has_its_own_cap_and_formats():
    assert media.decode("video", b64(MP4), CFG).kind == "mp4"
    with pytest.raises(ValueError, match="not an accepted file"):
        media.decode("video", b64(PNG), CFG)
    with pytest.raises(ValueError, match="larger than 96 bytes"):
        media.decode("video", b64(MP4 + b"\x00" * 96), CFG)
    assert media.limits("image", CFG) == (64, CFG.types) and media.limits("video", CFG) == (96, CFG.video_types)


def test_frames_are_decoded_in_order_with_indexed_names_and_capped_in_number():
    files = media.read({"frames": [b64(PNG), "data:image/jpeg;base64," + b64(JPEG)]}, CFG)
    assert [(f.slot, f.index, f.name) for f in files] == [("frames", 0, "frames-00.png"), ("frames", 1, "frames-01.jpg")]
    with pytest.raises(ValueError, match="at most 3"):
        media.read({"frames": [b64(PNG)] * 4}, CFG)
    with pytest.raises(ValueError, match=r"`frames\[1\]` is not valid base64"):
        media.read({"frames": [b64(PNG), "!!"]}, CFG)
    with pytest.raises(ValueError, match=r"`frames\[0\]`: not an accepted file"):
        media.read({"frames": [b64(MP4)]}, CFG)


def test_unknown_or_disallowed_formats_are_refused():
    with pytest.raises(ValueError, match="not an accepted file"):
        media.sniff(GIF, CFG.types)
    with pytest.raises(ValueError, match="not an accepted file"):
        media.sniff(PNG, ("jpeg",))
    with pytest.raises(ValueError, match="not an accepted file"):
        media.sniff(b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 8, CFG.types)   # RIFF but not WebP


def test_a_declared_type_must_match_the_data():
    assert media.sniff(PNG, CFG.types, "image/png; charset=binary") == "png"
    assert media.sniff(PNG, CFG.types, "application/octet-stream") == "png"
    with pytest.raises(ValueError, match="does not match"):
        media.sniff(PNG, CFG.types, "image/jpeg")
    with pytest.raises(ValueError, match="does not match"):
        media.sniff(PNG, CFG.types, "text/html")


def test_decode_takes_raw_base64_or_a_data_url():
    assert media.decode("image", b64(PNG), CFG).data == PNG
    im = media.decode("image", "data:image/jpeg;base64," + b64(JPEG), CFG)
    assert (im.kind, im.source, im.summary()) == ("jpeg", "inline", {"bytes": len(JPEG), "type": "jpeg", "source": "inline"})
    with pytest.raises(ValueError, match="does not match"):
        media.decode("image", "data:image/png;base64," + b64(JPEG), CFG)
    with pytest.raises(ValueError, match="not valid base64"):
        media.decode("image", "!!!!", CFG)


def test_decode_enforces_the_size_cap_before_and_after_decoding():
    exact = PNG + b"\x00" * (CFG.max_bytes - len(PNG))
    assert len(media.decode("image", b64(exact), CFG).data) == CFG.max_bytes
    with pytest.raises(ValueError, match="larger than 64 bytes"):
        media.decode("image", b64(exact + b"\x00"), CFG)
    with pytest.raises(ValueError, match="larger than 64 bytes"):
        media.decode("image", "!" * 4000, CFG)   # refused on length alone, before decoding
