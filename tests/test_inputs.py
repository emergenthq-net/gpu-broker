"""Input slots: request shape, and whether a model's catalog `inputs` fit what a job sends."""
import pytest

from gpu_broker import inputs

I2V = {"inputs": {"image": "required", "end_image": "optional"}}
VIEWS = {"inputs": {"frames": "one_of", "video": "one_of"}}
f = frozenset


def test_slots_are_the_filled_fields():
    assert inputs.slots({"prompt": "x"}) == f()
    assert inputs.slots({"image": "aGk=", "end_image_url": "http://h/e.png"}) == {"image", "end_image"}
    assert inputs.slots({"frames": ["aGk=", "aGk="], "video_url": "http://h/v.mp4"}) == {"frames", "video"}


@pytest.mark.parametrize(("body", "msg"), [
    ({"image": 3}, "non-empty string"), ({"image_url": ""}, "non-empty string"), ({"video": ""}, "non-empty string"),
    ({"image": "aGk=", "image_url": "http://h/a.png"}, "not both"),
    ({"video": "AAAA", "video_url": "http://h/v.mp4"}, "not both"),
    ({"frames": []}, "non-empty list"), ({"frames": "aGk="}, "non-empty list"), ({"frames": ["aGk=", ""]}, "non-empty list")])
def test_slot_shape_errors(body, msg):
    with pytest.raises(ValueError, match=msg):
        inputs.slots(body)


def test_required_optional_and_unknown_slots():
    assert inputs.accepts(I2V, f({"image"})) and inputs.accepts(I2V, f({"image", "end_image"}))
    assert not inputs.accepts(I2V, f()) and not inputs.accepts({}, f({"image"})) and inputs.accepts({}, f())
    with pytest.raises(ValueError, match="'m' needs an input: send image"):
        inputs.check("m", I2V, f())
    with pytest.raises(ValueError, match="takes no image input"):
        inputs.check("t2v", {}, f({"image"}))
    with pytest.raises(ValueError, match=r"takes no end_image input \(it takes: image\)"):
        inputs.check("e", {"inputs": {"image": "required"}}, f({"image", "end_image"}))
    inputs.check("m", I2V, f({"image"}))


def test_one_of_needs_exactly_one_of_its_slots():
    assert inputs.accepts(VIEWS, f({"frames"})) and inputs.accepts(VIEWS, f({"video"}))
    assert not inputs.accepts(VIEWS, f()) and not inputs.accepts(VIEWS, f({"frames", "video"}))
    with pytest.raises(ValueError, match="needs exactly one of frames, video"):
        inputs.check("v", VIEWS, f({"frames", "video"}))


@pytest.mark.parametrize("spec", [{"img": "required"}, {"image": "maybe"}, {"end_image": True}])
def test_spec_rejects_unknown_slots_or_needs(spec):
    with pytest.raises(ValueError, match="inputs must map"):
        inputs.validate_spec("x", spec)


def test_spec_one_of_needs_two_slots():
    with pytest.raises(ValueError, match="one_of needs at least two"):
        inputs.validate_spec("x", {"video": "one_of"})
    inputs.validate_spec("x", {"video": "one_of", "frames": "one_of", "image": "optional"})
