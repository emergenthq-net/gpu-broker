"""The host script and the broker parse recipes by the same rules: for every variant below, the
broker's parser (drivers/recipes.py) and `gpu-broker-ctl exec-info` both accept it with the
same timeout_s, or both refuse it. CR is refused on both sides, never stripped. The program
(argv[0]) follows `env --` on the host, so neither side lets it look like a variable or option."""
import pytest

from gpu_broker.drivers import recipes
from tests import ctlfake
from tests.test_host_ctl_exec import RECIPE

VARIANTS = {
    "good": RECIPE,
    "crlf": RECIPE.replace("\n", "\r\n"),
    "cr in a comment": RECIPE.replace("# test recipe", "# test recipe\r"),
    "cr-only line": RECIPE + "\r\n",
    "trailing cr on timeout": RECIPE.replace("timeout_s=600", "timeout_s=600\r"),
    "decimal timeout": RECIPE.replace("timeout_s=600", "timeout_s=0.5"),
    "zero timeout": RECIPE.replace("timeout_s=600", "timeout_s=0"),
    "zero decimal timeout": RECIPE.replace("timeout_s=600", "timeout_s=00.000"),
    "exponent timeout": RECIPE.replace("timeout_s=600", "timeout_s=1e3"),
    "negative timeout": RECIPE.replace("timeout_s=600", "timeout_s=-5"),
    "leading dot timeout": RECIPE.replace("timeout_s=600", "timeout_s=.5"),
    "empty timeout": RECIPE.replace("timeout_s=600", "timeout_s="),
    "spaced timeout": RECIPE.replace("timeout_s=600", "timeout_s= 600"),
    "duplicate key, last wins": RECIPE + "timeout_s=900\n",
    "line without =": RECIPE + "argv\n",
    "unknown key": RECIPE + "shell=sh\n",
    "indented key": RECIPE + " outputs=*.ply\n",
    "blank and indented comment": RECIPE + "\n   \n  # note\n",
    "out_dir not ending in jid": RECIPE.replace("broker/{jid}", "{jid}/broker"),
    "out_dir with two jids": RECIPE.replace("output/broker/{jid}", "{jid}/broker/{jid}"),
    "bad argv word": RECIPE.replace("--job {jid}", "--job $(id)"),
    "parent dir in argv": RECIPE.replace("--job {jid}", "--job ../x"),
    "glob in argv": RECIPE.replace("--job {jid}", "--job /bin/bas?"),
    "tab-separated argv": RECIPE.replace("-i {in_dir}", "-i\t{in_dir}"),
    "no final newline": RECIPE.rstrip("\n"),
    "program is a variable": RECIPE.replace("argv=/opt", "argv=LD_PRELOAD=/x/y.so /opt"),
    "= inside the program": RECIPE.replace("argv=/opt/t/bin/predict", "argv=/opt/t/bin/a=b"),
    "program is an option": RECIPE.replace("argv=/opt", "argv=-i /opt"),
    "program after leading spaces": RECIPE.replace("argv=/opt", "argv=  /opt"),
    "option-looking later word": RECIPE.replace("--job {jid}", "--job=x {jid}"),
    "two output globs": RECIPE.replace("outputs=*.ply", "outputs=w.mp4  *.log "),
    "tab between output globs": RECIPE.replace("outputs=*.ply", "outputs=w.mp4\t*.log"),
    "bad second output glob": RECIPE.replace("outputs=*.ply", "outputs=w.mp4 /x/*.log"),
    "only spaces as outputs": RECIPE.replace("outputs=*.ply", "outputs=  "),
}


@pytest.mark.parametrize("name", VARIANTS)
def test_host_and_broker_agree(tmp_path, name):
    text = VARIANTS[name]
    try:
        broker = recipes.parse("t", text).timeout_s
    except ValueError:
        broker = None
    r, log, *_ = ctlfake.make(tmp_path, RECIPE)("exec-info t", recipe=text)
    info = dict(line.split("=", 1) for line in r.stdout.decode().split())
    host = float(info["timeout_s"]) if r.returncode == 0 else None
    assert host == broker, (name, r.returncode, r.stderr)
    assert log == []
