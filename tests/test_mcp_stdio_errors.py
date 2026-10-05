"""`gpu-broker mcp` (stdio) answers every request even when the broker fails: a refused key is
a JSON-RPC error saying so, then exit 2; an unreachable broker is an error for the request in
flight, and the relay keeps serving. Run as a subprocess against the broker's real app."""
import json
import os
import subprocess
import sys
import threading

from tests.mcp_fakes import mcp_broker
from tests.test_mcp_e2e import live

__all__ = ["live", "mcp_broker"]
INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}
TIMEOUT_S = 20


def relay(url, env, tmp_path):
    return subprocess.Popen([sys.executable, "-m", "gpu_broker", "mcp", "--url", url], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env={"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), **env})


def first_reply(p):
    p.stdin.write(json.dumps(INIT) + "\n")
    p.stdin.flush()
    line: list[str] = []
    reader = threading.Thread(target=lambda: line.append(p.stdout.readline()), daemon=True)
    reader.start()
    reader.join(TIMEOUT_S)
    if not line:
        p.kill()
        raise AssertionError("the request was never answered")
    return json.loads(line[0])


def test_a_revoked_key_is_answered_with_why_then_exit_2(live, tmp_path):
    url, _ = live
    p = relay(url, {"GPU_BROKER_API_KEY": "gbk_" + "x" * 43}, tmp_path)
    reply = first_reply(p)
    assert reply["id"] == 1 and "key revoked or invalid" in reply["error"]["message"] and "HTTP 401" in reply["error"]["message"]
    assert p.wait(TIMEOUT_S) == 2 and "key revoked or invalid" in p.stderr.read()


def test_an_unreachable_broker_answers_the_request_in_flight(tmp_path):
    p = relay("http://127.0.0.1:9", {"GPU_BROKER_API_KEY": "gbk_" + "x" * 43}, tmp_path)
    reply = first_reply(p)
    assert reply["id"] == 1 and "error" in reply
    p.stdin.close()
    assert p.wait(TIMEOUT_S) == 0
