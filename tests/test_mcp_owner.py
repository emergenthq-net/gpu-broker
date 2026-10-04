"""Review of the MCP PR: who owns a job (the client key's id, never its name), what a client key
may send (no `<slot>_url` unless the operator allows it), only runnable models, and gpu_status."""
import dataclasses
from types import SimpleNamespace

import pytest

from gpu_broker import settings
from gpu_broker.keys import KeyStore
from gpu_broker.mcp_server import core
from gpu_broker.store import Store
from gpu_broker.web.jobs import CLIENT_ID_STATE, CLIENT_STATE, identity
from tests.mcp_fakes import PNG_B64, mcp_broker

__all__ = ["mcp_broker"]
FOX = {"model": "sdxl-base", "kind": "image", "caps": ["t2i"], "prompt": "a fox"}


def test_two_keys_with_the_same_name_do_not_see_each_others_jobs(mcp_broker):
    a, b = core.Caller("remote machine", "k1"), core.Caller("remote machine", "k2")   # every unnamed invite
    jid = core.submit(mcp_broker, a, dict(FOX))
    assert mcp_broker.store.job(jid)["owner"] == "k1" and mcp_broker.store.job(jid)["requester"] == "remote machine"
    assert core.report(mcp_broker, a, jid, 5)["state"] == "done"
    with pytest.raises(ValueError, match=core.HIDDEN):
        core.report(mcp_broker, b, jid, 0)


def test_a_key_named_mcp_does_not_see_the_main_tokens_jobs(mcp_broker):
    jid = core.submit(mcp_broker, core.Caller(core.MAIN_REQUESTER), dict(FOX))
    assert mcp_broker.store.job(jid)["owner"] is None
    with pytest.raises(ValueError, match=core.HIDDEN):
        core.report(mcp_broker, core.Caller("mcp", "k3"), jid, 0)
    assert core.report(mcp_broker, core.Caller(core.MAIN_REQUESTER), jid, 5)["state"] == "done"


def test_keys_identify_by_id_and_stored_responses_are_owned_by_it(tmp_path):
    keys = KeyStore(str(tmp_path / "b.db"))
    one, two = keys.issue("remote machine"), keys.issue("remote machine")
    assert keys.identify(one["key"]) == (one["id"], "remote machine") != keys.identify(two["key"])
    assert keys.check(one["key"]) == "remote machine" and keys.identify("gbk_nope") is None

    def req(kid, name):
        return SimpleNamespace(state=SimpleNamespace(**{CLIENT_ID_STATE: kid, CLIENT_STATE: name}), headers={})
    assert identity(req(one["id"], "remote machine")) != identity(req(two["id"], "remote machine"))


def test_an_older_database_gains_the_owner_column(tmp_path):
    import sqlite3
    db = sqlite3.connect(tmp_path / "b.db")
    db.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, created REAL, updated REAL, requester TEXT, requested TEXT, resolved TEXT, "
               "substitution TEXT, state TEXT, payload TEXT, result TEXT, error TEXT, download TEXT, exec_recipe TEXT)")
    db.commit()
    db.close()
    s = Store(str(tmp_path / "b.db"))
    assert s.job(s.create_job("x", "m", {}, owner="k1"))["owner"] == "k1"


def test_a_client_key_sends_files_as_base64_unless_urls_are_allowed(mcp_broker):
    key, main = core.Caller("laptop", "k1"), core.Caller(core.MAIN_REQUESTER)
    body = {"model": "qwen-image-edit", "kind": "image", "caps": ["edit"], "prompt": "x", "image_url": "https://example.com/a.png"}
    with pytest.raises(ValueError, match="may not send image_url"):
        core.submit(mcp_broker, key, dict(body))
    assert mcp_broker.store.jobs(10) == []                      # refused before anything is recorded or fetched
    with pytest.raises(ValueError, match="image_url") as main_err:   # the main token reaches the inputs layer
        core.submit(mcp_broker, main, dict(body))
    assert "may not send" not in str(main_err.value)
    mcp_broker.settings = dataclasses.replace(mcp_broker.settings, mcp=settings.Mcp(client_url_inputs=True))
    with pytest.raises(ValueError, match="image_url") as allowed:
        core.submit(mcp_broker, key, dict(body))
    assert "may not send" not in str(allowed.value)
    jid = core.submit(mcp_broker, key, {**body, "image_url": None} | {"image": PNG_B64})   # base64 is always fine
    assert core.report(mcp_broker, key, jid, 5)["state"] == "done"


def test_only_runnable_models_are_taken_by_name_and_chat_too(mcp_broker):
    for name in ("flux-schnell", "someone/new-model", "swarmui"):
        with pytest.raises(ValueError, match="can run now"):
            core.named(mcp_broker, name)
        with pytest.raises(ValueError, match="can run now"):
            core.chat(mcp_broker, core.Caller("mcp"), {"model": name, "messages": [{"role": "user", "content": "hi"}]})
    assert mcp_broker.store.jobs(10) == [] and mcp_broker.store.downloads(10) == []   # nothing queued, nothing downloading
    assert core.chat(mcp_broker, core.Caller("mcp"), {"model": "llama-8b", "messages": [{"role": "user", "content": "hi"}]})["text"]
