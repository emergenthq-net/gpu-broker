"""Turn a model request into what will actually run: the model itself, a substitute, a
download, or a reasoned rejection. Pure functions over the catalog — no IO.

A substitute must be the same kind and cover every requested capability; among those the
highest `quality` wins. The reason is always reported, so a caller never silently gets a
different model than it asked for.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .catalog import CatalogData, Model, Source
from .constants import DownloadKind, Kind, ModelStatus, Runner

HF_URL = re.compile(r"^https?://huggingface\.co/([\w.-]+/[\w.-]+)")
GH_URL = re.compile(r"^https?://github\.com/[\w.-]+/[\w.-]+/?$")
REPO_ID = re.compile(r"^[\w.-]+/[\w.-]+$")
SLUG_MAX = 60
GH_HOST = "github.com/"


@dataclass
class Download:
    kind: DownloadKind
    ref: str
    slug: str
    include: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "ref": self.ref, "slug": self.slug, "include": self.include}


@dataclass
class Resolution:
    requested: str
    resolved: str | None              # catalog key that will run
    substitution: str | None = None   # why resolved != requested
    download: Download | None = None
    register: Model | None = None     # catalog entry to add for an unknown repository
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def lookup(catalog: CatalogData, name: str) -> str | None:
    """Catalog key for a name: the key itself, an alias, the served name or the HF repo id."""
    models = catalog["models"]
    if name in models:
        return name
    low = name.lower()
    for key, m in models.items():
        names = {key, m.get("served_name", ""), m.get("source", {}).get("hf", ""), *m.get("aliases", [])}
        if low in {n.lower() for n in names if n}:
            return key
    return None


def budget(catalog: CatalogData) -> int:
    d = catalog["defaults"]
    return d["vram_total_mib"] - d["vram_reserve_mib"]


def runnable(catalog: CatalogData, key: str, session: bool = False) -> bool:
    """Ready and fits. A session-only entry (used interactively through its own front end)
    can be borrowed in a session but never queued as a job."""
    m = catalog["models"][key]
    return (m.get("status") == ModelStatus.READY and (session or not m.get("session_only"))
            and m.get("vram_mib", 0) <= budget(catalog))


def best_substitute(catalog: CatalogData, kind: str, caps: set[str], exclude: str | None = None) -> str | None:
    cands = [(m.get("quality", 0), k) for k, m in catalog["models"].items()
             if k != exclude and m.get("kind") == kind and caps <= set(m.get("caps", []))
             and runnable(catalog, k)]
    return max(cands)[1] if cands else None


def slug(ref: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", ref.lower().split(GH_HOST)[-1]).strip("-")[:SLUG_MAX]


def _download_for(src: Source, default_slug: str) -> Download | None:
    if "hf" in src:
        return Download(DownloadKind.HF, src["hf"], src.get("slug", default_slug), list(src.get("include", [])))
    if "gh" in src:
        return Download(DownloadKind.GH, src["gh"], src.get("slug", default_slug))
    return None


def _unknown(catalog: CatalogData, name: str, kind: str | None, caps: set[str]) -> Resolution:
    res = Resolution(requested=name, resolved=None)
    ref, dl_kind = None, None
    if (m := HF_URL.match(name)):
        ref, dl_kind = m.group(1), DownloadKind.HF
    elif GH_URL.match(name):
        ref, dl_kind = name.rstrip("/"), DownloadKind.GH
    elif REPO_ID.match(name):
        ref, dl_kind = name, DownloadKind.HF
    if ref and dl_kind:
        s = slug(ref)
        res.download = Download(dl_kind, ref, s)
        res.register = Model(kind=kind or Kind.UNKNOWN.value, runner=Runner.EXTERNAL.value, caps=sorted(caps),
                             quality=0, status=ModelStatus.NEEDS_INTEGRATION.value,
                             source=Source(slug=s, hf=ref) if dl_kind is DownloadKind.HF else Source(slug=s, gh=ref))
        res.notes.append("unknown model: queued for download; needs a runner before it can serve")
    if kind and (sub := best_substitute(catalog, kind, caps)):
        res.resolved = sub
        res.substitution = f"'{name}' is not in the catalog; using closest installed {kind} model '{sub}'"
    if res.resolved is None:
        res.error = (f"no installed {kind} model can stand in for '{name}' yet" if kind and res.download else
                     f"no installed model can stand in for '{name}' yet" if res.download else
                     f"unknown model '{name}' and no kind given to pick a substitute")
    return res


def _why_not(catalog: CatalogData, key: str, m: Model, res: Resolution) -> str:
    status = m.get("status")
    if m.get("vram_mib", 0) > budget(catalog):
        return f"needs {m.get('vram_mib')} MiB, more than the {budget(catalog)} MiB available"
    if status == ModelStatus.DOWNLOADABLE:
        res.download = _download_for(m.get("source", {}), key)
        return "not downloaded yet"
    if status == ModelStatus.NEEDS_INTEGRATION:
        if not m.get("downloaded"):
            res.download = _download_for(m.get("source", {}), key)
        return "downloaded or downloadable, but no runner is wired for it yet"
    if m.get("session_only"):
        return "is session-only (open it from the dashboard's model index; POST /v1/sessions)"
    return f"status '{status}'"


def resolve(catalog: CatalogData, name: str, kind: str | None = None, caps: list[str] | None = None,
            session: bool = False) -> Resolution:
    capset = set(caps or [])
    key = lookup(catalog, name)
    if key is None:
        return _unknown(catalog, name, kind, capset)
    m = catalog["models"][key]
    requested_kind = kind
    kind = kind or m.get("kind", "")
    capset = set(m.get("caps", [])) if caps is None and requested_kind is None else set(caps or [])
    missing = capset - set(m.get("caps", []))
    wrong_kind = bool(kind and m.get("kind") != kind)
    if runnable(catalog, key, session) and not wrong_kind and not missing:
        return Resolution(requested=name, resolved=key)
    res = Resolution(requested=name, resolved=None)
    if wrong_kind:
        why = f"is kind {m.get('kind')!r}, not requested kind {kind!r}"
    elif missing:
        why = f"does not provide capabilities {sorted(missing)}"
    else:
        why = _why_not(catalog, key, m, res)
    sub = best_substitute(catalog, kind, capset, exclude=key)
    if sub:
        res.resolved, res.substitution = sub, f"'{key}' {why}; using '{sub}' instead"
    else:
        res.error = f"'{key}' {why}, and no installed {kind} model covers {sorted(capset)}"
    return res
