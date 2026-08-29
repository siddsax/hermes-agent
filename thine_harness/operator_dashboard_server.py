"""Loopback-only HTTP transport for the built-in Operator Dashboard."""

from __future__ import annotations

import ipaddress
import json
import secrets
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .operator_dashboard import OperatorDashboard


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def create_operator_dashboard_app(dashboard: OperatorDashboard) -> FastAPI:
    """Create a process-local app which refuses proxy-mediated access."""
    app = FastAPI(title="Local Thine Operator", docs_url=None, redoc_url=None)
    control_token = secrets.token_urlsafe(32)

    @app.middleware("http")
    async def loopback_guard(request: Request, call_next: Any) -> Any:
        forwarded = {
            "forwarded",
            "x-forwarded-for",
            "x-forwarded-host",
            "x-forwarded-proto",
            "x-real-ip",
        }
        if forwarded.intersection(request.headers):
            return JSONResponse(
                {"error": "proxied_operator_access_forbidden"}, status_code=403
            )
        if not _is_loopback(request.client.host if request.client else None):
            return JSONResponse(
                {"error": "operator_dashboard_loopback_only"}, status_code=403
            )
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _page(control_token)

    @app.get("/api/snapshot")
    async def snapshot(limit: int = 50) -> Any:
        try:
            return dashboard.snapshot(limit=limit)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async def _command(request: Request, *, execute: bool) -> JSONResponse:
        if request.headers.get("x-operator-token") != control_token:
            return JSONResponse({"error": "invalid_operator_token"}, status_code=403)
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("command must be a JSON object")
            result = (
                dashboard.execute_control(payload)
                if execute
                else dashboard.preview_control(payload)
            )
            return JSONResponse(result)
        except (ValueError, KeyError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/controls/preview")
    async def preview(request: Request) -> JSONResponse:
        return await _command(request, execute=False)

    @app.post("/api/controls/execute")
    async def execute(request: Request) -> JSONResponse:
        return await _command(request, execute=True)

    return app


def _page(control_token: str) -> str:
    token = json.dumps(control_token)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local Thine Operator</title><style>
:root{{--bg:#0b0d10;--card:#15191e;--line:#2b333d;--text:#edf2f7;--muted:#9aa7b4;--ok:#51d88a;--bad:#ff7b72;--accent:#5cb7e8}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}}header{{position:sticky;top:0;background:#0b0d10ed;border-bottom:1px solid var(--line);padding:16px 24px;z-index:2;display:flex;gap:16px;align-items:center}}h1{{font-size:18px;margin:0}}button{{background:var(--accent);color:#06131a;border:0;border-radius:7px;padding:8px 12px;font-weight:700;cursor:pointer}}main{{padding:20px;display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}}article{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;min-width:0}}h2{{margin:0 0 7px;font-size:15px}}.meta{{color:var(--muted);font-size:11px;margin-bottom:8px}}.ok{{color:var(--ok)}}.unavailable,.error{{color:var(--bad)}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-height:360px;overflow:auto;margin:0;font-size:11px}}#status{{margin-left:auto;color:var(--muted)}}dialog{{background:var(--card);color:var(--text);border:1px solid var(--line);border-radius:10px;width:min(680px,90vw)}}textarea{{width:100%;min-height:180px;background:#090b0e;color:var(--text);border:1px solid var(--line)}}
</style></head><body><header><h1>Local Thine Operator</h1><button id="refresh">Refresh</button><button id="command">Safe control</button><span id="status" role="status">Loading</span></header><main id="panels"></main>
<dialog id="modal"><form method="dialog"><h2>Preview then execute</h2><p>Enter a command JSON object. Destructive controls return an exact confirmation payload before they can execute.</p><label for="payload">Operator command JSON</label><textarea id="payload">{{"action":"reset","scope":"working_memory_topics"}}</textarea><p><button value="cancel">Close</button> <button type="button" id="preview">Preview</button> <button type="button" id="execute">Execute preview</button></p><pre id="result"></pre></form></dialog>
<script>const TOKEN={token};const panels=document.querySelector('#panels'),status=document.querySelector('#status'),modal=document.querySelector('#modal'),result=document.querySelector('#result');let execution=null;
function esc(v){{return JSON.stringify(v,null,2).replaceAll('&','&amp;').replaceAll('<','&lt;')}}
async function refresh(){{status.textContent='Refreshing…';try{{const r=await fetch('/api/snapshot');const s=await r.json();panels.innerHTML=Object.entries(s.panels).map(([name,p])=>`<article><h2>${{name.replaceAll('_',' ')}}</h2><div class="meta"><span class="${{p.status}}">${{p.status}}</span> · ${{p.source}} · ${{new Date(p.generated_at_ms).toLocaleTimeString()}}</div>${{p.error?`<p class="error">${{p.error}}</p>`:''}}<pre>${{esc(p.data)}}</pre></article>`).join('');status.textContent=`Updated ${{new Date(s.generated_at_ms).toLocaleTimeString()}} · last ${{s.limit}}`;}}catch(e){{status.textContent='Refresh failed: '+e}}}}
async function send(path,payload){{const r=await fetch(path,{{method:'POST',headers:{{'Content-Type':'application/json','X-Operator-Token':TOKEN}},body:JSON.stringify(payload)}});const body=await r.json();if(!r.ok)throw Error(body.error||r.status);return body}}
document.querySelector('#refresh').onclick=refresh;document.querySelector('#command').onclick=()=>modal.showModal();document.querySelector('#preview').onclick=async()=>{{try{{execution=await send('/api/controls/preview',JSON.parse(document.querySelector('#payload').value));result.textContent=JSON.stringify(execution,null,2)}}catch(e){{result.textContent=String(e)}}}};document.querySelector('#execute').onclick=async()=>{{try{{if(!execution?.execute_payload)throw Error('Preview first');const out=await send('/api/controls/execute',execution.execute_payload);result.textContent=JSON.stringify(out,null,2);await refresh()}}catch(e){{result.textContent=String(e)}}}};refresh();setInterval(refresh,3000);</script></body></html>"""


__all__ = ["create_operator_dashboard_app"]
