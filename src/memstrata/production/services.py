"""Declarative service layer for production runs.

Different pipelines need different long-lived services with different launch commands, so the
*decision of which services to bring up and how* lives here in source (next to the runner),
not in ad-hoc shell/tmux. A run declares the services it needs (see ``required_services``) and
the manager brings each one up **reuse-first**:

  * health-check the endpoint; if already serving -> reuse it (shared-node safe: we NEVER kill
    a process we did not start), just point the pipeline at it;
  * otherwise launch it detached (``start_new_session``) with logs under ``<run_dir>/services/``
    and poll until healthy or timeout.

Scope note: the FLUX image server, Helios/Wan video servers and the S5 crop-acq server already
self-start from inside their backends (file-queue servers that own their own interpreter/GPU), so
they are intentionally NOT re-managed here. The one service that was previously manual — the
Qwen OpenAI-compatible MLLM endpoint shared by the keyframe composer (R3/R4) and the generation
router — is what this layer provisions.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ServiceSpec:
    """A long-lived service the pipeline depends on."""

    name: str
    start_cmd: list[str]
    health_url: str
    ready_timeout: float = 900.0
    poll_interval: float = 5.0
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path | None = None
    # substring that must appear in the health-endpoint body (e.g. served model name); "" = any 200
    ready_contains: str = ""


@dataclass
class ServiceHandle:
    name: str
    health_url: str
    reused: bool
    pid: int | None = None
    log_path: Path | None = None


def _healthy(url: str, contains: str = "", timeout: float = 4.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (localhost)
            if resp.status != 200:
                return False
            if not contains:
                return True
            return contains in resp.read().decode("utf-8", "ignore")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


class ServiceManager:
    """Reuse-or-launch services; tracks only what *we* started so shutdown never kills others."""

    def __init__(self, run_dir: Path) -> None:
        self.log_dir = Path(run_dir) / "services"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._launched: list[ServiceHandle] = []

    def ensure(self, spec: ServiceSpec) -> ServiceHandle:
        if _healthy(spec.health_url, spec.ready_contains):
            print(f"[svc] {spec.name}: reusing existing @ {spec.health_url}", flush=True)
            return ServiceHandle(spec.name, spec.health_url, reused=True)

        log_path = self.log_dir / f"{spec.name}.log"
        env = {**os.environ, **spec.env}
        print(f"[svc] {spec.name}: launching -> {log_path}", flush=True)
        with open(log_path, "ab") as log:
            proc = subprocess.Popen(
                spec.start_cmd, cwd=str(spec.cwd or Path.cwd()), env=env,
                stdout=log, stderr=log, start_new_session=True)
        handle = ServiceHandle(spec.name, spec.health_url, reused=False,
                               pid=proc.pid, log_path=log_path)

        deadline = time.time() + spec.ready_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"service {spec.name!r} exited early (code {proc.returncode}); see {log_path}")
            if _healthy(spec.health_url, spec.ready_contains):
                print(f"[svc] {spec.name}: ready (pid={proc.pid})", flush=True)
                self._launched.append(handle)
                return handle
            time.sleep(spec.poll_interval)
        raise TimeoutError(
            f"service {spec.name!r} not healthy after {spec.ready_timeout:.0f}s; see {log_path}")

    def ensure_all(self, specs: list[ServiceSpec]) -> list[ServiceHandle]:
        return [self.ensure(s) for s in specs]

    def shutdown_launched(self) -> None:
        """Stop only services this manager launched (leave reused ones alone)."""
        import signal
        for h in self._launched:
            if h.pid is None:
                continue
            try:
                os.killpg(os.getpgid(h.pid), signal.SIGTERM)
                print(f"[svc] {h.name}: stopped (pid={h.pid})", flush=True)
            except (ProcessLookupError, PermissionError):
                pass
        self._launched.clear()


def _memstrata_root() -> Path:
    return Path(__file__).resolve().parents[3]


def qwen_mllm_spec(
    *,
    gpu: str = "0",
    port: int = 8000,
    kind: str = "text",
    served_model_name: str = "Qwen3.5-9B-Instruct",
    ready_timeout: float = 900.0,
    extra_env: dict[str, str] | None = None,
) -> ServiceSpec:
    """Qwen OpenAI-compatible endpoint for the R3/R4/router MLLM roles.

    Delegates the actual vLLM invocation to ``scripts/memstrata/servers/serve_qwen.sh`` (single
    source of truth for the vLLM flags; each pipeline keeps its service launchers under its own
    ``servers/`` dir); this factory only fixes the per-run choices (gpu/port/served name) and the
    health check. Qwen3.5-9B is multimodal, so the single ``text`` server plays all roles.
    """
    root = _memstrata_root()
    script = root / "scripts/memstrata/servers/serve_qwen.sh"
    env = {"SERVED_MODEL_NAME": served_model_name}
    if extra_env:
        env.update(extra_env)
    return ServiceSpec(
        name=f"qwen_{kind}",
        start_cmd=["bash", str(script), kind, str(gpu), str(port)],
        health_url=f"http://127.0.0.1:{port}/v1/models",
        ready_contains=served_model_name,
        ready_timeout=ready_timeout,
        env=env,
        cwd=root,
    )


def required_services(
    *,
    flux: bool,
    decompose: str,
    use_router_mllm: bool,
    mllm_gpu: str = "0",
    mllm_port: int = 8000,
) -> list[ServiceSpec]:
    """Which externally-managed services this run configuration needs.

    Only the Qwen MLLM endpoint qualifies today (video/image/crop servers self-start inside
    their backends). It is needed when the ``crop_server`` decompose path names/decomposes via
    the MLLM, when the router uses the MLLM, or when the keyframe composer runs in the legacy
    ``collage`` mode (R3/R4). The default native FLUX keyframe path composes multi-image with no
    Qwen, so ``flux`` alone no longer pulls in the MLLM endpoint.
    """
    legacy_keyframe = flux and (
        os.environ.get("MEMSTRATA_KEYFRAME_MODE", "native").strip().lower() == "collage"
    )
    needs_mllm = legacy_keyframe or decompose == "crop_server" or use_router_mllm
    specs: list[ServiceSpec] = []
    if needs_mllm:
        specs.append(qwen_mllm_spec(gpu=mllm_gpu, port=mllm_port))
    return specs
