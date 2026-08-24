"""Modal-only browser verification for the static evidence surfaces."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import modal

APP_NAME = "unseen-loop-ui-verification"
VOLUME_NAME = "unseen-loop-artifacts"
STUDY_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")

app = modal.App(APP_NAME)
artifacts = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("playwright==1.55.0")
    .run_commands("playwright install --with-deps chromium")
    .add_local_dir("site", "/site", copy=True)
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {"git_commit": commit, "git_dirty": dirty, "modal_sdk_version": modal.__version__}


@app.function(
    image=image,
    cpu=(4.0, 4.0),
    memory=(8_192, 8_192),
    volumes={"/artifacts": artifacts},
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    scaledown_window=60,
    timeout=1_800,
    retries=0,
)
def verify_ui_remote(study_id: str, source_json: str) -> str:
    """Drive Chromium against the real static site and persist screenshots plus assertions."""
    import socket
    import sys
    import time

    from playwright.sync_api import sync_playwright

    if STUDY_ID_PATTERN.fullmatch(study_id) is None:
        raise ValueError("invalid study_id")
    source = json.loads(source_json)
    if not isinstance(source, dict):
        raise TypeError("source_json must encode an object")
    destination = Path("/artifacts/studies") / study_id
    artifacts.reload()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise RuntimeError("UI verification destination is not empty")
    destination.mkdir(parents=True, exist_ok=True)

    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8000", "--directory", "/site"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", 8000), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError("static site server did not become ready")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto("http://127.0.0.1:8000/index.html", wait_until="networkidle")
            page.wait_for_function("document.body.dataset.evidenceState === 'loaded'")
            index_state = page.evaluate(
                """() => ({
                    state: document.body.dataset.evidenceState,
                    notMeasured: document.body.innerText.includes('NOT MEASURED'),
                    metricValues: [...document.querySelectorAll('.metric-value')].map(
                        n => n.textContent.trim()
                    ),
                    scrollWidth: document.documentElement.scrollWidth,
                    innerWidth,
                })"""
            )
            if (
                index_state["notMeasured"]
                or index_state["scrollWidth"] != index_state["innerWidth"]
            ):
                raise RuntimeError("index surface failed evidence or layout gate")

            page.goto("http://127.0.0.1:8000/studies.html", wait_until="networkidle")
            page.wait_for_function("document.body.dataset.publicationState === 'loaded'")
            studies_state = page.evaluate(
                """() => ({
                    state: document.body.dataset.publicationState,
                    notVerified: document.body.innerText.includes('NOT VERIFIED'),
                    expandedRows: document.querySelectorAll('#expanded-body tr').length,
                    ablationRows: document.querySelectorAll('#ablation-body tr').length,
                    effectRows: document.querySelectorAll('#effect-ledger > *').length,
                    sourceRows: document.querySelectorAll('#sources-body tr').length,
                    h1: document.querySelectorAll('h1').length,
                    main: document.querySelectorAll('main').length,
                    scrollWidth: document.documentElement.scrollWidth,
                    innerWidth,
                })"""
            )
            expected = {
                "expandedRows": 3,
                "ablationRows": 4,
                "effectRows": 3,
                "sourceRows": 7,
                "h1": 1,
                "main": 1,
            }
            if studies_state["notVerified"]:
                raise RuntimeError("Studies surface rendered an unverified state")
            for key, value in expected.items():
                if studies_state[key] != value:
                    raise RuntimeError(
                        f"Studies surface {key}={studies_state[key]}, expected {value}"
                    )
            if studies_state["scrollWidth"] != studies_state["innerWidth"]:
                raise RuntimeError("desktop Studies surface has page-level overflow")
            desktop_path = destination / "studies-desktop.png"
            page.screenshot(path=str(desktop_path), full_page=True)

            page.set_viewport_size({"width": 390, "height": 844})
            page.reload(wait_until="networkidle")
            page.wait_for_function("document.body.dataset.publicationState === 'loaded'")
            mobile_state = page.evaluate(
                """() => ({
                    state: document.body.dataset.publicationState,
                    scrollWidth: document.documentElement.scrollWidth,
                    innerWidth,
                    tables: document.querySelectorAll('table').length,
                    focusableRegions: document.querySelectorAll(
                        '.table-scroll[tabindex=\"0\"]'
                    ).length,
                })"""
            )
            if mobile_state["scrollWidth"] != mobile_state["innerWidth"]:
                raise RuntimeError("mobile Studies surface has page-level overflow")
            if mobile_state["focusableRegions"] < mobile_state["tables"]:
                raise RuntimeError("not every Studies table is a keyboard-focusable region")
            mobile_path = destination / "studies-mobile.png"
            page.screenshot(path=str(mobile_path), full_page=True)
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=5)

    summary = {
        "schema_version": "unseen-loop/ui-verification-v1",
        "study_id": study_id,
        "execution": {"location": "Modal", "browser": "Playwright Chromium", "headless": True},
        "source_provenance": source,
        "index": index_state,
        "studies_desktop": studies_state,
        "studies_mobile": mobile_state,
        "screenshots": {
            "desktop": {"path": str(desktop_path), "sha256": _sha256(desktop_path)},
            "mobile": {"path": str(mobile_path), "sha256": _sha256(mobile_path)},
        },
        "all_gates_passed": True,
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    files = (desktop_path, mobile_path, summary_path)
    checksums_path = destination / "checksums.sha256"
    checksums_path.write_text("\n".join(f"{_sha256(path)}  {path.name}" for path in files) + "\n")
    artifacts.commit()
    return _canonical(summary)


@app.local_entrypoint()
def main(study_id: str = "modal-ui-verification-001") -> str:
    """Dispatch Chromium verification; no browser or server runs on the local machine."""
    return verify_ui_remote.remote(study_id, _canonical(_source_provenance()))
