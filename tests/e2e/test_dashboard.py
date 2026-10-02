"""Browser end-to-end test of the SOC dashboard against the real API.

Skips unless the frontend is built (frontend/dist) and Playwright + Chromium are available.
Run:  cd frontend && npm run build && cd .. && pytest tests/e2e
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e
DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist" / "index.html"
sync_api = pytest.importorskip("playwright.sync_api")
if not DIST.exists():
    pytest.skip("frontend not built (cd frontend && npm run build)", allow_module_level=True)


@pytest.fixture(scope="module")
def server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "qadapt.api.app:app", "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import urllib.request
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1)
            break
        except OSError:
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("API did not start")
    yield f"http://127.0.0.1:{port}"
    proc.terminate()
    proc.wait(10)


@pytest.fixture(scope="module")
def page(server):
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:  # pragma: no cover - no browser installed
            pytest.skip(f"Chromium unavailable: {e}")
        pg = browser.new_page(viewport={"width": 1400, "height": 950})
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.goto(server)
        pg.wait_for_selector("text=Q-ADAPT Security Center")
        pg.errors = errors
        yield pg
        browser.close()


def nav(page, label):
    page.click(f"nav >> text={label}")


def test_every_screen_renders(page):
    for label, marker in [("Threat Detection", "Active threats"), ("Attack Graph", "Details"),
                          ("Risk Analysis", "Asset risk register"), ("Quantum Optimizer", "Problem & solver"),
                          ("Defense Plan", "No recommendation yet"), ("Solver Benchmark", "Run all solvers"),
                          ("SOC Overview", "Risk timeline")]:
        nav(page, label)
        page.wait_for_selector(f"text={marker}", timeout=15000)
    assert page.errors == []


def test_attack_graph_canvas_draws(page):
    nav(page, "Attack Graph")
    page.wait_for_selector("canvas", timeout=15000)
    assert page.locator("canvas").count() >= 1


def test_ml_detection_flags_host(page):
    nav(page, "Threat Detection")
    page.select_option("select >> nth=0", "FS-01")
    page.click("text=Run detection")
    page.wait_for_selector("text=host(s) flagged", timeout=60000)
    page.wait_for_selector("td >> text=FS-01", timeout=15000)


def test_optimize_approve_and_adapt(page):
    nav(page, "SOC Overview")
    page.click("text=Reset scenario")
    page.wait_for_timeout(500)
    risk_before = page.locator("text=Current risk >> xpath=..").inner_text()
    nav(page, "Quantum Optimizer")
    page.click("text=Run optimization")
    page.wait_for_selector("text=Circuit depth", timeout=60000)
    page.click("text=Review defense plan")
    page.wait_for_selector("text=why selected?")
    page.click("text=Approve all")
    page.wait_for_selector("text=no longer pending", timeout=15000)
    nav(page, "SOC Overview")
    page.wait_for_selector("text=Approved", timeout=15000)
    assert page.locator("text=Current risk >> xpath=..").inner_text() != risk_before
    page.click("text=Advance →")
    page.wait_for_selector("text=T2: Attacker moved", timeout=15000)
    assert page.errors == []


def test_benchmark_table(page):
    nav(page, "Solver Benchmark")
    page.fill("input >> nth=0", "8")
    page.click("text=Run all solvers")
    page.wait_for_selector("text=Objective (lower is better)", timeout=120000)
    assert page.locator("td >> text=exhaustive").count() == 1
    assert page.errors == []
