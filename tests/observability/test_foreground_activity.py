import json
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from openviking.server.routers.system import health_check
from openviking.storage.observers.queue_observer import QueueObserver
from openviking.service.debug_service import ObserverService
from openviking.utils.foreground_activity import activity_snapshot, foreground_is_active


def _write_lease(root, name: str, payload: dict) -> None:
    (root / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_only_unexpired_leases_block_background_work(tmp_path, monkeypatch):
    monkeypatch.setenv("KG_USER_ACTIVITY_DIR", str(tmp_path))
    _write_lease(tmp_path, "expired", {"attempt": "expired", "expires_at": 99})
    _write_lease(
        tmp_path,
        "active",
        {"attempt": "active", "started_at": 90, "expires_at": 101},
    )

    snapshot = activity_snapshot(now=100)

    assert snapshot["active"] is True
    assert snapshot["active_leases"] == 1
    assert snapshot["background_eligible"] is False


def test_malformed_lease_is_visible_but_does_not_deadlock_workers(tmp_path, monkeypatch):
    monkeypatch.setenv("KG_USER_ACTIVITY_DIR", str(tmp_path))
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")

    snapshot = activity_snapshot(now=100)

    assert snapshot["active"] is False
    assert snapshot["invalid_leases"] == 1
    assert snapshot["background_eligible"] is True
    assert foreground_is_active() is False


@pytest.mark.asyncio
async def test_queue_snapshot_names_current_and_pending_stages(tmp_path, monkeypatch):
    monkeypatch.setenv("KG_USER_ACTIVITY_DIR", str(tmp_path))

    class QueueManager:
        SEMANTIC = "Semantic"
        _queues = {}

        async def check_status(self):
            return {
                "Embedding": SimpleNamespace(
                    pending=3,
                    in_progress=0,
                    processed=4,
                    requeue_count=1,
                    error_count=0,
                ),
                "AddResource": SimpleNamespace(
                    pending=0,
                    in_progress=2,
                    processed=5,
                    requeue_count=0,
                    error_count=1,
                ),
            }

    snapshot = await QueueObserver(QueueManager()).snapshot_async()

    assert snapshot["pending_stages"] == ["Embedding"]
    assert snapshot["active_stages"] == ["AddResource"]
    assert snapshot["totals"]["pending"] == 3
    assert snapshot["totals"]["in_progress"] == 2
    assert snapshot["foreground_activity"]["background_eligible"] is True


@pytest.mark.asyncio
async def test_observer_service_workload_snapshot_keeps_structured_observers(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KG_USER_ACTIVITY_DIR", str(tmp_path))

    class QueueManager:
        SEMANTIC = "Semantic"
        _queues = {}

        async def check_status(self):
            return {}

    monkeypatch.setattr(
        "openviking.service.debug_service.get_queue_manager",
        lambda: QueueManager(),
    )

    snapshot = await ObserverService().workload_snapshot()

    assert snapshot["queue"]["totals"]["pending"] == 0
    assert snapshot["queue"]["foreground_activity"]["active"] is False
    assert snapshot["models"] == {"vlm": [], "embedding": [], "rerank": []}
    assert "total_queries" in snapshot["retrieval"]


@pytest.mark.asyncio
async def test_health_endpoint_reports_structured_workload(monkeypatch):
    workload = {
        "queue": {
            "totals": {"pending": 0, "in_progress": 0},
            "foreground_activity": {"active": False},
        },
        "models": {},
        "retrieval": {},
    }

    class Observer:
        async def workload_snapshot(self):
            return workload

    service = SimpleNamespace(
        _initialized=True,
        debug=SimpleNamespace(observer=Observer()),
    )
    monkeypatch.setattr(
        "openviking.server.routers.system.get_service", lambda: service
    )
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/health",
            "headers": [],
            "app": SimpleNamespace(state=SimpleNamespace(config=None)),
        }
    )

    body = await health_check(request)

    assert body["workload"] == workload
    assert body["idle"] is True


@pytest.mark.asyncio
async def test_health_endpoint_exposes_workload_observer_failure(monkeypatch):
    class Observer:
        async def workload_snapshot(self):
            raise RuntimeError("queue telemetry unavailable")

    service = SimpleNamespace(
        _initialized=True,
        debug=SimpleNamespace(observer=Observer()),
    )
    monkeypatch.setattr(
        "openviking.server.routers.system.get_service", lambda: service
    )
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/health",
            "headers": [],
            "app": SimpleNamespace(state=SimpleNamespace(config=None)),
        }
    )

    body = await health_check(request)

    assert body["workload"] == {
        "status": "error",
        "error_type": "RuntimeError",
        "error": "queue telemetry unavailable",
    }
    assert "idle" not in body
