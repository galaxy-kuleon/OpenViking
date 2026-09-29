import json
import re
from uuid import uuid4

from openviking.service.task_tracker import get_task_tracker
from openviking.storage.queuefs import QueueManager, get_queue_manager
from openviking.storage.queuefs.session_commit_msg import SessionCommitMsg
from openviking.utils.time_utils import get_current_timestamp
from openviking_cli.exceptions import FailedPreconditionError, NotFoundError


async def retry_commit(session, expected_task_id, archive_uri=None):
    tracker = get_task_tracker()
    owner = {"account_id": session.ctx.account_id, "user_id": session.ctx.user.user_id}
    archives = await session._list_archive_refs()
    if archive_uri:
        archives = [a for a in archives if a["archive_uri"] == archive_uri]
    for archive in archives:
        uri = archive["archive_uri"]
        if not re.fullmatch(r"archive_[0-9]{3,}", archive["archive_id"]):
            continue
        meta = await session._read_archive_meta(uri)
        original = (meta.get("phase1") or {}).get("queue_message") or {}
        attempts = meta.get("retry_attempts") or {}
        if expected_task_id != original.get("task_id") and expected_task_id not in attempts:
            continue
        if (original.get("user") != owner or original.get("session_id") != session.session_id
                or original.get("session_uri") != session.uri or original.get("archive_uri") != uri):
            raise FailedPreconditionError("Archived commit ownership does not match this session")
        replay = next((a for a in attempts.values() if a.get("retry_of_task_id") == expected_task_id), None)
        if replay and replay.get("state") == "queued":
            msg = SessionCommitMsg.from_dict(replay["queue_message"])
            task = await tracker.get(msg.task_id, **owner)
            if not task:
                raise FailedPreconditionError("Retry task is unavailable; retained archive requires reconciliation")
            return {"session_id": session.session_id, "status": "accepted", "task_id": msg.task_id,
                    "archive_uri": uri, "archived": False, "retry_of_task_id": expected_task_id}
        path = session._viking_fs._uri_to_path(uri, ctx=session.ctx)
        lease = await session._viking_fs._async_agfs.pathlock_acquire_exact(path, timeout_secs=10)
        try:
            meta = await session._read_archive_meta(uri)
            phase1 = meta.get("phase1") or {}
            original = phase1.get("queue_message") or {}
            attempts = dict(meta.get("retry_attempts") or {})
            if expected_task_id != original.get("task_id") and expected_task_id not in attempts:
                continue
            if (original.get("user") != owner or original.get("session_id") != session.session_id
                    or original.get("session_uri") != session.uri or original.get("archive_uri") != uri):
                raise FailedPreconditionError("Archived commit ownership does not match this session")
            replay = next((a for a in attempts.values() if a.get("retry_of_task_id") == expected_task_id), None)
            if replay:
                msg = SessionCommitMsg.from_dict(replay["queue_message"])
                existing = await tracker.get(msg.task_id, **owner)
                if not existing:
                    raise FailedPreconditionError("Retry task is unavailable; retained archive requires reconciliation")
            else:
                task = await tracker.get(expected_task_id, **owner)
                if not task:
                    raise NotFoundError(expected_task_id, "task")
                if task.task_type != "session_commit" or task.resource_id != session.session_id or task.status.value != "failed":
                    raise FailedPreconditionError("Only a failed session extraction can be retried")
                if tracker._work_index.has_work(expected_task_id):
                    raise FailedPreconditionError("The previous task still owns unfinished work")
                if phase1.get("status") != "ready" or await session._archive_file_exists(uri, ".done"):
                    raise FailedPreconditionError("Archive requires reconciliation, not extraction retry")
                failure_raw = await session._viking_fs.read_file(f"{uri}/.failed.json", ctx=session.ctx)
                failure = json.loads(failure_raw)
                if failure.get("stage") != "memory_extraction" or (
                    failure.get("task_id") or original["task_id"]) != expected_task_id:
                    raise FailedPreconditionError("Failure receipt does not identify a retryable extraction")
                if not await session._read_archive_messages(uri):
                    raise FailedPreconditionError("Archive has no intact messages to retry")
                msg = SessionCommitMsg.from_dict({**original, "task_id": str(uuid4()),
                                                  "retry_of_task_id": expected_task_id})
                replay = {"retry_of_task_id": expected_task_id, "queue_message": msg.to_dict(),
                          "prior_failure_raw": failure_raw, "state": "admitted",
                          "created_at": get_current_timestamp()}
                attempts[msg.task_id] = replay
                await session._merge_archive_meta(uri, {"retry_attempts": attempts})
            task = await tracker.create("session_commit", resource_id=session.session_id,
                                        task_id=msg.task_id, meta={"archive_uri": uri,
                                        "retry_of_task_id": expected_task_id}, **owner)
            if replay["state"] == "admitted" and task.status.value == "pending" and not tracker._work_index.has_work(msg.task_id):
                await get_queue_manager().enqueue(QueueManager.SESSION_COMMIT, msg.to_dict())
                replay["state"] = "queued"
                await session._merge_archive_meta(uri, {"retry_attempts": attempts})
            return {"session_id": session.session_id, "status": "accepted", "task_id": msg.task_id,
                    "archive_uri": uri, "archived": False, "retry_of_task_id": expected_task_id}
        finally:
            await session._viking_fs._async_agfs.pathlock_release(lease)
    raise NotFoundError(expected_task_id, "archive task")


async def authorized_retry(session, msg):
    if not msg.retry_of_task_id:
        return False
    meta = await session._read_archive_meta(msg.archive_uri)
    receipt = (meta.get("retry_attempts") or {}).get(msg.task_id) or {}
    return (receipt.get("retry_of_task_id") == msg.retry_of_task_id
            and receipt.get("queue_message") == msg.to_dict()
            and msg.user == session.ctx.user.to_dict()
            and msg.session_uri == session.uri and msg.session_id == session.session_id)
