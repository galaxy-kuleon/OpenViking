import asyncio
import threading
import time

from openviking.models.vlm import ConcurrencyLimitedVLM, VLMBase


class _TrackingVLM(VLMBase):
    def __init__(self):
        super().__init__({"provider": "test", "model": "test"})
        self._state_lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def _enter(self):
        with self._state_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def _leave(self):
        with self._state_lock:
            self.active -= 1

    def get_completion(self, *args, **kwargs):
        self._enter()
        try:
            time.sleep(0.05)
            return "ok"
        finally:
            self._leave()

    async def get_completion_async(self, *args, **kwargs):
        self._enter()
        try:
            await asyncio.sleep(0.05)
            return "ok"
        finally:
            self._leave()

    get_vision_completion = get_completion
    get_vision_completion_async = get_completion_async


def test_async_limit_is_shared_across_worker_event_loops():
    delegate = _TrackingVLM()
    limited = ConcurrencyLimitedVLM(delegate, max_concurrent=1)
    barrier = threading.Barrier(3)

    def run_call():
        barrier.wait()
        asyncio.run(limited.get_completion_async("hello"))

    workers = [threading.Thread(target=run_call) for _ in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=2)

    assert all(not worker.is_alive() for worker in workers)
    assert delegate.max_active == 1
