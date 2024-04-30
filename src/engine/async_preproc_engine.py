# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py

import asyncio
import os
import typing as tp

from virtual_fitting.engine.arg_utils import AsyncEngineArgs
from virtual_fitting.outputs import RequestOutput


def _raise_exception_on_finish(
        task: asyncio.Task, error_callback: tp.Callable[[Exception],
                                                     None]) -> None:
    msg = ("Task finished unexpectedly. This should never happen! "
           "Please open an issue on Github.")

    exception = None
    try:
        task.result()
        # NOTE: This will be thrown if task exits normally (which it should not)
        raise AsyncEngineDeadError(msg)
    except Exception as e:
        exception = e
        logger.error("Engine background task failed", exc_info=e)
        error_callback(exception)
        raise AsyncEngineDeadError(
            msg + " See stack trace above for the actual cause.") from e
        
class AsyncEngineDeadError(RuntimeError):
    pass


class AsyncStream:
    """A stream of RequestOutputs for a request that can be
    iterated over asynchronously."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False

    def put(self, item: tp.Union[RequestOutput, Exception]) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopAsyncIteration())
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> RequestOutput:
        result = await self._queue.get()
        if isinstance(result, Exception):
            raise result
        return result


class RequestTracker:
    """Synchronous abstraction for tracking requests."""
    def __init__(self):
        self._new_requests: asyncio.Queue[Tuple[AsyncStream,
                                                dict]] = asyncio.Queue()
        self.new_requests_event = asyncio.Event()
    
    def add_request(self, request_id: str,
                    **engine_add_request_kwargs):
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        stream = AsyncStream(request_id)
        self._new_requests.put_nowait((stream, {
            "request_id": request_id,
            **engine_add_request_kwargs
        }))

        self.new_requests_event.set()
    
    async def wait_for_new_requests(self):
        if not self.has_new_requests():
            await self.new_requests_event.wait()
        self.new_requests_event.clear()

    def has_new_requests(self):
        return not self._new_requests.empty()

    
class AsyncPreprocEngine:
    def __init__(
        self, 
        start_engine_loop: bool = True, 
        **kwargs
    ):
        self.background_loop: tp.Optional[asyncio.Future] = None
        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage
        # collected
        self._background_loop_unshielded: tp.Optional[asyncio.Task[Any]] = None
        self.start_engine_loop = start_engine_loop
        
        # Lazy initialized fields
        self._request_tracker: RequestTracker
        
    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True):
        
        engine = cls(
            start_engine_loop=start_engine_loop
        )
        
        return engine
    
    async def add_request(self):
        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")
    
    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
                and self._background_loop_unshielded is not None
                and not self._background_loop_unshielded.done())
    
    def start_background_loop(self) -> None:
        """Start the background loop."""

        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        # Initialize the RequestTracker here so it uses the right event loop.
        self._request_tracker = RequestTracker()
        
        self._background_loop_unshielded = asyncio.get_event_loop(
        ).create_task(self.run_engine_loop())
        self._background_loop_unshielded.add_done_callback(
            partial(_raise_exception_on_finish,
                    error_callback=self._error_callback))
        self.background_loop = asyncio.shield(self._background_loop_unshielded)
    
    async def engine_loop(self):
        while True:
            pass
        
    async def preprocess(self):
        # Preprocess the request.
        arrival_time = time.time()

        try:
            stream = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                prompt_token_ids=prompt_token_ids,
                arrival_time=arrival_time,
                lora_request=lora_request,
                multi_modal_data=multi_modal_data,
            )

            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:
            # If there is an exception or coroutine is cancelled, abort the
            # request.
            self._abort(request_id)
            raise e
        
        
    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if not self.is_running:
            raise AsyncEngineDeadError(
                "Background loop is not running. If it was running, "
                "inspect the output to find the stacktrace of the "
                "error that caused the background loop to stop "
                "(AsyncEngineDeadError).")

        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        self._request_tracker.abort_request(request_id,
                                            verbose=self.log_requests)
        
    