# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py

import asyncio
from functools import partial
import os
import time
from typing import (Any, AsyncIterator, Callable, Dict, Iterable, List,
                    Optional, Set, Tuple, Type, Union)

from virtual_fitting.logger import init_logger
from virtual_fitting.engine.arg_utils import AsyncEngineArgs
from virtual_fitting.engine.preproc_engine import PreprocEngine
from virtual_fitting.outputs import RequestOutput


logger = init_logger(__name__)
ENGINE_ITERATION_TIMEOUT_S = int(
    os.environ.get("ENGINE_ITERATION_TIMEOUT_S", "300"))

def _raise_exception_on_finish(
        task: asyncio.Task, error_callback: Callable[[Exception],
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

    def put(self, item: Union[RequestOutput, Exception]) -> None:
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
        self._request_streams: Dict[str, AsyncStream] = {}
        self._finished_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream,
                                                dict]] = asyncio.Queue()
        self.new_requests_event = asyncio.Event()
        
    def process_request_output(self,
                               request_output: RequestOutput,
                               *,
                               verbose: bool = False) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id

        self._request_streams[request_id].put(request_output)
        if request_output.finished:
            if verbose:
                logger.info(f"Finished request {request_id}.")
            self.abort_request(request_id)
    
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
        
        return stream
    
    def abort_request(self, request_id: str, *, verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info(f"Aborted request {request_id}.")

        self._finished_requests.put_nowait(request_id)

        if request_id not in self._request_streams or self._request_streams[
                request_id].finished:
            # The request has already finished or been aborted.
            return

        self._request_streams[request_id].finish()

    def get_new_and_finished_requests(self) -> Tuple[List[Dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[Dict] = []
        finished_requests: Set[str] = set()

        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_requests.add(request_id)
            self._request_streams.pop(request_id, None)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)

        return new_requests, finished_requests

    
    async def wait_for_new_requests(self):
        if not self.has_new_requests():
            await self.new_requests_event.wait()
        self.new_requests_event.clear()

    def has_new_requests(self):
        return not self._new_requests.empty()
    

class _AsyncPreprocEngine(PreprocEngine):

    async def step_async(self) -> List[RequestOutput]:
        pass

    async def add_request_async(
        self,
        request_id: str,
        image: Optional[str] = None,
        cloth: Optional[str] = None,
        arrival_time: Optional[float] = None):
        if arrival_time is None:
            arrival_time = time.time()

        return self.add_request(request_id,
                                image=image,
                                cloth=cloth,
                                arrival_time=arrival_time)

    
class AsyncPreprocEngine:

    _engine_class: Type[_AsyncPreprocEngine] = _AsyncPreprocEngine

    def __init__(
        self, 
        *args,
        log_requests: bool = True,
        start_engine_loop: bool = True, 
        **kwargs
    ) -> None:
        self.log_requests = log_requests
        self.engine = self._init_engine(*args, **kwargs)

        self.background_loop: Optional[asyncio.Future] = None
        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage
        # collected
        self._background_loop_unshielded: Optional[asyncio.Task[Any]] = None
        self.start_engine_loop = start_engine_loop
        
        # Lazy initialized fields
        self._request_tracker: RequestTracker
        
    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True
    ) -> "AsyncPreprocEngine":
        engine_config = engine_args.create_engine_config()
        
        engine = cls(
            **engine_config.to_dict(),
            start_engine_loop=start_engine_loop
        )
        
        return engine
    
    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
                and self._background_loop_unshielded is not None
                and not self._background_loop_unshielded.done())
    
    @property
    def is_stopped(self) -> bool:
        return self.errored or (self.background_loop is not None and
                                self._background_loop_unshielded is not None
                                and self._background_loop_unshielded.done())

    @property
    def errored(self) -> bool:
        return self._errored_with is not None

    def set_errored(self, exc: Exception) -> None:
        self._errored_with = exc

    def _error_callback(self, exc: Exception) -> None:
        self.set_errored(exc)
        self._request_tracker.propagate_exception(exc)
    
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

    def _init_engine(self, *args, **kwargs) -> _AsyncPreprocEngine:
        return self._engine_class(*args, **kwargs)

    async def run_engine_loop(self):
        has_requests_in_progress = False
        while True:
            if not has_requests_in_progress:
                logger.debug("Waiting for new requests...")
                await self._request_tracker.wait_for_new_requests()
                logger.debug("Got new requests!")

            # Abort if iteration takes too long due to unrecoverable errors.
            try:
                has_requests_in_progress = await asyncio.wait_for(
                    self.engine_step(), ENGINE_ITERATION_TIMEOUT_S)
            except asyncio.TimeoutError as exc:
                logger.error(
                    "Engine iteration timed out. This should never happen!")
                self.set_errored(exc)
                raise
            await asyncio.sleep(0)
    
    async def engine_step(self) -> bool:
        """Kick the engine to process the waiting requests.

        Returns True if there are in-progress requests."""

        new_requests, finished_requests = (
            self._request_tracker.get_new_and_finished_requests())

        for new_request in new_requests:
            # Add the request into the engine's waiting queue.
            try:
                await self.engine.add_request_async(**new_request)
            except ValueError as e:
                # TODO: use a specific error for failed validation
                self._request_tracker.process_exception(
                    new_request["request_id"],
                    e,
                    verbose=self.log_requests,
                )

        if finished_requests:
            await self._engine_abort(finished_requests)

        request_outputs = await self.engine.step_async()

        # Put the outputs into the corresponding streams.
        for request_output in request_outputs:
            self._request_tracker.process_request_output(
                request_output, verbose=self.log_requests)

        return len(request_outputs) > 0
    
    async def _engine_abort(self, request_ids: Iterable[str]):
        self.engine.abort_request(request_ids)

    async def add_request(
        self,
        request_id: str,
        image: Optional[str] = None,
        cloth: Optional[str] = None,
        arrival_time: Optional[float] = None,
    ) -> AsyncStream:
        if image is not None or cloth is not None:
            logger.info(
                "Received request %s: image: %r, "
                "cloth: %s", request_id, image, cloth)
        else:
            raise RuntimeError("Empty request: no image or clothing provided.")

        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")

        if arrival_time is None:
            arrival_time = time.time()

        stream = self._request_tracker.add_request(
            request_id,
            image=image,
            cloth=cloth,
            arrival_time=arrival_time,
        )

        return stream
        
    async def preprocess(
        self, 
        request_id: str,
        image: Optional[str] = None,
        cloth: Optional[str] = None,
    ) -> AsyncIterator[RequestOutput]:
        # Preprocess the request.
        arrival_time = time.time()

        try:
            stream = await self.add_request(
                request_id,
                image=image,
                cloth=cloth,
                arrival_time=arrival_time,
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
        
    