import time
from typing import (Iterable, Optional, Union)

from virtual_fitting.core.scheduler import Scheduler
from virtual_fitting.preprocessing.processor import ImageProcessor
from virtual_fitting.utils import Counter

class PreprocEngine:
    def __init__(self):
        self.counter = Counter()

    def add_request(
        self,
        request_id: str,
        image: Optional[str] = None,
        cloth: Optional[str] = None,
        arrival_time: Optional[float] = None,
    ) -> None:
        if arrival_time is None:
            arrival_time = time.time()
        
        pass

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        pass

