import typing as tp

class RequestOutput:
    def __init__(
        self,
        request_id: str,
        outputs: tp.List[tp.Tuple[str, None]],
        finished: bool,
    ) -> None:
        self.request_id = request_id
        self.outputs = outputs
        self.finished = finished

    def __repr__(self) -> str:
        return (f"RequestOutput(request_id={self.request_id}, "
                f"outputs={self.outputs}, "
                f"finished={self.finished}")
