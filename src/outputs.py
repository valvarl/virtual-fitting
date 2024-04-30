import typing as tp

class RequestOutput:
    def __init__(
        self,
        request_id: str,
        outputs: tp.List[tp.Tuple[str, None]]
    ) -> None:
        self.request_id = request_id
        self.outputs = outputs
