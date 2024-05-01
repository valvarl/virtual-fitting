import enum
from typing import TYPE_CHECKING, Dict, List, Optional, Union

class SequenceStatus(enum.Enum):
    """Status of a sequence."""
    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status in [
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_ABORTED,
            SequenceStatus.FINISHED_IGNORED,
        ]

    @staticmethod
    def get_finished_reason(status: "SequenceStatus") -> Union[str, None]:
        if status == SequenceStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == SequenceStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == SequenceStatus.FINISHED_IGNORED:
            # The ignored sequences are the sequences whose prompt lengths
            # are longer than the model's length cap. Therefore, the stop
            # reason should also be "length" as in OpenAI API.
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


class SequenceType(enum.Enum):
    IMAGE = enum.auto()
    CLOTH = enum.auto()


class Sequence:
    def __init__(
        self,
        seq_id: int,
        image: str,
        image_type: SequenceType,
    ) -> None:
        self.seq_id = seq_id
        self.image = image
        self.image_type = image_type

        self.status = SequenceStatus.WAITING
        self.stop_reason: Union[int, str, None] = None

    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.status)
    
    def __repr__(self) -> str:
        return (f"Sequence(seq_id={self.seq_id}, "
                f"status={self.status.name}")


class SequenceGroup:
    def __init__(
        self,
        request_id: str,
        seqs: List[Sequence],
        arrival_time: float,
    ) -> None:
        self.request_id = request_id
        self.seqs_dict = {seq.seq_id: seq for seq in seqs}

    def get_seqs(
        self,
        status: Optional[SequenceStatus] = None,
    ) -> List[Sequence]:
        return list(self.seqs_dict.values()) if status is None else [
            seq for seq in self.seqs_dict.values() if seq.status == status
        ]
    
    def get_unfinished_seqs(self) -> List[Sequence]:
        return [
            seq for seq in self.seqs_dict.values() if not seq.is_finished()
        ]

    def get_finished_seqs(self) -> List[Sequence]:
        return [seq for seq in self.seqs_dict.values() if seq.is_finished()]
    
    def num_seqs(self, status: Optional[SequenceStatus] = None) -> int:
        # Optimization. We don't need to call get_seqs if we don't need to
        # filter by states.
        if status is None:
            return len(self.seqs_dict)

        return len(self.get_seqs(status))
    
    def num_unfinished_seqs(self) -> int:
        return len(self.get_unfinished_seqs())

    def num_finished_seqs(self) -> int:
        return len(self.get_finished_seqs())

    def find(self, seq_id: int) -> Sequence:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        return self.seqs_dict[seq_id]

    def add(self, seq: Sequence) -> None:
        if seq.seq_id in self.seqs_dict:
            raise ValueError(f"Sequence {seq.seq_id} already exists.")
        self.seqs_dict[seq.seq_id] = seq

    def remove(self, seq_id: int) -> None:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        del self.seqs_dict[seq_id]

    def is_finished(self) -> bool:
        return all(seq.is_finished() for seq in self.get_seqs())
    
    def __repr__(self) -> str:
        return (f"SequenceGroup(request_id={self.request_id}, "
                f"num_seqs={len(self.seqs_dict)})")
