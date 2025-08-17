from .ds import Dataset, Data, Task
from typing import ClassVar
import json


class NNTS_KIE(Dataset):
    TASKS: ClassVar[list[Task]] = [Task.KIE]

    def __init__(
        self,
        tasks: list[Task],
        split: str
    ) -> None:
        super().__init__(tasks=tasks, split=split)

    def _download() -> None:
        pass

    def _load_data(self) -> None:
        pass