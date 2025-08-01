from .ds import Task, Data, Dataset
from .docile import DocILE
from .sroie import SROIE


class MultiDataset():
    def __init__(
        self,
        selections: list[tuple[Dataset, list[Task]]],
        split: str
    ) -> None:
        self.data: list[Data] = []
        self._load_data(selections, split)

    def __iter__(self):
        return self.data.__iter__()

    def _load_data(
        self,
        selections: list[tuple[Dataset, list[Task]]],
        split: str
    ) -> None:
        for selection in selections:
            dataset: Dataset = selection[0]
            tasks: list[Task] = selection[1]

            self.data += dataset(tasks,split).data


__all__ = [
    "Task",
    "DocILE",
    "SROIE",
    "MultiDataset"
]