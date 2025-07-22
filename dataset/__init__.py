from .ds import Task, Data, Dataset
from .docile import DocILE
from .sroie import SROIE


dataset_map = {
    "DocILE": DocILE,
    "SROIE": SROIE
}

class MultiDataset():
    def __init__(
        self,
        datasets: list[str],
        task: list[Task],
        split: str
    ) -> None:
        self.data: list[Data] = []
        self._load_data(datasets, task, split)

    def _get_data(self, task: Task, data: Data) -> list:
        pass

    def _load_data(
        self,
        datasets: list[str],
        task: list[Task],
        split: str
    ) -> None:
        for dataset in datasets:
            dataset_obj: Dataset = dataset_map[dataset]
            dataset_tasks: list[Task] = dataset_obj.TASKS


__all__ = [
    "Task",
    "DocILE",
    "SROIE",
    "MultiDataset"
]