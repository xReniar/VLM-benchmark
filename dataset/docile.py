from .ds import Dataset, Data, Task
import json


class DocILE(Dataset):
    def __init__(
        self,
        task: Task,
        split: str
    ) -> None:
        super().__init__(task=task, split=split)

    def _load_data(self):
        pass