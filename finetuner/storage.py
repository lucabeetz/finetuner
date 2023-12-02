import os
import json
from abc import ABC, abstractmethod
from typing import Optional

from finetuner.dataset import Dataset


class Storage(ABC):
    @abstractmethod
    def list_datasets(self):
        pass

    @abstractmethod
    def append_to_dataset(self, dataset_name: str, input_kwargs: dict, completion: str):
        pass

    @abstractmethod
    def get_dataset(self, dataset_name: str):
        pass


class FileStorage(Storage):
    directory: str

    def __init__(self, directory: str):
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def list_datasets(self):
        for file in os.listdir(self.directory):
            if file.endswith(".json"):
                print(file)

    def append_to_dataset(self, dataset_name: str, input_kwargs: dict, completion: str):
        file_path = os.path.join(self.directory, f"{dataset_name}.json")
        completions = []
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    completions = json.load(f)
                except json.JSONDecodeError:
                    pass
        completions.append({"input_kwargs": input_kwargs, "completion": completion})
        with open(file_path, "w") as f:
            json.dump(completions, f)

    def get_dataset(self, dataset_name: str) -> Optional[Dataset]:
        file_path = os.path.join(self.directory, f"{dataset_name}.json")
        completions = []
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    completions = json.load(f)
                except json.JSONDecodeError:
                    print("The dataset is empty or does not exist")
                    return None
        return Dataset(completions=completions)
