import os
import json
from pydantic import BaseModel, Field
from typing import List, Iterator

from abc import ABC, abstractmethod


class Dataset(ABC, BaseModel):
    completions: List[dict] = Field(default_factory=list)

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def append_completion(self, input_kwargs: dict, completion: str):
        pass

    @abstractmethod
    def __iter__(self) -> Iterator:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class FileDataset(Dataset):
    file_path: str

    @classmethod
    def from_file(cls, file_path: str):
        completions = []
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    completions = json.load(f)
                except json.JSONDecodeError:
                    pass
        return cls(file_path=file_path, completions=completions)

    def save(self):
        with open(self.file_path, "w") as f:
            json.dump(self.completions, f)

    def __iter__(self):
        return iter(self.completions)

    def __len__(self):
        return len(self.completions)

    def append_completion(self, input_kwargs: dict, completion: str):
        self.completions.append(
            {"input_kwargs": input_kwargs, "completion": completion}
        )
