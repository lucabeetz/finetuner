import os
import json
from pydantic import BaseModel, Field
from typing import List, Iterator

from abc import ABC, abstractmethod


class Dataset(ABC, BaseModel):
    completions: List[dict] = Field(default_factory=list)

    @abstractmethod
    def append_completion(self, input_kwargs: dict, completion: str):
        pass

    def __iter__(self) -> Iterator:
        return iter(self.completions)

    def __len__(self) -> int:
        return len(self.completions)

    def to_finetuning_format(self) -> List[dict]:
        formatted_completions = []
        for completion in self.completions:
            messages = completion["input_kwargs"]["messages"]
            # for finetuning we have to add the completion as an assistant message
            messages.append({"role": "assistant", "content": completion["completion"]})
            formatted_completions.append({"messages": messages})
        return formatted_completions

    def split(self, train_ratio: float = 0.8) -> tuple["Dataset", "Dataset"]:
        train_len = int(len(self.completions) * train_ratio)
        train_completions = self.completions[:train_len]
        val_completions = self.completions[train_len:]
        return MemoryDataset(completions=train_completions), MemoryDataset(
            completions=val_completions
        )


class MemoryDataset(Dataset):
    def append_completion(self, input_kwargs: dict, completion: str):
        self.completions.append(
            {"input_kwargs": input_kwargs, "completion": completion}
        )


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

    def append_completion(self, input_kwargs: dict, completion: str):
        self.completions.append(
            {"input_kwargs": input_kwargs, "completion": completion}
        )
        self.save()

    def save(self):
        with open(self.file_path, "w") as f:
            json.dump(self.completions, f)
