import os
import json
from pydantic import BaseModel

from abc import ABC, abstractmethod


class Storage(ABC, BaseModel):
    @abstractmethod
    def append_completion(self, input_kwargs: dict, completion: str):
        pass


class FileStorage(Storage):
    file_path: str

    def append_completion(self, input_kwargs: dict, completion: str):
        # read json from file if it exists
        completions = []
        if not os.path.exists(self.file_path):
            completions = []
        else:
            with open(self.file_path, "r") as f:
                try:
                    completions = json.load(f)
                except json.JSONDecodeError:
                    # if file is empty
                    completions = []

        completions.append(
            {
                "input_kwargs": {
                    "messages": input_kwargs["messages"],
                    "model": input_kwargs["model"],
                },
                "completion": completion,
            }
        )

        with open(self.file_path, "w") as f:
            json.dump(completions, f)
