from pydantic import BaseModel, Field
from typing import List, Iterator


class Dataset(BaseModel):
    completions: List[dict] = Field(default_factory=list)

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
        return Dataset(completions=train_completions), Dataset(
            completions=val_completions
        )
