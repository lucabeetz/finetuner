from abc import ABC, abstractmethod
from typing import List, Optional

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from tqdm import tqdm

from finetuner import Client
from finetuner.dataset import Dataset


class Eval(ABC, BaseModel):
    client: Client

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def compare(self, prediction: str, target: str) -> bool:
        pass

    def get_prediction(
        self, model: str, messages: List[ChatCompletionMessageParam]
    ) -> Optional[str]:
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=model,
        )
        return chat_completion.choices[0].message.content

    def run(self, model: str, dataset: Dataset) -> float:
        correct = 0
        for sample in tqdm(dataset):
            prediction = self.get_prediction(model, sample["input_kwargs"]["messages"])
            if prediction is None:
                continue
            if self.compare(prediction, sample["completion"]):
                correct += 1
        return correct / len(dataset)
