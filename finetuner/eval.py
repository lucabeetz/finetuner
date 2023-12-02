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
        self, model: str, messages: List[ChatCompletionMessageParam], **kwargs
    ) -> Optional[str]:
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=model,
            **kwargs,
        )
        return chat_completion.choices[0].message.content

    def run(
        self, model: str, dataset: Dataset, verbose: bool = False, **kwargs
    ) -> Optional[float]:
        if not dataset:
            print("The dataset is empty")
            return

        correct = 0
        for sample in tqdm(dataset):
            messages = sample["input_kwargs"]["messages"]
            prediction = self.get_prediction(model, messages, **kwargs)
            if prediction is None:
                continue
            if self.compare(prediction, sample["completion"]):
                correct += 1
            if verbose:
                print(f"Prediction: {prediction} - Target: {sample['completion']}")
        acc = correct / len(dataset)
        if verbose:
            print(f"Accuracy: {acc}")
        return acc
