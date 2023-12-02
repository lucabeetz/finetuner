from abc import ABC, abstractmethod

from finetuner.dataset import Dataset


class Eval(ABC):
    dataset: Dataset

    @abstractmethod
    def compare(self, prediction: str, target: str) -> bool:
        pass

    @abstractmethod
    def get_prediction(self, input: str) -> str:
        pass

    def run(self) -> float:
        correct = 0
        for sample in self.dataset:
            prediction = self.get_prediction(sample.input)
            if self.compare(prediction, sample.target):
                correct += 1
        return correct / len(self.dataset)
