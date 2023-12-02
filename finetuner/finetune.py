import json
import uuid
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from openai.types.fine_tuning import FineTuningJob

from finetuner import OpenAI
from finetuner.dataset import Dataset


class Finetune(BaseModel):
    model: str
    dataset: Dataset
    client: OpenAI

    file_id: Optional[str] = None
    job: Optional[FineTuningJob] = None

    class Config:
        arbitrary_types_allowed = True

    def _validate_dataset(self):
        pass

    def upload_dataset(self):
        """Upload the dataset to the finetuning endpoint"""

        temp_file_path = None
        try:
            self._validate_dataset()
            formatted_dataset = self.dataset.to_finetuning_format()

            temp_dir = Path(".finetuner")
            temp_dir.mkdir(exist_ok=True)
            temp_file_path = temp_dir / f"{uuid.uuid4()}.jsonl"

            with temp_file_path.open("w") as file:
                for sample in formatted_dataset:
                    file.write(json.dumps(sample) + "\n")

            with temp_file_path.open("rb") as file:
                uploaded_file = self.client.files.create(file=file, purpose="fine-tune")
                self.file_id = uploaded_file.id

            print(f"Uploaded dataset: {self.file_id}")
        except Exception as e:
            print(f"Error during dataset upload: {e}")

        finally:
            if temp_file_path and temp_file_path.exists():
                temp_file_path.unlink()

    def start_job(self):
        """Start the finetuning job"""

        if not self.file_id:
            raise ValueError("Dataset not uploaded")
        self.job = self.client.fine_tuning.jobs.create(
            model=self.model,
            training_file=self.file_id,
        )
        print(f"Started job: {self.job.id}")

    def print_status(self):
        """Print the status of the finetuning job"""

        if self.job:
            job = self.client.fine_tuning.jobs.retrieve(self.job.id)
            print(f"Job status: {job.status}")
        else:
            print("No job started")
