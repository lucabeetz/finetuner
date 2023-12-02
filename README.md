# finetuner

finetuner helps you finetune models by automatically collecting prompts and completions from your existing GPT calls.

## Getting started

finetuner provides a wrapper around the native OpenAI client with the same base functionality.

### Storing prompts

When creating the client, you can pass a `Storage` object to set where your prompts and completions should be saved. Below is an example for the `FileStorage` that saves prompts to JSON files.

```python
from finetuner import Client
from finetuner.storage import FileStorage

storage = FileStorage(directory="example_finetune")
client = Client.for_openai(storage=storage)
```

Now, when making chat completion calls, you provide a `dataset_name` to set which dataset this inference should be added to. This is necessary to keep track of multiple different prompts.

```python
chat_completion = client.chat.completions.create(
    messages=[{"role": "user", "content": formatted_template}],
    model="gpt-4",
    temperature=0,
    dataset_name="search_classifier",
)
```

You can then use the storage object to see and retrieve the available datasets

```python
storage.list_datasets()
# Outputs: search_classifier.json

dataset = storage.get_dataset("search_classifier")
```

### Starting finetuning job

After getting a dataset, you can start a finetuning job (using [Anyscale](https://www.anyscale.com/)) with the `Finetuner` class by uploading the `Dataset` and then passing its id along with the base model to `start_job()`. You can optionally provide a validation dataset.

```python
from finetuner import OpenAI
from finetuner.dataset import FileDataset
from finetuner.finetuner import Finetuner

# Get client for anyscale to use finetuning endpoint
client = Client.for_anyscale()
finetuner = Finetuner(client=client)

train_dataset, val_dataset = dataset.split(0.8)

train_file_id = finetuner.upload_dataset(train_dataset)
val_file_id = finetuner.upload_dataset(val_dataset)

job = finetuner.start_job("meta-llama/Llama-2-7b-chat-hf", train_file_id, val_file_id)
```

You can check job status with `job.status`. It usually takes around 10-20min depending on the size and anyscale sends you an email when the job finishes or fails.

### Running evaluation

finetuner also provides an `Eval` class to evaluate the performance of models on datasets.
To use it, simply subclass it and implement the `compare` method.
Often this might just be an equality operator with some string formatting but for more complex tasks you could use an LLM call for evaluation.

```python
from finetuner.eval import Eval

class ClassificationEval(Eval):
    def compare(self, prediction: str, target: str):
        return prediction.strip() == target.strip()

# Eval result is accuracy
eval = ClassificationEval(client)
eval.run("meta-llama/Llama-2-7b-chat-h", val_dataset, temperature=0)
```


## TODO and ideas

- [x] Eval pipeline
- [ ] Prompt classes for versioning 
- [ ] Mock data generation
- [ ] Chain-of-thought mode
- [ ] CLI
