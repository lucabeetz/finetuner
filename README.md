# finetuner

*finetuner* helps you finetune models by automatically collecting prompts and completions from your existing GPT calls.

## Getting started

*finetuner* provides a wrapper around the native client with the same functionality.

When creating the client, you can pass a `Dataset` object to define how your prompts should be saved. Below is an example for the `FileDataset` that saves prompts to a JSON file.

```python
from finetuner import Client
from finetuner.dataset import FileDataset

dataset = FileDataset(file_path="search_classifier.json")
client = Client.for_openai(dataset=dataset)
```

After creating a dataset, you can start a finetuning job (using [Anyscale](https://www.anyscale.com/)) with the `Finetuner` class by uploading the `Dataset` and then passing its id along with the base model to `start_job()`.

```python
from finetuner import OpenAI
from finetuner.dataset import FileDataset
from finetuner.finetuner import Finetuner

client = Client.for_anyscale()

dataset = FileDataset.from_file("search_classifier.json")
finetuner = Finetuner(client=client)

dataset_id = finetuner.upload_dataset(dataset)
job = finetuner.start_job("meta-llama/Llama-2-7b-chat-hf", dataset_id)
```

## TODO and ideas

- [ ] Eval pipeline
- [ ] Prompt classes for versioning 
- [ ] Mock data generation
- [ ] Chain-of-thought mode
- [ ] Finetuning CLI
