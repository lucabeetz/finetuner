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

After creating a dataset, you can start a finetuning job (using [Anyscale](https://www.anyscale.com/)) with the `Finetuner` class by passing the `Dataset` you want to finetune on to `start_job()`.

```python
from finetuner import OpenAI
from finetuner.dataset import FileDataset
from finetuner.finetune import Finetuner

client = Client.for_anyscale()

# Load previously created dataset
dataset = FileDataset.from_file("search_classifier.json")
finetune = Finetune(model="meta-llama/Llama-2-7b-chat-hf", client=client)

finetune.start_job(dataset)
```

## TODO and ideas

- [ ] Eval pipeline
- [ ] Prompt classes for versioning 
- [ ] Mock data generation
- [ ] Chain-of-thought mode
- [ ] Finetuning CLI
