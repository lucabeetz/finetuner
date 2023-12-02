# finetuner

finetuner helps you finetune models by automatically collecting prompts and completions from your existing GPT calls.

## Getting started

finetuner provides a wrapper around the native OpenAI client with the same functionality, so all you have to do is change `from openai import OpenAI` to `from finetuner import OpenAI`.

When creating the OpenAI client, you can pass a `Dataset` object to define how your prompts should be saved. Below is an example for the `FileDataset` that saves prompts to a JSON file.

```python
from finetuner import OpenAI
from finetuner.dataset import FileDataset

dataset = FileDataset(file_path="search_classifier.json")
client = OpenAI(dataset=dataset)
```

After creating a dataset, you can start a finetuning job (using [Anyscale](https://www.anyscale.com/)) with the `Finetuner` class by passing it a `Dataset` and the model you want to fine tune.

```python
from finetuner import OpenAI
from finetuner.dataset import FileDataset
from finetuner.finetune import Finetuner

client = OpenAI(use_anyscale=True)

# Load previously created dataset
dataset = FileDataset.from_file("search_classifier.json")
finetune = Finetune(model="meta-llama/Llama-2-7b-chat-hf", dataset=dataset, client=client)

finetune.upload_dataset()
finetune.start_job()
```

## TODO and ideas

- [ ] Eval pipeline
- [ ] Prompt classes for versioning 
- [ ] Mock data generation
- [ ] Chain-of-thought mode
- [ ] Finetuning CLI
