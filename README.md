# finetuner

finetuner helps you finetune models by automatically collecting prompts and completions from your existing GPT calls.

## Getting started

finetuner provides a wrapper around the native OpenAI client with the same functionality, so all you have to do is change `from openai import OpenAI` to `from finetuner import OpenAI`.

When creating the OpenAI client, you can pass a `Storage` object to define how your prompts should be saved. Below is an example for the `FileStorage` that saves prompts to a JSON file.

```python
from finetuner import OpenAI
from finetuner.storage import FileStorage

storage = FileStorage(file_path="search_classifier.json")
client = OpenAI(storage=storage)
```

## TODO and ideas

- [ ] Finetuning CLI
- [ ] Eval pipeline
- [ ] Prompt classes for versioning 
- [ ] Mock data generation
- [ ] Chain-of-thought mode
