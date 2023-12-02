import os
from typing import Mapping, Optional, Union

import httpx
from ._constants import ANYSCALE_BASE_URL
from openai import OpenAI, Timeout
from openai._constants import DEFAULT_MAX_RETRIES
from openai._types import NOT_GIVEN, NotGiven
from openai.resources import Chat
from openai.resources.chat.completions import Completions
from openai.types.chat import ChatCompletion

from finetuner.dataset import Dataset


class CompletionsWrapper(Completions):
    dataset: Optional[Dataset]

    def __init__(self, client: OpenAI, dataset: Optional[Dataset]) -> None:
        super().__init__(client)
        self.dataset = dataset

    def create(self, *args, **kwargs) -> ChatCompletion:
        chat_completion = super().create(*args, **kwargs)

        res = chat_completion.choices[0].message.content
        if self.dataset is not None and res:
            self.dataset.append_completion(kwargs, res)
            self.dataset.save()

        return chat_completion


class ChatWrapper(Chat):
    def __init__(self, client: OpenAI, dataset: Optional[Dataset]) -> None:
        super().__init__(client)
        self.completions = CompletionsWrapper(client, dataset)


class OpenAIWrapper(OpenAI):
    chat: ChatWrapper
    storage: Optional[Dataset]

    @classmethod
    def for_openai(cls, **kwargs):
        return cls(**kwargs)

    @classmethod
    def for_anyscale(cls, **kwargs):
        return cls(use_anyscale=True, **kwargs)

    def __init__(
        self,
        *,
        dataset: Optional[Dataset] = None,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.Client | None = None,
        use_anyscale: bool = False,
    ):
        if use_anyscale:
            base_url = ANYSCALE_BASE_URL
            try:
                api_key = os.environ["ANYSCALE_API_KEY"]
            except KeyError:
                raise ValueError(
                    "You must set the ANYSCALE_API_KEY environment variable to use Anyscale's API."
                )

        super().__init__(
            api_key=api_key,
            organization=organization,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
        )
        self.dataset = dataset
        self.chat = ChatWrapper(self, self.dataset)
