from dotenv import load_dotenv
from tqdm import tqdm

from finetuner import OpenAI
from finetuner.storage import FileStorage

load_dotenv()

storage = FileStorage(file_path="search_classifier.json")
client = OpenAI(storage=storage, use_anyscale=True)

search_classifier_template = """You are helping an AI assistant decide whether a google search for real-time information is necessary to correctly answer a user's query.
For this, you have to output either 'Y' or 'N' depending on the user query below.

User query: {user_input}

Output:"""


def search_classifier(user_input: str) -> str | None:
    formatted_template = search_classifier_template.format(user_input=user_input)

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": formatted_template}],
        model="meta-llama/Llama-2-7b-chat-hf",
    )

    result = chat_completion.choices[0].message.content
    return result


if __name__ == "__main__":
    example_user_inputs = [
        "Hey",
        "What's the weather in new york?",
        "good morning",
        "how did the giants play yesterday",
        "I'm hungry",
        "how can i reverse a list in python",
        "what is a fibonacci sequence",
        "what are recent news about AI",
        "what's the current apple stock price",
        "thanks, that's helpful",
    ]

    for user_input in tqdm(example_user_inputs):
        search_classifier(user_input)
