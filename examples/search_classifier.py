from dotenv import load_dotenv
from tqdm import tqdm

from finetuner import Client
from finetuner.dataset import FileDataset

load_dotenv()

dataset = FileDataset(file_path="search_classifier.json")
client = Client.for_anyscale(dataset=dataset)

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
        "what's the latest in machine learning?",
        "how to make a cake?",
        "tell me a joke",
        "what's the time?",
        "how's the weather today?",
        "what's the latest news about the pandemic?",
        "what's the highest mountain in the world?",
        "who won the last world cup?",
        "how to lose weight?",
        "what's the capital of Australia?",
        "who is the president of the United States?",
        "how to make pizza at home?",
        "tell me a fun fact",
        "what's the distance to the moon?",
    ]

    for user_input in tqdm(example_user_inputs):
        search_classifier(user_input)
