from microchain.models.generator import Generator
from mistralai.models.chat_completion import ChatMessage
from termcolor import colored
try:
    from mistralai.client import MistralClient
except ImportError:
    raise ImportError("Please install mistral using pip install mistralai")

class MistralChatGenerator(Generator):
    def __init__(self, *, model, api_key, temperature=0.3, top_p=1, max_tokens=512):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.client = MistralClient(api_key=self.api_key)

    def __call__(self, context, stop=None):
        message_history = None
        if type(context) != str:
            print(f'{len(context)} messages in API call.')
            message_history = [ChatMessage(role=message["role"], content=message["content"]) for message in context]
        chat_response = self.client.chat(
        model=self.model,
        messages=message_history or [ChatMessage(role="user", content=context)],
        temperature=self.temperature,   
        max_tokens=self.max_tokens,
        top_p=self.top_p,
        )
        total_tokens = chat_response.usage.total_tokens or 0
        print(f'Used {total_tokens} tokens')
        output = chat_response.choices[0].message.content.strip()

        if stop == ['\n']:
            # NOTE: The output contains the full sequence of commands, separated by newlines.
            # print('Returning first command from response.')
            # output = output.split('\n')[0]
            pass
        return output, total_tokens
