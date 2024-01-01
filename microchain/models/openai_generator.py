from microchain.models.generator import Generator
from termcolor import colored
try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Error! Try pip install openai before using this generator")
from openai import OpenAIError

class OpenAIChatGenerator(Generator):
    def __init__(self, *, model, api_key, api_base=None, temperature=0.9, top_p=1, max_tokens=512, timeout=30):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
    
    def __call__(self, messages, stop=None):
        assert isinstance(messages, list), "messages must be a list of messages https://platform.openai.com/docs/guides/text-generation/chat-completions-api"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stop=stop,
                timeout=self.timeout
            )
        except OpenAIError as e:
            print(colored(f"Error: {e}", "red"))
            return "Error: timeout"
        
        total_tokens = response.usage.total_tokens or 0
        print(f'Used {total_tokens} tokens')
        output = response.choices[0].message.content
        if output is None:
            print('openai_generator returned None. Replacing with empty string.')
            output = ''
        return output, total_tokens
 