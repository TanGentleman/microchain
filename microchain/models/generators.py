from termcolor import colored

class OpenAIChatGenerator:
    def __init__(self, *, model, api_key, api_base, temperature=0.9, top_p=1, max_tokens=512, timeout=30):
        try:
            import openai
        except ImportError:
            raise ImportError("Please install OpenAI python library using pip install openai")
    
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
    
    def __call__(self, messages, stop=None):
        import openai
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
        except openai.error.OpenAIError as e:
            print(colored(f"Error: {e}", "red"))
            return "Error: timeout"
        
        output = response.choices[0].message.content.strip()

        return output
    


class OpenAITextGenerator:
    def __init__(self, *, model, api_key, api_base, temperature=0.9, top_p=1, max_tokens=512):
        try:
            import openai
        except ImportError:
            raise ImportError("Please install OpenAI python library using pip install openai")
    
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
    
    def __call__(self, prompt, stop=None):
        import openai
        assert isinstance(prompt, str), "prompt must be a string https://platform.openai.com/docs/guides/text-generation/chat-completions-api"

        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stop=stop
            )
        except openai.error.OpenAIError as e:
            print(colored(f"Error: {e}", "red"))
            return "Error: timeout"
        
        output = response.choices[0].text.strip()

        return output
    
class MistralChatGenerator:
    def __init__(self, *, model, api_key, temperature=0.7, top_p=1, max_tokens=512, timeout=30):
        try:
            from mistralai.client import MistralClient
        except ImportError:
            raise ImportError("Please install mistral using pip install mistralai")
    
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.client = MistralClient(api_key=self.api_key, timeout=timeout)
        
    def __call__(self, messages, stop=None):
        from mistralai.models.chat_completion import ChatMessage
        chat_response = self.client.chat(
        model=self.model,
        # convert messages from OpenAI format to Mistral format
        messages=messages,
        temperature=self.temperature,   
        max_tokens=self.max_tokens,
        top_p=self.top_p,
        )
        output = chat_response.choices[0].message.content.strip()

        return output
    
class MistralTextGenerator:
    def __init__(self, *, model, api_key, temperature=0.5, top_p=1, max_tokens=500):
        try:
            from mistralai.client import MistralClient
        except ImportError:
            raise ImportError("Please install mistral using pip install mistralai")
    
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.client = MistralClient(api_key=self.api_key)
    
    def __call__(self, prompt, stop=None):
        from mistralai.models.chat_completion import ChatMessage
        message_history = None
        if type(prompt) != str:
            print(f'{len(prompt)} messages in API call.')
            message_history = [ChatMessage(role=message["role"], content=message["content"]) for message in prompt]
        chat_response = self.client.chat(
        model=self.model,
        messages=message_history or [ChatMessage(role="user", content=prompt)],
        temperature=self.temperature,   
        max_tokens=self.max_tokens,
        top_p=self.top_p,
        )
        
        output = chat_response.choices[0].message.content.strip()
        if stop == ['\n']:
            # NOTE: The output contains the full sequence of commands, separated by newlines.
            # print('Returning first command from response.')
            output = output.split('\n')[0]
        return output
