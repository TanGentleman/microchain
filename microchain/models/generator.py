from abc import ABC, abstractmethod

class Generator(ABC):
    @abstractmethod
    def __init__(self, model, api_key, temperature, top_p, max_tokens, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, messages, *args, **kwargs):
        pass
