import os

from microchain import MistralTextGenerator, LLM, Function, Engine, Agent
from microchain.functions import Reasoning, Stop

EXAMPLE = """Example: What is 76*7+3*(5+7)?
>> Reasoning("I need to reason step-by-step")
>> Product(76, 7)
>> Sum(5, 7)
>> Product(3, 12)
>> Sum(532, 36)
> Stop()"""
# QUERY = "What is (3 + 2) × 4 ÷ (6 - 1) + 2²?"
# QUERY = "What is 88 + 2 × 5 - 3 × 4 + 3²?"
QUERY = "What is 2² + 3³ + 4⁴?"
# QUERY = "What is 2² + (4-3)³"
# QUERY = "What is 2²*(7³+4)-1?"

class Sum(Function):
    @property
    def description(self):
        return "Use this function to compute the sum of two constants."
    
    @property
    def example_args(self):
        return [2, 2]
    
    def __call__(self, a: float, b: float):
        return a + b

class Subtraction(Function):
    @property
    def description(self):
        return "Use this function to compute the subtraction of two constants."
    
    @property
    def example_args(self):
        return [2, 2]
    
    def __call__(self, a: float, b: float):
        return a - b

class Product(Function):
    @property
    def description(self):
        return "Use this function to compute the product of two constants."
    
    @property
    def example_args(self):
        return [2, 2]
    
    def __call__(self, a: float, b: float):
        return a * b

class Division(Function):
    @property
    def description(self):
        return "Use this function to compute the division of two constants."
    
    @property
    def example_args(self):
        return [2, 2]
    
    def __call__(self, a: float, b: float):
        return a / b

class Exponent(Function):
    @property
    def description(self):
        return "Use this function to compute the exponent of two constants."
    
    @property
    def example_args(self):
        return [2, 2]
    
    def __call__(self, a: float, b: float):
        return a ** b

generator = MistralTextGenerator(
    model = os.environ.get("MODEL_NAME", "mistral-small"),
    api_key=os.environ["MISTRAL_API_KEY"],
    temperature=0.7
)

llm = LLM(generator=generator)

engine = Engine()
# Base functions
engine.register(Reasoning())
engine.register(Stop())

# Calculator functions
engine.register(Sum())
engine.register(Subtraction())
engine.register(Product())
engine.register(Division())
engine.register(Exponent())

agent = Agent(llm=llm, engine=engine)
agent.prompt = f"""Act as a calculator. You can use the following functions:

{engine.help}

Only one function call can be included per step.
Not Allowed:
>> Sum(Product(2, 3), 4)

You are computing one step at a time. Respond with a valid Python function call.

{EXAMPLE}

{QUERY}
"""
# How much is (2*4 + 3)*5?

agent.bootstrap = [
    'Reasoning("I need to reason step-by-step, remembering to only output a single function call at a time")',
]
agent.run()
