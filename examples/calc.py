import os

from microchain import MistralTextGenerator, LLM, Function, Engine, Agent
from microchain.functions import Reasoning, Stop
from dotenv import load_dotenv
load_dotenv()
assert "MISTRAL_API_KEY" in os.environ, "Please set the MISTRAL_API_KEY environment variable"
MISTRAL_KEY = os.environ["MISTRAL_API_KEY"]
MISTRAL_MODEL = os.environ.get("MODEL_NAME", "mistral-small")
# MISTRAL_MODEL = 'mistral-medium'


MAX_TOKENS = 200
TEMPERATURE = 0.3

MAX_TRIES = 3
MAX_STEPS = 20

EXAMPLE = """Example: (76*7+3*(5+7))
>> Product(76, 7)
>> Sum(5, 7)
>> Product(3, 12)
>> Sum(532, 36)
> Stop()"""

# expression = "2² + 3³ + 4⁴"
# expression = "1+12+(7+3)*(5+7)"
# expression = "(3 + 2) × 4 - (6 - 1) + 2²"
# expression = "88 - 11 + 3 × 4 + (3^2)"
# expression = "2² + (4-3)³"
# expression = "4+9+(8+1)*(2+8)"
expression = "2+8+(6+2)*(4+6)"
# expression = "2^2*(7^3+4)-1"
# expression = "10^2*(7^2-1)-1"
# expression = "(3 + 2) * 4 - 1"

PREFIX = "Evaluate"
EXPRESSION = expression

QUERY = f"{PREFIX} `({EXPRESSION})`"
# QUERY = "What is 2² + (4-3)³"
# QUERY = "What is 2²*(7³+4)-1?"

class Sum(Function):
    @property
    def description(self):
        return "Use Sum(a: int, b: int) to compute the sum of two constants."
    
    @property
    def example_args(self):
        return [2, 2]
    
    def __call__(self, a: int, b: int):
        return a + b

class Subtraction(Function):
    @property
    def description(self):
        return "Use Subtraction(a: int, b: int) to compute the subtraction of two constants."
    
    @property
    def example_args(self):
        return [2, 2]
    
    def __call__(self, a: int, b: int):
        return a - b

class Product(Function):
    @property
    def description(self):
        return "Use Product(a: int, b: int) to compute the product of two constants."
    
    @property
    def example_args(self):
        return [2, 2]
    
    def __call__(self, a: int, b: int):
        return a * b

# class Validate(Function):
#     @property
#     def description(self):
#         return "Use this function to validate an expression against the schema."
    
#     @property
#     def example_args(self):
#         return ["Sum(2, 2)"]
    
#     def __call__(self, expression: str):
#         approved_functions = ["Sum", "Product", "Subtraction", "Exponent", "Reasoning", "Stop"]
#         prefix = expression[:expression.find("(")]
#         if prefix in approved_functions:
#             return True
#         return prefix in approved_functions

# class Division(Function):
#     @property
#     def description(self):
#         return "Use this function to compute the division of two constants."
    
#     @property
#     def example_args(self):
#         return [2, 2]
    
#     def __call__(self, a: float, b: float):
#         return a / b
class Exponent(Function):
    @property
    def description(self):
        return "Use this function to compute the exponent of two constants."
    
    @property
    def example_args(self):
        return [2, 2]
    
    def __call__(self, a: float, b: float):
        return a ** b

def initialize_agent() -> Agent:
    generator = MistralTextGenerator(
        model = MISTRAL_MODEL,
        api_key=MISTRAL_KEY,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
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
    # engine.register(Division())
    engine.register(Exponent())

    agent = Agent(llm=llm, engine=engine, max_tries=MAX_TRIES, max_steps=MAX_STEPS)
    agent.prompt = f"""Act as a calculator. Allowed function schema:
    #
    {engine.help}
    #
    Avoid nested functions. Output the next step as only an independent function call.
    #

    {QUERY}

    Valid functions: {", ".join(engine.functions.keys())}
    """
    # REASONING_START = "Rewrite the expression for clarity, then begin a sequence of valid calculator functions."
    REASONING_START = "My goal is to evaluate the expression correctly using the order of operations. I will then call a valid calculator function."
    agent.bootstrap = [
        f'Reasoning("{REASONING_START}")',
    ]
    return agent

def main():
    agent = initialize_agent()
    agent.run()

if __name__ == "__main__":
    main()