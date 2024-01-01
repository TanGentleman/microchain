import os

from microchain import MistralChatGenerator, LLM, Function, Engine, Agent
from microchain.functions import Reasoning, Stop
from dotenv import load_dotenv

from microchain import OpenAIChatGenerator
MODES = ["mistral", "openai", "local"]
MODE = MODES[2]

if MODE != "local":
    load_dotenv()
    if MODE == "mistral":
        MISTRAL_KEY = os.environ["MISTRAL_API_KEY"]
        MISTRAL_MODEL = os.environ.get("MODEL_NAME", "mistral-small")
    elif MODE == "openai":
        OPENAI_MODEL = "gpt-3.5-turbo-1106"
        OPENAI_KEY = os.environ["OPENAI_API_KEY"]
    else:
        raise ValueError(f"Unknown mode {MODE}. Please use mistral, openai, or local.")

LOCAL_BASE_URL = "http://localhost:1234/v1"
MAX_TOKENS = 150
TEMPERATURE = 0

MAX_TRIES = 3
MAX_STEPS = 20
MAX_SESSION_TOKENS = 30000

# expression = "2**2 + 3**3 + 4**4"
# expression = "3**3-4"
# expression = "1+12+(7+3)*(5+7)"
# expression = "2**2 + (4-3)**3"
# expression = "4+9+(8+1)*(2+8)"
# expression = "2+8+(6+2)*(4+6)"
expression = "10**2*(7-2*1)+1"
# expression = "(3 + 2) * 4 - 1"
# expression = "(5*4)**3"

PREFIX = "Evaluate"
EXPRESSION = expression

example_expression = "(3*9-2*1)**2"
EXAMPLE_STEPS = {
    0: 'Reasoning("Now planning steps for ((3*9-2*1)**2).")',
    1: 'Reasoning("Plan: Step 1| exp1 = 3*9. Step 2| exp2 = 2*1. Step 3| exp3 = exp1 - exp2. Step 4| exp3**2. Then Stop.")',
    2: 'Reasoning("exp1 = 3*9")',
    3: 'Multiply(3, 9)',
    4: 'Reasoning("exp2 = 2*1")',
    5: 'Multiply(2, 1)',
    6: 'Reasoning("exp3 = exp1 - exp2")',
    7: 'Subtract(27, 2)',
    8: 'Reasoning("exp3**2")',
    9: 'Power(25, 2)',
    10: 'Reasoning("Final answer: 625")'
}

PROMPT = f"{PREFIX} `({EXPRESSION})`"
EXAMPLE_PROMPT = f"{PREFIX} `({example_expression})`"

def get_generator():
    if MODE == "mistral":
        return MistralChatGenerator(
            model = MISTRAL_MODEL,
            api_key=MISTRAL_KEY,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
    elif MODE == "openai":
        return OpenAIChatGenerator(
            model = OPENAI_MODEL,
            api_key=OPENAI_KEY,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
    elif MODE == "local":
        return OpenAIChatGenerator(
            model = "local-model",
            api_key="not-needed",
            api_base=LOCAL_BASE_URL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
    raise ValueError(f"Unknown mode {MODE}. Please use mistral, openai, or local.")


class Add(Function):
    @property
    def description(self):
        return "Use Add(a: int, b: int) to compute the sum of two constants."
    
    @property
    def example_args(self):
        return [2, 2]
    
    def __call__(self, a: int, b: int):
        return a + b

class Subtract(Function):
    @property
    def description(self):
        return "Use Subtract(a: int, b: int) for (a-b)."
    
    @property
    def example_args(self):
        return [2, 2]
    
    def __call__(self, a: int, b: int):
        return a - b

class Multiply(Function):
    @property
    def description(self):
        return "Use Multiply(a: int, b: int) for (a*b)."
    
    @property
    def example_args(self):
        return [2, 2]
    
    def __call__(self, a: int, b: int):
        return a * b

class Power(Function):
    @property
    def description(self):
        return "Use Power(a: int, b: int) for (a**b)."
    
    @property
    def example_args(self):
        return [2, 2]
    
    def __call__(self, a: float, b: float):
        return a ** b

# class PlanSteps(Function):
#     @property
#     def description(self):
#         return "Use this function to plan a sequence of function calls as a list. It should be immediately followed by a function call."
    
#     @property
#     def example_args(self):
#         # (3*9 - 2*1)**2
#         return [["exp1 = 3*9", "exp2 = 2*1", "exp1 - exp2", "exp3**2"]]
#         # return [["exp1 = Multiply(3, 9)", "exp2 = Multiply(2, 1)", "exp3 = Subtract(exp1, exp2)", "Power(exp3, 2)"]]
    
#     def __call__(self, steps: list):
#         return f"Proceed to the next step."

def initialize_agent() -> Agent:
    generator = get_generator()

    llm = LLM(generator=generator)

    engine = Engine()
    # Base functions
    engine.register(Reasoning())
    engine.register(Stop())

    # Calculator functions
    engine.register(Add())
    engine.register(Subtract())
    engine.register(Multiply())
    engine.register(Power())

    agent = Agent(llm=llm, engine=engine, max_tries=MAX_TRIES, max_steps=MAX_STEPS, session_tokens=MAX_SESSION_TOKENS)
    agent.system_message = f"""Act as a calculator using a precise function schema.
    #Allowed Functions:
    {engine.help}
    
    #Rules
    1. Follow the order of operations, where "**" is notation for Power.
    2. Avoid nested functions. 
    3. Only use constants in function calls.
    4. Only respond with the next independent function call.
    5. When the goal is reached, call Stop().
    #
    
    Valid functions: {", ".join(engine.functions.keys())}
    """
    agent.example_prompt = EXAMPLE_PROMPT
    agent.bootstrap = list(EXAMPLE_STEPS.values())
    agent.prompt = PROMPT
    return agent

def main():
    agent = initialize_agent()
    agent.run()

if __name__ == "__main__":
    main()