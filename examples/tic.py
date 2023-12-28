import os
import random
from dotenv import load_dotenv   # pip install python-dotenv
from tictactoe import Board      # pip install python-tictactoe

from microchain import MistralTextGenerator, LLM, Function, Engine, Agent
from microchain.functions import Reasoning, Stop

load_dotenv()
assert "MISTRAL_API_KEY" in os.environ, "Please set the MISTRAL_API_KEY environment variable"
MISTRAL_KEY = os.environ["MISTRAL_API_KEY"]
MISTRAL_MODEL = os.environ.get("MODEL_NAME", "mistral-small")


MAX_TOKENS = 200
TEMPERATURE = 0.3

MAX_TRIES = 3
MAX_STEPS = 20

def check_win(board):
    if board.has_won(1):
        return ". You won!"
    elif board.has_won(2):
        return ". You lost!"
    return ""

class State(Function):
    @property
    def description(self):
        return "Use this function to get the state of the board"

    @property
    def example_args(self):
        return []

    def __call__(self):
        return str(self.state["board"]) + check_win(self.state["board"])

class PlaceMark(Function):
    @property
    def description(self):
        return "Use this function to place a mark on the board. x represents the row and y the column. Starts at 0"

    @property
    def example_args(self):
        return [1, 1]

    def __call__(self, x: int, y: int):
        if (x, y) not in self.state["board"].possible_moves():
            return f"Error: the move {x} {y} is not valid"
        
        try:
            self.state["board"].push((x, y))
        except Exception as e:
            return f"Error: {e}"

        if len(self.state["board"].possible_moves()) > 0:
            move = random.choice(self.state["board"].possible_moves())
            self.state["board"].push(move)
            # display board
            print(self.state["board"])
            return f"Placed mark at {x} {y}. Your opponent placed a mark at {move[0]} {move[1]}." + check_win(self.state["board"])

        self.engine.stop()
        return f"Placed mark at {x} {y}." + check_win(self.state["board"]) + "The game is over"

generator = MistralTextGenerator(
    model = MISTRAL_MODEL,
    api_key=MISTRAL_KEY,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
)


llm = LLM(generator=generator)

engine = Engine(state=dict(board=Board()))
engine.register(State())
engine.register(PlaceMark())
engine.register(Reasoning())
engine.register(Stop())

agent = Agent(llm=llm, engine=engine, max_tries=MAX_TRIES, max_steps=MAX_STEPS)
agent.prompt = f"""Act as a tic tac toe playing AI. You can use the following functions:

{engine.help}

You are playing with X.
Take a deep breath and work on this problem step-by-step.
Call the State() function before each move, then place a mark on a valid square.


Valid functions: {", ".join(engine.functions.keys())}
"""
agent.bootstrap = [
    'Reasoning("I need to check the state of the board")',
    'State()',
]
agent.run()