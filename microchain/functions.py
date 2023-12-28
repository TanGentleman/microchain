from microchain import Function

class Reasoning(Function):
    @property
    def description(self):
        return "Use this function for your internal reasoning. It should be immediately followed by a function call."

    @property
    def example_args(self):
        return ["'Next, take the sum of the two integers'"]

    def __call__(self, reasoning: str):
        return f"Proceed to the next step."

class Stop(Function):
    @property
    def description(self):
        return "Use this function to stop the program when the goal has been reached."

    @property
    def example_args(self):
        return []

    def __call__(self):
        self.engine.stop()
        return "The program has been stopped"
