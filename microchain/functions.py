from microchain import Function

class Reasoning(Function):
    @property
    def description(self):
        return "Use this function for your internal reasoning. It should be immediately followed by a function call."

    @property
    def example_args(self):
        return ["'Next, take the sum of the two integers'"]

    def __call__(self, reasoning: str):
        return f"Proceed to the next step towards the goal."

class Stop(Function):
    @property
    def description(self):
        return "Use this function to stop the program when the goal has been reached."

    @property
    def example_args(self):
        return []

    def __call__(self):
        self.engine.stop()
        return ""

class PlanSteps(Function):
    @property
    def description(self):
        return "Use this function to plan the steps towards the goal. Create a sequence of valid function calls, like ['Add(1,2)', 'Stop()']."

    @property
    def example_args(self):
        return [["Add(3, 4)", "Power(7, 2)", 'Reasoning("Final answer: 625")', "Stop()]"]]

    def __call__(self, steps: str):
        print('Called PlanSteps')
        return steps
        # return f"Planned steps have a syntax issue. Try again, following strictly to the function schema."