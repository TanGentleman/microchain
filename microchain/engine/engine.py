import ast
from microchain.engine.function import Function, FunctionResult

class Engine:
    def __init__(self, state: dict = dict()):
        self.state = state
        self.functions: dict[str, Function] = dict()
        self.help_called = False
        self.agent = None
    
    def register(self, function: Function):
        self.functions[function.name] = function
        function.bind(state=self.state, engine=self)

    def bind(self, agent):
        self.agent = agent

    def stop(self):
        if self.agent is None:
            raise ValueError("You must bind the engine to an agent before stopping")
        self.agent.stop()

    def execute(self, command: str):
        if self.agent is None:
            raise ValueError("You must bind the engine to an agent before executing commands")
        if not self.help_called:
            raise ValueError("You never accessed the help property. Building a prompt without including the help string is a very bad idea.")
        try:
            tree = ast.parse(command)
        except SyntaxError:
            return FunctionResult.ERROR, f"Error: syntax error in command {command}. Please try again."
        
        if len(tree.body) != 1:
            return FunctionResult.ERROR, f"Error: unknown command {command}. Please try again."

        if not isinstance(tree.body[0], ast.Expr):
            return FunctionResult.ERROR, f"Error: unknown command {command}. Please try again."

        if not isinstance(tree.body[0].value, ast.Call):
            return FunctionResult.ERROR, f"Error: the command {command} must be a function call. Please try again."
        
        if not isinstance(tree.body[0].value.func, ast.Name):
            return FunctionResult.ERROR, f"Error: the command {command} must be a function call. Please try again."

        function_name = tree.body[0].value.func.id
        function_args = tree.body[0].value.args
        function_kwargs = tree.body[0].value.keywords

        # Check that all arguments are constants
        ERROR_MESSAGE = f"Error: Failed function call. Instead, follow the function schema one step at a time."
        for arg in function_args:
            if not isinstance(arg, ast.Constant):
                if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub) and isinstance(arg.operand, ast.Constant):
                    continue
                return FunctionResult.ERROR, ERROR_MESSAGE + ' Hint: function arg must be a constant.'

        for kwarg in function_kwargs:
            if not isinstance(kwarg, ast.keyword):
                return FunctionResult.ERROR, ERROR_MESSAGE + ' Hint: function kwarg must be a keyword.'
            if not isinstance(kwarg.value, ast.Constant):
                if isinstance(kwarg.value, ast.UnaryOp) and isinstance(kwarg.value.op, ast.USub) and isinstance(kwarg.value.operand, ast.Constant):
                    continue
                return FunctionResult.ERROR, ERROR_MESSAGE + ' Hint: function kwarg must be a constant.'

        # For function_args
        function_args_values = []
        for arg in function_args:
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                function_args_values.append(-arg.operand.value)
            else:
                function_args_values.append(arg.value)
        function_args = function_args_values

        # For function_kwargs
        function_kwargs_dict = {}
        for kwarg in function_kwargs:
            if isinstance(kwarg.value, ast.UnaryOp) and isinstance(kwarg.value.op, ast.USub):
                function_kwargs_dict[kwarg.arg] = -kwarg.value.operand.value
            else:
                function_kwargs_dict[kwarg.arg] = kwarg.value.value
        function_kwargs = function_kwargs_dict

        if function_name not in self.functions:
            return FunctionResult.ERROR, f"Error: unknown command {command}. Please try again."
        
        if len(function_args_values) + len(function_kwargs_dict) != len(self.functions[function_name].call_parameters):
            return FunctionResult.ERROR, self.functions[function_name].error
        
        valid_function = self.functions[function_name]
        return valid_function.safe_call(args=function_args_values, kwargs=function_kwargs_dict)   
    
    @property
    def help(self):
        self.help_called = True
        return "\n".join([f.help for f in self.functions.values()])
