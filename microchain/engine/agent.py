from termcolor import colored
from os.path import exists
from json import dump as json_dump

from microchain.engine.function import Function, FunctionResult
from microchain.models.llm import LLM
from microchain.engine.engine import Engine
AGENT_MAX_TRIES = 3
MAX_STEPS = 10
MAX_SESSION_TOKENS = 30000
from time import time
class Agent:
    def __init__(self, llm: LLM, engine: Engine, max_tries=AGENT_MAX_TRIES, max_steps=MAX_STEPS, session_tokens=MAX_SESSION_TOKENS, success_fn = lambda _: True):
        self.llm = llm
        self.engine = engine

        self.max_tries = max_tries
        self.max_steps = max_steps

        self.system_message: str | None = None
        self.example_prompt: str | None = None
        self.prompt: str | None = None
        self.bootstrap: list[str] = []
        self.do_stop = False
        self.engine.bind(self)
        self.reset()

        self.total_tokens = 0
        self.success_step_count = None
        self.finish_reason = None

        self.max_session_tokens = session_tokens

        self.last_output = None
        self.is_valid_goal_value = success_fn

    def reset(self):
        self.history = []
        self.do_stop = False

    def build_initial_messages(self):
        self.history = [ # This should be role:"system, <content>:"System instructions"
            dict(
                role="system",
                content=self.system_message
            )
        ]
        # if self.example_prompt:
        #     self.history.append(dict(
        #         role="user",
        #         content=self.example_prompt
        #     ))
        if self.bootstrap:
            self.apply_commands(self.bootstrap, no_stop=True)
        self.history.append(dict(
            role="user",
            content=self.prompt
        ))
    
    def apply_commands(self, commands: list[str], no_stop = False):
        for command in commands:
            result = None
            output = None
            if no_stop and command == "Stop()":
                result = FunctionResult.SUCCESS
                output = ""
            else:
                result, output = self.engine.execute(command)
            if result == FunctionResult.ERROR:
                raise Exception(f"Your bootstrap commands contain an error. output={output}")

            print(colored(f">> {command}", "blue"))
            print(colored(f"{output}", "green"))

            self.history.append(dict(
                role="assistant",
                content=command
            ))
            if output:
                self.history.append(dict(
                    role="user",
                    content=output
                ))            
    def clean_reply(self, reply:str):
        # in the future this will be passed function names as a list of strings.
        for function_name in self.engine.functions:
            if reply.startswith(function_name):
                # logic here for reasoning and others
                if function_name == "PlanSteps":
                    if self.is_valid_goal_value(reply):
                        print(f'Got plan! {reply}')
                        self.last_output = reply
                        return 'Stop()'
                    pass
                    # PlanSteps(["a", "b"]) should become 
                elif function_name != "Reasoning":
                    # truncate the string at the first ")"
                    return reply.split(")", 1)[0] + ")"
                    
                return reply
            continue
        return 'Reasoning("After following the plan step-by-step, I will call Stop() at the goal.")'

    def stop(self):
        self.do_stop = True

    def step(self):
        result = FunctionResult.ERROR
        temp_messages = []
        tries = 0
        abort = False
        output = ""
        reply = ""
        while result != FunctionResult.SUCCESS:
            tries += 1

            if self.do_stop:
                abort = True
                break

            if tries > self.max_tries:
                print(colored(f"Tried {self.max_tries} times (agent.max_tries) Aborting", "red"))
                abort = True
                break

            if self.total_tokens > self.max_session_tokens:
                print(colored(f"Exceeded {self.max_session_tokens} tokens. Aborting", "red"))
                abort = True
                break
            
            reply, tokens = self.llm(self.history + temp_messages)
            self.total_tokens += tokens
            reply = self.clean_reply(reply)

            if len(reply) < 1:
                print(colored("Empty reply: aborting task", "red"))
                abort = True
                break

            print(colored(f">> {reply}", "yellow"))
            
            result, output = self.engine.execute(reply)
            if type(output) != str:
                raise ValueError('ERROR: The output from engine.execute must be a string!')

            if result == FunctionResult.ERROR:
                print(colored(output, "red"))
                temp_messages.append(dict(
                    role="assistant",
                    content=reply
                ))
                temp_messages.append(dict(
                    role="user",
                    content=output
                ))
            else:
                print(colored(output, "green"))
                if self.is_valid_goal_value(output):
                    self.last_output = output
                break
        
        return dict(
            abort=abort,
            reply=reply,
            output=output,
        )

    def run(self):
        max_steps = self.max_steps
        if self.prompt is None:
            raise ValueError("You must set a prompt before running the agent")

        if self.example_prompt:
            print(colored(f"Example prompt:\n{self.example_prompt}", "blue"))

        self.reset()
        start_time = time()
        
        self.build_initial_messages()
        print(colored(f"Prompt:\n{self.prompt}", "blue"))
        print(colored(f"Running {max_steps} steps", "green"))

        step_count = 0
        # finish_reasons = ['Exhausted', 'Aborted', 'Completed']
        for i in range(max_steps):
            if self.do_stop:
                break

            step_output = self.step()
            
            if step_output["abort"]:
                self.finish_reason = "Aborted"
                break
            
            # we need to clear cache of old, unhelpful replies.
            self.history.append(dict(
                role="assistant",
                content=step_output["reply"]
            ))
            self.history.append(dict(
                role="user",
                content=step_output["output"]
            ))
            step_count += 1
            if step_count == max_steps:
                self.finish_reason = "Exhausted"
        
        if self.finish_reason is None:
            self.finish_reason = "Completed"
            self.success_step_count = step_count
        
        end_time = round(time() - start_time,2)
        print(colored(f"{self.finish_reason} in {self.success_step_count} steps {end_time}s", "green"))
        self.end_run()
        return self.last_output

    def save_file(self, data):
        # save history to file
        history_file = 'history.json'
        if exists(history_file):
            index = 1
            while exists(f'history-{index}.json'):
                index += 1
            history_file = f'history-{index}.json'

        filepath = 'logs/' + history_file
        with open(filepath, 'w') as f:
            json_dump(data, f, indent=4)
        
    def end_run(self):
        print(colored(f"Total tokens consumed: {self.total_tokens}", "green"))
        model_name = self.llm.generator.model
        session_cost = get_price(model_name, self.total_tokens)
        if session_cost:
            print(colored(f"Session cost: ${session_cost}", "green"))
        finish_message = self.finish_reason
        if self.success_step_count is not None:
            finish_message += f" in {self.success_step_count} steps"
        config_entry = {
        "configuration": {
            "max_tries": self.max_tries,
            "max_steps": self.max_steps,
            "session_tokens": self.total_tokens,
        },
        "prompt": self.prompt,
        "model": model_name,
        "finish": finish_message,
        "final_answer": self.last_output or "N/A"
        }
        data = [config_entry] + self.history
        self.save_file(data)

def get_price(model: str, tokens: int) -> float | None:
    round_digits = 4
    multiplier_input_ratio = 0.9 # Assume ~10% of the usage is output tokens
    cost = None
    # Mistral logic
    multiplier_euro_to_dollar = 1.1
    mistral_pricing_per_million = {
    'mistral-tiny' : (0.14, 0.42),
    'mistral-small' : (0.6, 1.8),
    'mistral-medium' : (2.5, 7.5)
    }
    # OpenAI logic
    openai_pricing_per_thousand = {
    'gpt-4-1106-preview' : (0.01, 0.03),
    'gpt-4-32k' : (0.01, 0.03),
    'gpt-4' : (0.03, 0.06),
    'gpt-3.5-turbo-1106' : (0.001, 0.002),
    'gpt-3.5-turbo' : (0.003, 0.006)
    }
    unadjusted_price = None
    if model in mistral_pricing_per_million:
        unadjusted_price = tokens / 1000000 * mistral_pricing_per_million[model][0] * multiplier_euro_to_dollar
    elif model in openai_pricing_per_thousand:
        unadjusted_price = tokens / 1000 * openai_pricing_per_thousand[model][0]
    else:
        print(f"Model {model} not found in pricing tables")
        return None
    cost = round(unadjusted_price * (1/multiplier_input_ratio), round_digits)
    return cost