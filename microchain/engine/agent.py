from termcolor import colored
from os.path import exists
from json import dump as json_dump

from microchain.engine.function import Function, FunctionResult
from microchain.models.llm import LLM
from microchain.engine.engine import Engine
AGENT_MAX_TRIES = 3
MAX_STEPS = 10
MAX_SESSION_TOKENS = 30000
class Agent:
    def __init__(self, llm: LLM, engine: Engine, max_tries=AGENT_MAX_TRIES, max_steps=MAX_STEPS, session_tokens=MAX_SESSION_TOKENS):
        self.llm = llm
        self.engine = engine

        self.max_tries = max_tries
        self.max_steps = max_steps

        self.system_message = None
        self.example_prompt = None
        self.prompt = None
        self.bootstrap = []
        self.do_stop = False
        self.engine.bind(self)
        self.reset()

        self.total_tokens = 0
        self.success_step_count = None
        self.finish_reason = None

        self.max_session_tokens = session_tokens

        self.last_output = None

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
        for command in self.bootstrap:
            result, output = self.engine.execute(command)
            if result == FunctionResult.ERROR:
                raise Exception(f"Your bootstrap commands contain an error. output={output}")

            print(colored(f">> {command}", "blue"))
            print(colored(f"{output}", "green"))

            self.history.append(dict(
                role="assistant",
                content=command
            ))
            self.history.append(dict(
                role="user",
                content=output
            ))
        self.history.append(dict(
            role="user",
            content=self.prompt
        ))
            
    def clean_reply(self, reply:str):
        # in the future this will be passed function names as a list of strings.
        # reply = reply.replace("\_", "_")
        # reply = reply.strip()
        reply = reply.split('\n')[0]
        if reply.startswith('Reasoning("'):
            # reply = reply.split('\n')[0]
            end_index = reply.find('")') + 2
            reply = reply[:end_index]
        elif False: # These will be custom clauses
            pass
        else:
            end_index = reply.find(") #") + 1
            if end_index == 0:
                end_index = reply.rfind(")") + 1
            reply = reply[:end_index]
            # reply = reply[:reply.rfind(")")+1]
            # Clean reasoning output
            # suffix = reply[-4:]
            # if suffix == '")")':
            #     reply = reply[:-2]
        return reply

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
            
            reply, tokens = self.llm(self.history + temp_messages, stop=["\n"])
            self.total_tokens += tokens
            reply = self.clean_reply(reply)

            if len(reply) < 2:
                print(colored("Empty reply: aborting task", "red"))
                abort = True
                break

            print(colored(f">> {reply}", "yellow"))
            
            result, output = self.engine.execute(reply)

            if result == FunctionResult.ERROR:
                print(colored(output, "red"))
                temp_messages.append(dict(
                    role="assistant",
                    content=reply
                    # content='#Error context#\n' + reply + '\n' + 'Reminder: Output the next step as only an independent function call.'
                ))
                temp_messages.append(dict(
                    role="user",
                    content=output
                ))
            else:
                print(colored(output, "green"))
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
        print(colored(f"{self.finish_reason} in {self.success_step_count} steps", "green"))
        self.end_run()

    def save_file(self, data):
        # save history to file
        history_file = 'history.json'
        if exists(history_file):
            index = 1
            while exists(f'history-{index}.json'):
                index += 1
            history_file = f'history-{index}.json'


        with open(history_file, 'w') as f:
            json_dump(data, f, indent=4)
        
    def end_run(self):
        print(colored(f"Total tokens consumed: {self.total_tokens}", "green"))
        # Costs by default in euro
        multiplier_euro_to_dollar = 1.1
        multiplier_charge_for_output = 1.12
        mistral_input_cost_per_mill = {
        'mistral-tiny' : 0.14,
        'mistral-small' : 0.6,
        'mistral-medium' : 2.5
        }
        round_digits = 4
        model = self.llm.generator.model
        input_cost = None
        if model in mistral_input_cost_per_mill:
            input_cost = mistral_input_cost_per_mill[model]
        if input_cost:
            uninflated_price = self.total_tokens / 1000000 * input_cost
            session_cost = round(uninflated_price * multiplier_euro_to_dollar * multiplier_charge_for_output, round_digits)
            print(colored(f"Session cost: ${session_cost}", "green"))
        
        config_entry = {
        "configuration": {
            "max_tries": self.max_tries,
            "max_steps": self.max_steps,
            "session_tokens": self.total_tokens,
        },
        "prompt": self.prompt,
        "model": self.llm.generator.model,
        "finish": f"{self.finish_reason}" + (f" in {self.success_step_count} steps" if self.success_step_count else ""),
        "final_answer": self.last_output or "N/A"
        }
        data = [config_entry] + self.history
        self.save_file(data)

