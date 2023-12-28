from microchain.engine.function import Function, FunctionResult
from termcolor import colored
AGENT_MAX_TRIES = 3
MAX_STEPS = 10
MAX_SESSION_TOKENS = 30000
MISTRAL_METRICS = True
class Agent:
    def __init__(self, llm, engine, max_tries=AGENT_MAX_TRIES, max_steps=MAX_STEPS):
        self.llm = llm
        self.engine = engine

        self.max_tries = max_tries
        self.max_steps = max_steps

        self.prompt = None
        self.bootstrap = []
        self.do_stop = False
        self.engine.bind(self)
        self.reset()

        self.total_tokens = 0

    def reset(self):
        self.history = []
        self.do_stop = False

    def build_initial_messages(self):
        self.history = [
            dict(
                role="user",
                content=self.prompt
            ),
        ]
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

            if self.total_tokens > MAX_SESSION_TOKENS:
                print(colored(f"Exceeded {MAX_SESSION_TOKENS} tokens. Aborting", "red"))
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

        print(colored(f"prompt:\n{self.prompt}", "blue"))
        print(colored(f"Running {max_steps} steps", "green"))

        self.reset()
        self.build_initial_messages()

        step_count = 0
        # finish_reasons = ['Exhausted', 'Aborted', 'Completed']
        finish_reason = None
        for i in range(max_steps):
            if self.do_stop:
                break

            step_output = self.step()
            
            if step_output["abort"]:
                finish_reason = "Aborted"
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
                finish_reason = "Exhausted"
        
        if finish_reason is None:
            finish_reason = "Completed"
        print(colored(f"{finish_reason} in {step_count} steps", "green"))
        print(colored(f"Total tokens consumed: {self.total_tokens}", "green"))
        # Input: 0.6€ / 1M tokens | Output: 1.8€ / 1M tokens
        # At 80/20 split, average is $0.93 per 1M tokens
        # $0.93
        if MISTRAL_METRICS:
            session_cost = round(self.total_tokens / 1000000 * 0.93, 4)
            print(colored(f"Session cost: ${session_cost}", "green"))
        # save history to file
        with open('history.json', 'w') as f:
            from json import dump as json_dump
            json_dump(self.history, f, indent=4)
