from microchain.engine.function import Function, FunctionResult
from termcolor import colored
AGENT_MAX_TRIES = 3
ITERATION_COUNT = 10
class Agent:
    def __init__(self, llm, engine):
        self.llm = llm
        self.engine = engine
        self.max_tries = AGENT_MAX_TRIES
        self.prompt = None
        self.bootstrap = []
        self.do_stop = False

        self.engine.bind(self)
        self.reset()

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
        if reply.startswith('Reasoning("'):
            # reply = reply.split('\n')[0]
            reply = reply[:reply.find('")')+2]
        elif False: # These will be custom clauses
            pass
        else:
            # reply = reply[:reply.rfind(")")+1]
            # Clean reasoning output
            suffix = reply[-4:]
            if suffix == '")")':
                reply = reply[:-2]
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
            
            reply = self.llm(self.history + temp_messages, stop=["\n"])
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

    def run(self, iterations=ITERATION_COUNT):
        if self.prompt is None:
            raise ValueError("You must set a prompt before running the agent")

        print(colored(f"prompt:\n{self.prompt}", "blue"))
        print(colored(f"Running {iterations} iterations", "green"))

        self.reset()
        self.build_initial_messages()

        step_count = 0
        for it in range(iterations):
            if self.do_stop:
                break

            step_output = self.step()
            
            if step_output["abort"]:
                break

            self.history.append(dict(
                role="assistant",
                content=step_output["reply"]
            ))
            self.history.append(dict(
                role="user",
                content=step_output["output"]
            ))
            step_count += 1
            
        print(colored(f"Completed in {step_count} steps", "green"))
