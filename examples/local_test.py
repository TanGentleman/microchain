# Example: reuse your existing OpenAI setup
import time
from openai import OpenAI
STREAM_RESPONSE = True
EXAMPLE_ITEM = 'chess piece'
# SINGULAR_ITEM = EXAMPLE_ITEM
SINGULAR_ITEM = 'coin'
INSTRUCTION = f"List 5 {SINGULAR_ITEM}s, with a comma between each {SINGULAR_ITEM} and no newlines. E.g., 1, 2, 3, ..."
# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
start_time = time.time()
end_time = None
completion = client.chat.completions.create(
model="local-model", # this field is currently unused
messages=[
    {"role": "system", "content": "Perform the task."},
    {"role": "user", "content": INSTRUCTION}
    ],
temperature=0.8,
stream=STREAM_RESPONSE,
)
# create variables to collect the stream of chunks
collected_chunks = []
collected_messages = []
if STREAM_RESPONSE:
    for chunk in completion:
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk.choices[0].delta.content  # extract the message
        if chunk_message is None:
            break
        assert isinstance(chunk_message, str), "chunk_message must be a string"
        print(chunk_message, end='')  # print the message
        collected_messages.append(chunk_message)  # save the message
else:
    message = completion.choices[0].message.content
    if message is None:
        print('local generator returned None. Replacing with empty string.')
        message = ''
    print(message)

end_time = time.time() - start_time

print('\n**********')
print(f"Total time taken: {end_time:.2f} seconds")
