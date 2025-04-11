from gpt_request import *

# response = request_api("hello")

# if response.status_code == 200:
#     result = response.json()
#     print(result["choices"][0]["message"]["content"])
# else:
#     print("Lỗi:", response.status_code, response.text)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What are the symptoms of Duchenne muscular dystrophy?"}
]

try:
    result = request_api(messages)
    if "choices" in result and len(result["choices"]) > 0:
        print(result["choices"][0]["message"]["content"])
    else:
        print("Unexpected response format:", result)
except RuntimeError as e:
    print(e)