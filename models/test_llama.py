from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

LLM_MODEL = "llama3.2:3b-instruct-fp16"  # Mô hình LLM


# Khởi tạo LLM và Embeddings
llm = Ollama(model=LLM_MODEL, base_url="http://localhost:11434", request_timeout=3000)

# Tên model
model_name = "ministral/Ministral-3b-instruct"

# Load tokenizer và model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Prompt test
input_text = "### Câu hỏi: Hãy giới thiệu về bản thân bạn.\n### Trả lời:"

# Token hóa input
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# Cấu hình sinh văn bản
generation_config = GenerationConfig(
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    max_new_tokens=256,
    eos_token_id=tokenizer.eos_token_id
)

# Sinh văn bản
outputs = model.generate(**inputs, generation_config=generation_config)

# In kết quả
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
