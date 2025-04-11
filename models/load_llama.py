import os
import torch
from transformers import pipeline

# Thiết lập thư mục cache mặc định
# os.environ['TRANSFORMERS_CACHE'] = '/root/thu/KG_RAG/models/huggingface_cache'

model_id = "meta-llama/Llama-3.2-1B"

pipe = pipeline(
    "text-generation", 
    model=model_id,
    device_map="cpu",       # Chạy trên CPU
    torch_dtype=torch.float32  # Loại dtype phù hợp với CPU
)

# Ví dụ sử dụng
output = pipe("Once upon a time,", max_new_tokens=50)
print(output[0]["generated_text"])


#---------------GPU-----------------------
# import os
# import torch
# from transformers import pipeline
# #from transformers import AutoTokenizer, AutoModelForCausalLM

# # Thiết lập thư mục cache mặc định
# os.environ['TRANSFORMERS_CACHE'] = '/root/thu/KG_RAG/models/huggingface_cache'

# model_id = "meta-llama/Llama-3.2-3B"

# pipe = pipeline(
#     "text-generation", 
#     model=model_id, 
#     torch_dtype=torch.bfloat16, 
#     device_map="auto"
# )

# print(pipe("The key to life is", max_new_tokens=50)[0]['generated_text'])


# Đăng nhập với token Hugging Face (nếu cần quyền truy cập)
# hf_token = "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"  # Thay "your_huggingface_token_here" bằng token thực của bạn
# login(token=hf_token)