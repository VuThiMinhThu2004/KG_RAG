import requests
import logging
from dotenv import load_dotenv
import os 

API_URL = "https://cf.mnapi.com/v1/chat/completions"  

load_dotenv("/root/thu/KG_RAG/models/config/gpt_config.env")
api = os.getenv("API_KEY")

# def request_api(messages, model="gpt-4o-mini", temperature=0):
#     logging.getLogger("urllib3").setLevel(logging.CRITICAL)
#     logging.getLogger("requests").setLevel(logging.CRITICAL)
#     logging.getLogger("http.client").setLevel(logging.CRITICAL)

#     headers = {
#         "Authorization": f"{api}",
#         "Content-Type": "application/json",
#         "Accept": "application/json",
#     }

#     data = {
#         "model": model,
#         "temperature": temperature,
#         "messages": messages
#     }

#     response = requests.post(API_URL, headers=headers, json=data)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         raise RuntimeError(f"API request failed: {response.status_code} - {response.text}")
  

def request_api(messages, model="gpt-4o-mini", temperature=0):
    # Tắt log warning từ requests
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)
    logging.getLogger("requests").setLevel(logging.CRITICAL)
    logging.getLogger("http.client").setLevel(logging.CRITICAL)

    # Header với token
    headers = {
        "Authorization": f"{api}",  # Đảm bảo api đã có prefix "Bearer "
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Dữ liệu gửi lên API
    data = {
        "model": model,
        "temperature": temperature,
        "messages": messages
    }

    # Gửi POST request
    response = requests.post(API_URL, headers=headers, json=data)

    # Trả về kết quả hoặc raise lỗi
    if response.status_code == 200:
        return response.json()  # Chuyển đổi phản hồi thành JSON
    else:
        raise RuntimeError(f"API request failed: {response.status_code} - {response.text}")
      
      
# def request_api(text):
#     logging.getLogger("urllib3").setLevel(logging.CRITICAL)
#     logging.getLogger("requests").setLevel(logging.CRITICAL)
#     logging.getLogger("http.client").setLevel(logging.CRITICAL)

#     headers = {
#         "Authorization": f"{api}",
#         "Content-Type": "application/json",
#         "Accept": "application/json",
#     }

#     data = {
#       "model": "gpt-4o-mini",
#       "messages": [
#         {
#           "role": "user",
#           "content": text
#         }
#       ],
#     }

#     response = requests.post(API_URL, headers=headers, json=data)
#     return response