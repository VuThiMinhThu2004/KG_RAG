# llm_setup.py
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import config

def setup_llm():
    """Initializes and configures the Ollama LLM."""
    llm = Ollama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        request_timeout=config.REQUEST_TIMEOUT,
        # Add other parameters like temperature if needed
    )
    Settings.llm = llm
    print(f"✅ LLM configured: {config.LLM_MODEL} at {config.OLLAMA_BASE_URL}")
    return llm

# Example usage (optional, for testing)
if __name__ == "__main__":
    llm_instance = setup_llm()
    # You could add a test chat here if desired
    # response = llm_instance.complete("Why is the sky blue?")
    # print(response)