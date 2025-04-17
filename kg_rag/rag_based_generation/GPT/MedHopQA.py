
from kg_rag.utility import *
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-g', type=str, default='gpt-4', help='GPT model selection')
parser.add_argument('-f', type=str, default='kg_rag/test/MedHopQA.json', help='Path to input JSON file')
parser.add_argument('-o', type=str, default='kg_rag/test/MedHopQA_Ouput.json', help='Path to output JSON file')
args = parser.parse_args()

CHAT_MODEL_ID = args.g
INPUT_FILE = args.f
OUTPUT_FILE = args.o
INTERACTIVE = False
EDGE_EVIDENCE = False

# Config and setup
SYSTEM_PROMPT = system_prompts["KG_RAG_BASED_TEXT_GENERATION"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]

CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID if openai.api_type == "azure" else None

# Load models and data
vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)

def main():
    # Load input questions
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for qid, entry in data.items():
        question = entry["Question"]
        print(f"Processing Q{qid}: {question}")

        context = retrieve_context(question, vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, EDGE_EVIDENCE)

        enriched_prompt = "Context: " + context + "\n" + "Question: " + question
        output = get_GPT_response(
            enriched_prompt,
            SYSTEM_PROMPT,
            CHAT_MODEL_ID,
            CHAT_DEPLOYMENT_ID,
            temperature=TEMPERATURE
        )

        # Save the response into the dict
        data[qid]["Answer"] = output.strip()
        print(f"Answer saved for Q{qid}")

    # Write updated JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nAll answers written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
