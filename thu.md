fix .yaml
    python3 -m kg_rag.run_setup

hugging_face_API = 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

gpt:
python3 -m kg_rag.rag_based_generation.GPT.text_generation -g "gpt-35-turbo"

python3 -m kg_rag.rag_based_generation.GPT.text_generation -g "gpt4"


llama:
op1:
python3 -m kg_rag.rag_based_generation.Llama.text_generation -m "method-1"

op2:
python3 -m kg_rag.rag_based_generation.Llama.text_generation -m "method-2"

What is the genetic cause of Hutchinson-Gilford Progeria Syndrome?

What drugs are currently approved for treating Duchenne muscular dystrophy?

Are there any drugs used for weight management in patients with Bardet-Biedl Syndrome?