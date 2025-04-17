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

What compounds are used to treat Alagille syndrome, and what other diseases share similar clinical features with it?

MedHopQA:
Q1:
Which benign, autosomal recessive hyperbilirubinemia syndrome is caused by a mutation in the UGT1A1 gene on chromosome 2?
- gene
Q2:


# DATASET
## benchmark_data
- mcq_question.csv: multiple choice questions: mối quan hệ giữa bệnh và gene, biến thể di truyền (variants)
    có các trường:
        disease_pair
        correct_node: đáp án đúng
        negative_samples: là các đáp án sai
        disease_1
        disease_2
        options_combined: danh sách tất cả đáp án (cả đúng + sai)
        text: Câu hỏi trắc nghiệm được tạo ra từ các trường dữ liệu kia.

    1 sample: "('psoriasis', ""Takayasu's arteritis"")",HLA-B,"SHTN1, DTNB, BTBD9, SLC14A2",psoriasis,Takayasu's arteritis,"SHTN1, HLA-B,  SLC14A2,  BTBD9,  DTNB","Out of the given list, which Gene is associated with psoriasis and Takayasu's arteritis. Given list is: SHTN1, HLA-B,  SLC14A2,  BTBD9,  DTNB"

- true_false_question.csv: có stt, text, label: ví dụ - 0,enhanced S-cone syndrome is not a vitreoretinal degeneration,False

## hyperparam_tuning_data
- single_disease_entity_prompts.csv: 
    disease_1
    compounds: ds các hợp chất/ thuốc
    diseases: ds các bệnh tương tự hoặc liên quan
    text: câu hỏi
- two_disease_entity_prompts.csv: 
    central_nodes: Danh sách các đặc điểm liên quan (như cơ quan, triệu chứng, hoặc gen) được chia sẻ giữa hai bệnh.
