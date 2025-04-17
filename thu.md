fix .yaml
    python3 -m kg_rag.run_setup

hugging_face_API = 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

gpt:
python3 -m kg_rag.rag_based_generation.GPT.text_generation -g "gpt-35-turbo"

python3 -m kg_rag.rag_based_generation.GPT.text_generation -g "gpt4"

python3 -m kg_rag.rag_based_generation.GPT.MedHopQA -g "gpt4"

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
Trả lời đúng (9): 1,6,7,9,11,12,13,14,15
Trả lời sai (6): 2,3,4,5,8,10

Câu 2

Câu hỏi: Tình trạng hợp sọ sớm thấy trong hội chứng Shprintzen–Goldberg có do đột biến gen fibrillin-1 không?

Golden: No

Trả lời: Có liên quan đến đột biến gen FBN1

Đánh giá: ❌ Sai

Câu 3

Câu hỏi: Gen nào chứa miền helicase bị đột biến ở một số bệnh nhân mắc hội chứng Smith–Fineman–Myers?

Golden: ATRX

Trả lời: DDX3X

Đánh giá: ❌ Sai

Câu 4

Câu hỏi: Bệnh tự miễn nào gây viêm màng bồ đào và liên quan đến viêm phổi và viêm mạch?

Golden: Granulomatosis with polyangiitis

Trả lời: Vasculitis

Đánh giá: ❌ Sai

Câu 5

Câu hỏi: Gen nào trên nhiễm sắc thể 2 không được phát hiện bất thường ở bệnh nhân mắc hội chứng Holmes-Collins?

Golden: HOXD10

Trả lời: MYH8

Đánh giá: ❌ Sai

Câu 8

Câu hỏi: Tetramer lactate dehydrogenase nào phổ biến nhất trong cơ quan chính của hệ thần kinh?

Golden: LDH-1

Trả lời: LDH-5

Đánh giá: ❌ Sai

Câu 10

Câu hỏi: Nguyên nhân quan trọng thứ hai của sa sút trí tuệ sau bệnh Alzheimer là gì?

Golden: Microinfarcts

Trả lời: Vascular dementia

Đánh giá: ❌ Sai
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
