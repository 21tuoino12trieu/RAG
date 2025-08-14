from vietnamese_legal_rag import VietnameseLegalRAG
import json

with open("train_question_answer.json","r", encoding = "utf-8") as f:
    question_answer = json.load(f)

rag = VietnameseLegalRAG()
rag.load_index("faiss_index.index")
    
result = {}
loop = [1, 3, 5, 10, 50, 100]
for k in loop:
    total_acc = 0.0
    total_mrr = 0.0
    n = 0
    for item in question_answer['items']:
        n += 1
        # Tạo set ground-truth (có thể nhiều article)
        gt = {f"{a['law_id']}/{a['article_id']}" for a in item['relevant_articles']}

        # Retrieval
        rag_json = rag.search(item['question'], k)
        rag_answer = [f"{r['law_id']}/{r['article_id']}" for r in rag_json]

        # ACC@k
        hit = any(doc in gt for doc in rag_answer[:k])
        total_acc += hit

        # MRR@k
        for rank, doc in enumerate(rag_answer[:k], start=1):
            if doc in gt:
                total_mrr += 1.0 / rank
                break
    result[f"ACC@{k}"] = total_acc / n
    result[f"MRR@{k}"] = total_mrr / n
    with open("retrieval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

# "ACC@1":float0.726846057571965
# "MRR@1":float0.726846057571965
# "ACC@3":float0.8979974968710889
# "MRR@3":float0.8197225698790154
# "ACC@5":float0.9305381727158949
# "MRR@5":float0.8273258239465997
# "ACC@10":float0.9565081351689612
# "MRR@10":float0.8308588861076339
# "ACC@50":float0.9877972465581978
# "MRR@50":float0.8326083862540914
# "ACC@100":float0.9924906132665833
# "MRR@100":float0.8326769614927131