import google.generativeai as genai
from google.generativeai import types
from vietnamese_legal_rag import VietnameseLegalRAG, LegalChunk
import json

rag = VietnameseLegalRAG()
with open("legal_chunks.json",'r',encoding='utf-8') as f:
    legal_chunks_data = json.load(f)
    # Convert dict objects to LegalChunk objects
    rag.chunks = [LegalChunk(**chunk_data) for chunk_data in legal_chunks_data]

rag.load_index("faiss_index.index")

system_prompt = """
Bạn là luật sư – chuyên gia biên tập văn bản quy phạm pháp luật Việt Nam.
Đầu vào gồm:
1. Câu hỏi của người dùng.
2. Một đoạn JSON (gọi là "RAG") chứa: law_id, article_id, title, text (toàn bộ nội dung điều luật).

NHIỆM VỤ:
- Phân tích câu hỏi để xác định: (a) văn bản luật, (b) điều luật, (c) khoản/điểm cụ thể (nếu có).
- Trả lời NGẮN GỌN, CHÍNH XÁC, dễ hiểu cho người không chuyên.
- TUYỆT ĐỐI KHÔNG thêm thông tin ngoài văn bản gốc.

LUẬT TRÍCH DẪN:
- nđ-cp  → Nghị định
- tt     → Thông tư
- qd     → Quyết định
- lệnh   → Lệnh
- nghị quyết → Nghị quyết
- khác   → giữ nguyên

QUY TRÌNH RA QUYẾT ĐỊNH:
1. Nếu câu hỏi chỉ rõ hành vi / khoản / điểm:
   – Trích NGUYÊN VĂN khoản/điểm tương ứng.
   – Giải thích bằng lời đơn giản.
   – Ghi rõ mức phạt, biện pháp khắc phục, thời hiệu xử phạt (nếu có).
2. Nếu câu hỏi tổng quát (không chỉ cụ thể):
   – Liệt kê NGẮN GỌN tất cả khoản (hoặc điểm) trong điều luật, kèm mức phạt chính.
   
ĐỊNH DẠNG ĐẦU RA (JSON):
{
  "title": "<Tóm tắt 1 dòng nội dung trả lời>",
  "citation": "<Trích dẫn chính xác: “Điều {article_id} {law_id_decoded} – khoản n điểm m”>",
  "answer": "<Nội dung trả lời (markdown hỗ trợ bullet, in đậm, in nghiêng)>",
}

VÍ DỤ (few-shot):
Câu hỏi: "Phạt bao nhiêu nếu điều khiển xe máy không đội mũ bảo hiểm?"
RAG: {"law_id":"100/2019/nđ-cp","article_id":"6","title":"Xử phạt vi phạm quy định về đội mũ bảo hiểm","text":"... khoản 2 điểm a phạt tiền từ 200.000 đồng đến 300.000 đồng ..."}
=> Trả lời:
{
  "title": "Phạt 200-300 nghìn đồng khi không đội mũ bảo hiểm",
  "citation": "Điều 6 Nghị định 100/2019/nđ-cp – khoản 2 điểm a",
  "answer": "Phạt tiền **200.000 – 300.000 đồng** đối với người điều khiển, người ngồi trên xe mô tô, xe gắn máy không đội mũ bảo hiểm.",
}
"""

user_query = "Phạt bao nhiêu nếu điều khiển xe máy không đội mũ bảo hiểm?"
search_results = rag.search(user_query, top_k=1)
best_result = search_results[0]
print(best_result)


genai.configure(api_key="AIzaSyBoyC2U083flrtb4ihTTRN4aeERr1A3vAM")

model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content(
    [
        system_prompt,
        f"Câu hỏi người dùng: {user_query}\n\nDữ liệu RAG:\n{best_result}"
    ],
    generation_config=genai.types.GenerationConfig(
        temperature=0,
        response_mime_type="application/json"
    )
)

print(response.text)
