import streamlit as st
import json
from datetime import datetime
import google.generativeai as genai
from google.generativeai import types
from vietnamese_legal_rag import VietnameseLegalRAG, LegalChunk
import time

# Configure page
st.set_page_config(
    page_title="Tư Vấn Pháp Luật AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Vietnamese legal theme
st.markdown("""
<style>
    /* Main theme colors */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: #e8f4f8;
        font-size: 1.2rem;
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-style: italic;
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* User message */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        margin-left: 20%;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* AI message */
    .ai-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        margin-right: 20%;
        box-shadow: 0 4px 12px rgba(245, 87, 108, 0.3);
    }
    
    /* Legal citation styling */
    .legal-citation {
        background: #f8f9fa;
        border-left: 4px solid #1e3c72;
        padding: 0.8rem 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        color: #2c3e50;
    }
    
    /* System status */
    .system-status {
        background: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 1rem 0;
        color: #2e7d32;
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-block;
        animation: loading 1.5s infinite;
    }
    
    @keyframes loading {
        0%, 20% { opacity: 0; }
        50% { opacity: 1; }
        80%, 100% { opacity: 0; }
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e1e8ed;
        padding: 0.8rem 1rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1e3c72;
        box-shadow: 0 0 0 3px rgba(30, 60, 114, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #1e3c72, #2a5298);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(30, 60, 114, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Statistics cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #1e3c72;
        margin: 0;
    }
    
    .stat-label {
        color: #6c757d;
        font-size: 0.9rem;
        margin: 0.5rem 0 0 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system with caching"""
    rag = VietnameseLegalRAG()
    with open("legal_chunks.json",'r',encoding='utf-8') as f:
        legal_chunks_data = json.load(f)
        # Convert dict objects to LegalChunk objects
        rag.chunks = [LegalChunk(**chunk_data) for chunk_data in legal_chunks_data]

    rag.load_index("faiss_index.index")
    return rag

@st.cache_resource
def initialize_gemini():
    """Initialize Gemini AI with caching"""
    genai.configure(api_key="AIzaSyBoyC2U083flrtb4ihTTRN4aeERr1A3vAM")
    return genai.GenerativeModel("gemini-2.5-flash")

def get_system_prompt():
    """Get the system prompt for legal consultation"""
    return """
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
                    "citation": "<Trích dẫn chính xác: "Điều {article_id} {law_id_decoded} – khoản n điểm m">",
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

def process_query(rag_system, model, query):
    """Process user query and return AI response"""
    try:
        # Search relevant legal documents
        search_results = rag_system.search(query, top_k=1)
        if not search_results:
            return None, "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu pháp luật."
        
        # Get the best match
        best_result = search_results[0]
        
        # Generate response using Gemini
        system_prompt = get_system_prompt()
        response = model.generate_content(
            [
                system_prompt,
                f"Câu hỏi người dùng: {query}\n\nDữ liệu RAG:\n{best_result}"
            ],
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                response_mime_type="application/json"
            )
        )
        
        # Parse JSON response
        response_data = json.loads(response.text)
        return best_result, response_data
        
    except Exception as e:
        st.error(f"Lỗi xử lý truy vấn: {str(e)}")
        return None, None

def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">⚖️ Tư Vấn Pháp Luật AI</h1>
        <p class="header-subtitle">Hệ thống tư vấn pháp luật Việt Nam thông minh</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_system" not in st.session_state:
        with st.spinner("🔄 Đang khởi tạo hệ thống..."):
            st.session_state.rag_system = initialize_rag_system()
            st.session_state.model = initialize_gemini()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Thống Kê Hệ Thống")
        
        # System statistics
        if hasattr(st.session_state, 'rag_system'):
            rag = st.session_state.rag_system
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(rag.chunks):,}</div>
                <div class="stat-label">Đoạn văn bản pháp luật</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{rag.index.ntotal:,}</div>
                <div class="stat-label">Vector embeddings</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(st.session_state.messages)}</div>
                <div class="stat-label">Câu hỏi đã trả lời</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Hướng Dẫn Sử Dụng")
        st.markdown("""
        - **Đặt câu hỏi cụ thể** về pháp luật Việt Nam
        - **Ví dụ**: "Phạt bao nhiêu khi vượt đèn đỏ?"
        - **Ví dụ**: "Thủ tục đăng ký kinh doanh như thế nào?"
        - Hệ thống sẽ tìm kiếm và trả lời dựa trên văn bản pháp luật chính thức
        """)
        
        st.markdown("---")
        if st.button("🗑️ Xóa Lịch Sử Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>🙋‍♂️ Bạn:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            ai_response = message["content"]
            st.markdown(f"""
            <div class="ai-message">
                <strong>🤖 Luật sư AI:</strong><br>
                <strong>{ai_response['title']}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="legal-citation">
                📜 <strong>Căn cứ pháp lý:</strong> {ai_response['citation']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(ai_response['answer'])
            
            if 'search_info' in message:
                with st.expander("🔍 Thông tin tìm kiếm chi tiết"):
                    search_info = message['search_info']
                    st.json({
                        "Độ chính xác": f"{search_info['score']:.3f}",
                        "Mã luật": search_info['law_id'],
                        "Điều": search_info['article_id'],
                        "Tiêu đề": search_info['title'],
                        "Nội dung": search_info['text']
                    })
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    with st.container():
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Nhập câu hỏi pháp luật của bạn...",
                placeholder="Ví dụ: Phạt bao nhiều khi không đội mũ bảo hiểm?",
                key="user_input",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("📤 Gửi", use_container_width=True)
    
    # Process user input
    if send_button and user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Show loading
        with st.spinner("🔍 Đang tìm kiếm và phân tích..."):
            # Process query
            search_result, ai_response = process_query(
                st.session_state.rag_system,
                st.session_state.model,
                user_input
            )
            
            if ai_response and search_result:
                # Add AI response
                message_data = {
                    "role": "assistant",
                    "content": ai_response,
                    "search_info": search_result,
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(message_data)
            else:
                st.error("Không thể xử lý câu hỏi. Vui lòng thử lại.")
        
        # Clear input and rerun
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
        <p>⚖️ Hệ thống tư vấn pháp luật AI - Phiên bản 1.0</p>
        <p><em>Lưu ý: Thông tin chỉ mang tính chất tham khảo. Vui lòng tham khảo ý kiến chuyên gia pháp lý cho các vấn đề phức tạp.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()