# Vietnamese Legal RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system designed for Vietnamese legal documents, featuring semantic search and an intelligent web interface for legal consultation.

## Features

### Core RAG System
- **Semantic Chunking**: Preserves legal document structure and context
- **Vietnamese Optimization**: Uses AITeamVN/Vietnamese_Embedding model for Vietnamese text understanding
- **Efficient Search**: FAISS-based vector similarity search
- **Legal-Specific Preprocessing**: Handles Vietnamese legal document formatting and structure
- **Performance Evaluation**: Built-in evaluation metrics (Accuracy@K, MRR@K)

### Web Interface (Streamlit)
- **Interactive Chat Interface**: User-friendly legal consultation experience
- **AI-Powered Responses**: Powered by Google Gemini AI with specialized legal prompting
- **Real-time Streaming**: Token-by-token response generation for better user experience
- **Auto-clearing Input**: Input field clears automatically after submission
- **Smart Auto-scroll**: Automatically scrolls to new responses
- **Legal Citations**: Precise references to Vietnamese legal documents with proper formatting
- **System Dashboard**: Real-time statistics showing document count and usage metrics

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Before using the web interface, you need to:

1. **Set up Google Gemini API**: 
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Update the API key in `streamlit_app.py` line 191 (replace the placeholder)

2. **Prepare Legal Data**:
   - Ensure `legal_chunks.json` contains your processed legal documents
   - Use the pre-built `faiss_index.index` or build your own using the RAG system

## Usage

### 1. Web Interface (Streamlit)

Launch the interactive web application for legal consultation:

```bash
streamlit run streamlit_app.py
```

The web interface provides:
- **Legal Question Answering**: Ask questions in Vietnamese about legal matters
- **Intelligent Document Search**: Semantic search through Vietnamese legal corpus
- **Structured Responses**: Get answers with proper legal citations and references
- **Real-time Statistics**: View system performance and document coverage

### 2. Basic RAG Pipeline

```python
from vietnamese_legal_rag import VietnameseLegalRAG

# Initialize RAG system
rag = VietnameseLegalRAG()

# Load pre-processed legal chunks
with open("legal_chunks.json", 'r', encoding='utf-8') as f:
    legal_chunks_data = json.load(f)
    rag.chunks = [LegalChunk(**chunk_data) for chunk_data in legal_chunks_data]

# Load pre-built FAISS index
rag.load_index("faiss_index.index")

# Search for relevant legal documents
results = rag.search("quy định về lái xe không bằng lái", top_k=3)

# Process results
for result in results:
    print(f"Law: {result['law_id']}")
    print(f"Article: {result['article_id']}")
    print(f"Title: {result['title']}")
    print(f"Relevance Score: {result['score']:.3f}")
```

### 3. Build RAG System from Scratch

```python
from vietnamese_legal_rag import VietnameseLegalRAG

# Initialize RAG system
rag = VietnameseLegalRAG()

# Load and chunk legal corpus
rag.load_and_chunk_data('legal_corpus.json')

# Build embeddings and search index
rag.build_embeddings()
rag.build_index()

# Search with the newly built system
results = rag.search("your question here", top_k=5)
```

### 4. Simple Search Script

```bash
python search.py
```

### 5. Evaluation

Run evaluation on question-answer pairs:

```bash
python evaluation_on_rag.py
```

## File Structure

```
RAG/
├── legal_corpus.json          # Original legal documents
├── legal_chunks.json          # Pre-processed legal chunks
├── faiss_index.index          # Pre-built FAISS search index
├── train_question_answer.json # Evaluation dataset
├── vietnamese_legal_rag.py    # Main RAG implementation
├── streamlit_app.py           # Web UI for legal consultation
├── search.py                  # Simple search script
├── evaluation_on_rag.py       # Performance evaluation script
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Model Details

### Embedding Model
- **Model**: AITeamVN/Vietnamese_Embedding
- **Max Sequence Length**: 2048 tokens
- **Embedding Dimension**: 1024
- **Similarity**: Cosine similarity with FAISS IndexFlatIP

### Language Model (Web Interface)
- **Model**: Google Gemini (gemini-2.5-flash)
- **Purpose**: Generate structured legal responses
- **Output Format**: JSON with title, citation, and answer
- **Specialization**: Vietnamese legal document interpretation

## Key Design Decisions

### Semantic Chunking Strategy

**Why semantic chunking over fixed-size?**

1. **Legal Context Preservation**: Legal articles contain complete concepts that shouldn't be split
2. **Better Retrieval Quality**: Semantic boundaries improve search relevance  
3. **Document Structure**: Maintains hierarchical structure (articles, subsections)
4. **Model Optimization**: Works within 2048 token limit of Vietnamese_Embedding

### Preprocessing Approach

- **Text Normalization**: Removes extra whitespace, preserves legal formatting
- **Structure Preservation**: Keeps article titles, IDs, and numbering
- **Vietnamese-Specific**: Handles Vietnamese text characteristics
- **Token-Aware Chunking**: Respects model token limits while maintaining semantic coherence

## Performance

The system includes built-in evaluation metrics:

- **Accuracy@K**: Measures if relevant documents appear in top-K results
- **MRR@K**: Mean Reciprocal Rank measures ranking quality
- **Evaluation Dataset**: Uses `train_question_answer.json` for testing

## Performance Tips

1. **Batch Processing**: Embeddings generated in batches of 32 for efficiency
2. **Index Persistence**: Save/load FAISS index for faster subsequent runs
3. **Memory Management**: Large legal corpus handled efficiently with FAISS
4. **Chunking Optimization**: Semantic chunking preserves legal context while fitting model limits

## Example Queries

- "Phạt bao nhiêu khi vượt đèn đỏ?"
- "Quy định về lái xe không bằng lái"
- "Thủ tục đăng ký kinh doanh"
- "Mức phạt vi phạm giao thông"
- "Quy định về bảo hiểm y tế"

## Legal Disclaimer

This system provides information for reference purposes only. For complex legal matters, please consult with qualified legal professionals.