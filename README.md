# Vietnamese Legal RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system optimized for Vietnamese legal documents with an intelligent web interface for legal consultation.

## Features

### Core RAG System
- **Semantic Chunking**: Preserves legal document structure and context
- **Vietnamese Optimization**: Uses AITeamVN/Vietnamese_Embedding model
- **Two-Stage Retrieval**: Fast retrieval + precision reranking
- **Vietnamese Reranking**: Optional AITeamVN/Vietnamese_Reranker integration
- **Efficient Search**: FAISS-based similarity search
- **Legal-Specific Preprocessing**: Handles Vietnamese legal document formatting
- **Performance Evaluation**: Compare retrieval vs reranked results

### Web Interface (Streamlit)
- **Interactive Chat Interface**: User-friendly legal consultation experience
- **Intelligent AI Responses**: Powered by Google Gemini AI with legal expertise
- **Real-time Streaming**: Token-by-token response generation for better UX
- **Auto-clearing Input**: Input field clears automatically after submission
- **Smart Auto-scroll**: Automatically scrolls to new responses
- **Legal Citations**: Precise references to Vietnamese legal documents
- **System Dashboard**: Real-time statistics and usage metrics

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
   - Build the FAISS index using `vietnamese_legal_rag.py` or use existing `faiss_index.index`

## Usage

### 1. Basic Pipeline (with Reranking)

```python
from vietnamese_legal_rag import VietnameseLegalRAG

# Initialize RAG system with reranking
rag = VietnameseLegalRAG(enable_reranking=True)

# Load and process legal corpus
rag.load_and_chunk_data('legal_corpus.json')

# Build embeddings and search index
rag.build_embeddings()
rag.build_index()

# Search with reranking
results = rag.search("quy định về tuần tra canh gác đê điều", top_k=3)

# Compare methods
comparison = rag.compare_search_methods("nhiệm vụ bảo vệ đê", top_k=3)
```

### 1b. Retrieval-Only Mode

```python
# For speed-critical applications
rag = VietnameseLegalRAG(enable_reranking=False)
# ... same usage
```

### 2. Run Complete Demo

```bash
python example_usage.py
```

### 3. Load Pre-built Index

```bash
python example_usage.py --load
```

### 4. Analyze Chunking Strategies

```bash
python chunking_analysis.py
```

### 5. Evaluate Reranking Performance

```bash
python reranking_evaluation.py
```

### 6. Web Interface (Streamlit)

Launch the interactive web application for legal consultation:

```bash
streamlit run streamlit_app.py
```

The web interface provides:
- **User-friendly chat interface** for legal questions
- **Vietnamese legal document search** with AI-powered responses
- **Real-time streaming responses** from Gemini AI
- **Auto-clearing input field** and smooth scrolling
- **Legal citations** with precise article references
- **System statistics** showing database size and query count

## Key Design Decisions

### Semantic Chunking Strategy

**Why semantic chunking over fixed-size?**

1. **Legal Context Preservation**: Legal articles contain complete concepts that shouldn't be split
2. **Better Retrieval Quality**: Semantic boundaries improve search relevance  
3. **Document Structure**: Maintains hierarchical structure (articles, subsections)
4. **Model Optimization**: Works within 2048 token limit of Vietnamese_Embedding

### Two-Stage Retrieval Strategy

**Why add reranking?**

1. **Better Precision**: Cross-encoder rerankers understand query-document relationships better than bi-encoders
2. **Legal Context**: Handles nuanced legal language and concept relationships
3. **Flexible Performance**: Fast retrieval for exploration, precise reranking for accuracy

### Preprocessing Approach

- **Text Normalization**: Removes extra whitespace, preserves legal formatting
- **Structure Preservation**: Keeps article titles, IDs, and numbering
- **Vietnamese-Specific**: Handles Vietnamese text characteristics

## File Structure

```
RAG/
├── legal_corpus.json          # Input legal documents
├── vietnamese_legal_rag.py    # Main RAG implementation with reranking
├── streamlit_app.py           # Web UI for legal consultation
├── example_usage.py           # Usage examples
├── chunking_analysis.py       # Chunking strategy comparison
├── reranking_evaluation.py    # Reranking performance evaluation
├── requirements.txt           # Dependencies
├── legal_chunks.json          # Pre-processed legal chunks
├── faiss_index.index          # Pre-built FAISS index
└── README.md                 # This file
```

## Model Details

### Embedding Model
- **Model**: AITeamVN/Vietnamese_Embedding
- **Max Sequence Length**: 2048 tokens
- **Embedding Dimension**: 1024
- **Similarity**: Cosine similarity with FAISS

### Reranking Model (Optional)
- **Model**: AITeamVN/Vietnamese_Reranker
- **Type**: Cross-encoder for query-document scoring
- **Fallback**: Graceful fallback to retrieval-only if unavailable

## Performance Tips

1. **Batch Processing**: Embeddings generated in batches of 32
2. **Index Saving**: Save/load index for faster subsequent runs
3. **Memory Management**: Large corpus handled efficiently with FAISS
4. **Reranking Trade-offs**: 
   - Use reranking for precision-critical legal research
   - Use retrieval-only for speed-critical applications
   - Reranking adds ~2-3x latency but improves accuracy

## Example Queries

- "quy định về tuần tra canh gác đê điều"
- "nhiệm vụ của lực lượng bảo vệ đê"
- "trang bị dụng cụ cho đội tuần tra"
- "báo động lũ cấp độ khác nhau"