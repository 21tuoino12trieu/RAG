import json
import re
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dataclasses import dataclass
from transformers import AutoTokenizer, pipeline
import warnings
import os
warnings.filterwarnings("ignore")

@dataclass
class LegalChunk:
    text: str
    law_id: str
    article_id: str
    title: str
    chunk_id: str

pipe = pipeline("fill-mask",model="vinai/phobert-base")
    
class VietnameseLegalRAG:
    def __init__(self, model_name: str = "AITeamVN/Vietnamese_Embedding", max_seq_length: int = 2048):
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_length
        self.chunks: List[LegalChunk] = []
        self.embeddings = None
        self.index = None
        
        # Initialize tokenizer for counting tokens
        try:
            self.tokenizer = self.initialize_tokenizer()
        except:
            # Fallback simple tokenizer
            self.tokenizer = None

    def initialize_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        return tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer is None:
            # Fallback: rough estimation
            return len(text.split()) * 1.3  # Vietnamese words tend to be tokenized into ~1.3 tokens
        
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        return len(tokens)        
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess Vietnamese legal text"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Preserve legal formatting
        text = re.sub(r'(\d+)\.\s+', r'\1. ', text)  # Normalize numbering
        
        return text
    
    def semantic_chunk_article(self, article: Dict[str, Any], law_id: str, max_tokens: int = 1800) -> List[LegalChunk]:
        """Semantically chunk a legal article"""
        chunks = []
        
        title = article.get('title', '')
        text = article.get('text', '')
        article_id = article.get('article_id', '')
        
        # Preprocess text
        preprocessed_text = self.preprocess_text(text)
        
        # Create full text with title for context
        full_text = f"{title}\n{preprocessed_text}"
        
        # Check if entire article fits within token limit
        if self.count_tokens(full_text) <= max_tokens:
            chunks.append(LegalChunk(
                text=full_text,
                law_id=law_id,
                article_id=article_id,
                title=title,
                chunk_id=f"{law_id}_{article_id}_full"
            ))
        else:
            # Split by numbered subsections if text is too long
            subsections = re.split(r'\n(\d+)\.\s+', preprocessed_text)
            
            if len(subsections) > 1:
                # Process subsections
                current_chunk = title + "\n"
                current_subsection = 1
                
                for i in range(1, len(subsections), 2):
                    if i + 1 < len(subsections):
                        section_number = subsections[i]
                        section_text = subsections[i + 1].strip()
                        section_content = f"{section_number}. {section_text}"
                        
                        # Check if adding this section exceeds limit
                        test_chunk = current_chunk + section_content + "\n"
                        
                        if self.count_tokens(test_chunk) > max_tokens and current_chunk != title + "\n":
                            # Save current chunk
                            chunks.append(LegalChunk(
                                text=current_chunk.strip(),
                                law_id=law_id,
                                article_id=article_id,
                                title=title,
                                chunk_id=f"{law_id}_{article_id}_{current_subsection}"
                            ))
                            
                            # Start new chunk
                            current_chunk = title + "\n" + section_content + "\n"
                            current_subsection += 1
                        else:
                            current_chunk += section_content + "\n"
                
                # Add final chunk
                if current_chunk.strip() != title:
                    chunks.append(LegalChunk(
                        text=current_chunk.strip(),
                        law_id=law_id,
                        article_id=article_id,
                        title=title,
                        chunk_id=f"{law_id}_{article_id}_{current_subsection}"
                    ))
            else:
                # No subsections, do simple text splitting
                sentences = re.split(r'(?<=[.!?])\s+', preprocessed_text)
                current_chunk = title + "\n"
                chunk_num = 1
                
                for sentence in sentences:
                    test_chunk = current_chunk + sentence + " "
                    
                    if self.count_tokens(test_chunk) > max_tokens and current_chunk != title + "\n":
                        chunks.append(LegalChunk(
                            text=current_chunk.strip(),
                            law_id=law_id,
                            article_id=article_id,
                            title=title,
                            chunk_id=f"{law_id}_{article_id}_{chunk_num}"
                        ))
                        
                        current_chunk = title + "\n" + sentence + " "
                        chunk_num += 1
                    else:
                        current_chunk += sentence + " "
                
                # Add final chunk
                if current_chunk.strip() != title:
                    chunks.append(LegalChunk(
                        text=current_chunk.strip(),
                        law_id=law_id,
                        article_id=article_id,
                        title=title,
                        chunk_id=f"{law_id}_{article_id}_{chunk_num}"
                    ))
        
        return chunks
    
    def load_and_chunk_data(self, json_file_path: str):
        """Load and chunk legal corpus"""
        print("Loading legal corpus...")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            legal_data = json.load(f)
        
        print(f"Processing {len(legal_data)} legal documents...")
        
        for doc in legal_data:
            law_id = doc.get('law_id', '')
            articles = doc.get('articles', [])
            
            for article in articles:
                chunks = self.semantic_chunk_article(article, law_id)
                self.chunks.extend(chunks)
        
        print(f"Created {len(self.chunks)} chunks from legal corpus")
    
    def build_embeddings(self):
        """Build embeddings for all chunks"""
        print("Building embeddings...")
        
        texts = [chunk.text for chunk in self.chunks]
        
        # Generate embeddings in batches
        batch_size = 32
        embeddings_list = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=True)
            embeddings_list.append(batch_embeddings)
        
        self.embeddings = np.vstack(embeddings_list)
        print(f"Generated embeddings shape: {self.embeddings.shape}")
    
    def build_index(self):
        """Build FAISS index for similarity search"""
        print("Building FAISS index...")
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"Built index with {self.index.ntotal} vectors")

    def load_index(self, file_path:str):
        self.index = faiss.read_index(file_path)
        return self.index
    
    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search for relevant legal chunks"""
        # Generate query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            chunk = self.chunks[idx]
            results.append({
                'rank': i + 1,
                'score': float(score),
                'law_id': chunk.law_id,
                'article_id': chunk.article_id,
                'title': chunk.title,
                'text': chunk.text,
                'chunk_id': chunk.chunk_id
            })
        
        return results
    
    def save_to_kaggle(self, output_name:str = "legal_rag_system"):
        base_path = f"/kaggle/working/{output_name}"
        os.makedirs(base_path, exist_ok=True)

        print(f"Saving RAG System to Kaggle output: {base_path}")

        #Save FAISS Index
        index_path = os.path.join(base_path,"faiss_index.index")
        faiss.write_index(self.index, index_path)
        print(f"Saved FAISS Index: {index_path}")

        # Save embeddings
        embedding_path = os.path.join(base_path,"embeddings.npz")
        np.savez_compressed(embedding_path,embeddings = self.embeddings)

        # Saved chunks
        chunks_path = os.path.join(base_path,"chunks.json")
        chunks_data = []
        for chunk in self.chunks:
            chunks_data.append({
                'text':chunk.text,
                "law_id":chunk.law_id,
                "article_id":chunk.article_id,
                "title":chunk.title,
                "chunk_id":chunk.chunk_id
            })

        with open(chunks_path,"w",encoding = "utf-8") as f:
            json.dump(chunks_data,f,ensure_ascii=False,indent=2)
        print(f"Saved chunks at {chunks_path}")

        # Saved metadata
        metadata_path = os.path.join(base_path, "metadata.json")
        metadata = {
            'model_name':self.model.model_name,
            'max_seq_length': self.model.max_seq_length,
            'num_chunks': len(self.chunks),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else None,
            'index_total': self.index.ntotal if self.index is not None else None
        }

        with open(metadata_path,"w",encoding ="utf-8") as f:
            json.dump(metadata,f,ensure_ascii=False,indent=2)


    def load_data_from_kaggle(self, dataset_path:str):
        metadata_path = os.path.join(dataset_path, "metadata.json")
        if os.path.exist(metadata_path):
            with open(metadata_path,'r',encoding="utf-8") as f:
                metadata = json.load(f)


        index_path = os.path.join(dataset_path, "faiss_index.index")
        self.index = faiss.read_index(index_path)

        embeddings_npz_path = os.path.join(dataset_path, "embeddings.npz")
        loaded_data = np.load(embeddings_npz_path)
        self.embeddings = loaded_data['embeddings']

        chunks_path = os.path.join(dataset_path,"chunks.json")
        with open(chunks_path,'r',encoding='utf-8') as f:
            chunks_data = json.load(f)
            
        self.chunks = []
        for chunk_data in chunks_data:
            self.chunks.append(LegalChunk(**chunk_data))
            
def main():
    print("Initialize Vietnames Legal RAG from Kaggle dataset...")
    rag = VietnameseLegalRAG()
    rag.load_data_from_kaggle("legal_corpus.json")
    print("Building Embeddings...")
    rag.build_embeddings()
    print("Building Index...")
    rag.build_index()

if __name__ == "__main__":
    main()