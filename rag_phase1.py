"""
RAG System Phase 1: Data Preparation & ChromaDB Setup
Handles document ingestion, chunking, and vector storage
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
import PyPDF2
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """Handles document loading and preprocessing"""
    
    def __init__(self):
        self.supported_formats = ['.md', '.pdf', '.txt']
    
    def load_documents(self, doc_path: str) -> List[Dict[str, Any]]:
        """Load documents from a directory or single file"""
        path = Path(doc_path)
        documents = []
        
        if path.is_file():
            doc = self._load_single_doc(path)
            if doc:
                documents.append(doc)
        elif path.is_dir():
            for file_path in path.rglob('*'):
                if file_path.suffix.lower() in self.supported_formats:
                    doc = self._load_single_doc(file_path)
                    if doc:
                        documents.append(doc)
        
        return documents
    
    def _load_single_doc(self, file_path: Path) -> Dict[str, Any]:
        """Load a single document file"""
        try:
            if file_path.suffix.lower() == '.pdf':
                content = self._extract_pdf(file_path)
            else:  # .md or .txt
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            return {
                'content': content,
                'filename': file_path.name,
                'filepath': str(file_path),
                'doc_type': file_path.suffix.lower()
            }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _extract_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        return text
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        return text.strip()


class TextChunker:
    """Handles intelligent text chunking using LangChain's RecursiveCharacterTextSplitter"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: Target size in characters (roughly ~200-250 tokens)
            chunk_overlap: Overlap between chunks to preserve context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize LangChain's RecursiveCharacterTextSplitter
        # This intelligently splits on separators like paragraphs, sentences, words
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence endings
                "! ",    # Exclamation endings
                "? ",    # Question endings
                "; ",    # Semicolon breaks
                ", ",    # Comma breaks
                " ",     # Space breaks
                "",      # Character breaks (fallback)
            ]
        )
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split document into overlapping chunks with metadata using LangChain"""
        content = document['content']
        
        # Use LangChain's intelligent splitting
        text_chunks = self.text_splitter.split_text(content)
        
        # Convert to our format with metadata
        chunks = []
        for chunk_id, chunk_text in enumerate(text_chunks):
            chunks.append(self._create_chunk_metadata(
                chunk_text, 
                document, 
                chunk_id
            ))
        
        return chunks
    
    def _create_chunk_metadata(self, chunk_text: str, document: Dict[str, Any], chunk_id: int) -> Dict[str, Any]:
        """Create chunk with metadata"""
        return {
            'content': chunk_text.strip(),
            'metadata': {
                'source': document['filename'],
                'doc_type': document['doc_type'],
                'chunk_id': chunk_id,
                'filepath': document['filepath']
            },
            'id': f"{document['filename']}_chunk_{chunk_id}"
        }


class ChromaDBManager:
    """Manages ChromaDB operations for RAG system"""
    
    def __init__(self, collection_name: str = "product_docs", persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB with persistent storage
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Load embedding model - using BGE large for better quality
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        
        # Load reranking model for better retrieval quality
        print("Loading reranking model...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Product guides and FAQs for RAG system"}
        )
        
        print(f"✓ ChromaDB initialized: {self.collection.count()} documents in collection")
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to ChromaDB"""
        documents = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [chunk['id'] for chunk in chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True).tolist()
        
        # Add to ChromaDB in batches (ChromaDB has limits)
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
        
        print(f"✓ Added {len(documents)} chunks to ChromaDB")
    
    def query(self, query_text: str, n_results: int = 5, filter_metadata: Dict = None) -> Dict:
        """
        Query ChromaDB for relevant chunks
        
        Args:
            query_text: User query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter (e.g., {"source": "faq.md"})
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=filter_metadata
        )
        
        return results
    
    def query_with_reranking(self, query_text: str, retrieve_n: int = 10, final_n: int = 5, filter_metadata: Dict = None) -> Dict:
        """
        Query with reranking for better results
        
        Args:
            query_text: User query
            retrieve_n: Number of chunks to retrieve initially
            final_n: Number of chunks to return after reranking
            filter_metadata: Optional metadata filter
            
        Returns:
            Dict with reranked results
        """
        # Step 1: Retrieve more chunks than needed
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        
        initial_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=retrieve_n,
            where=filter_metadata
        )
        
        if not initial_results['documents'][0]:
            return initial_results
        
        # Step 2: Prepare pairs for reranking
        query_doc_pairs = []
        for doc in initial_results['documents'][0]:
            query_doc_pairs.append([query_text, doc])
        
        # Step 3: Rerank with cross-encoder
        rerank_scores = self.reranker.predict(query_doc_pairs)
        
        # Step 4: Sort by rerank scores and take top final_n
        scored_results = list(zip(
            initial_results['documents'][0],
            initial_results['metadatas'][0],
            initial_results['distances'][0],
            initial_results['ids'][0],
            rerank_scores
        ))
        
        # Sort by rerank scores (descending - higher is better)
        scored_results.sort(key=lambda x: x[4], reverse=True)
        
        # Take top final_n results
        top_results = scored_results[:final_n]
        
        # Reconstruct results in original format
        reranked_results = {
            'documents': [[item[0] for item in top_results]],
            'metadatas': [[item[1] for item in top_results]],
            'distances': [[item[2] for item in top_results]],
            'ids': [[item[3] for item in top_results]]
        }
        
        return reranked_results
    
    def clear_collection(self):
        """Clear all documents from collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(self.collection_name)
        print("✓ Collection cleared")


class RAGDataPipeline:
    """Main pipeline for Phase 1: Data preparation and storage"""
    
    def __init__(self, chroma_db_path: str = "./chroma_db", data_path: str = "./data"):
        self.processor = DocumentProcessor()
        self.chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        self.db = ChromaDBManager(persist_directory=chroma_db_path)
        self.data_path = data_path
    
    def ingest_documents(self, doc_path: str):
        """Complete pipeline: Load -> Preprocess -> Chunk -> Store"""
        print(f"\n{'='*50}")
        print("Starting RAG Data Ingestion Pipeline")
        print(f"{'='*50}\n")
        
        # Step 1: Load documents
        print("📂 Step 1: Loading documents...")
        documents = self.processor.load_documents(doc_path)
        print(f"✓ Loaded {len(documents)} documents\n")
        
        # Step 2: Preprocess and chunk
        print("✂️  Step 2: Preprocessing and chunking...")
        all_chunks = []
        for doc in documents:
            doc['content'] = self.processor.preprocess_text(doc['content'])
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
            print(f"  - {doc['filename']}: {len(chunks)} chunks")
        print(f"✓ Created {len(all_chunks)} total chunks\n")
        
        # Step 3: Store in ChromaDB
        print("💾 Step 3: Storing in ChromaDB...")
        self.db.add_documents(all_chunks)
        
        print(f"\n{'='*50}")
        print("✓ Pipeline Complete!")
        print(f"{'='*50}\n")
        
        return len(all_chunks)
    
    def test_retrieval(self, query: str, n_results: int = 3):
        """Test the retrieval system"""
        print(f"\n🔍 Testing Query: '{query}'")
        print("-" * 50)
        
        results = self.db.query(query, n_results=n_results)
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f"\nResult {i+1} (similarity: {1-distance:.3f}):")
            print(f"Source: {metadata['source']}")
            print(f"Content: {doc[:200]}...")
            print("-" * 50)


# Example usage and testing
if __name__ == "__main__":
    # Create sample documents for testing
    sample_docs_dir = "./sample_docs"
    os.makedirs(sample_docs_dir, exist_ok=True)
    
    # Sample Product Guide
    with open(f"{sample_docs_dir}/product_guide.md", "w") as f:
        f.write("""# Product Setup Guide

## Installation
To install our product, follow these steps:
1. Download the installer from our website
2. Run the installer with administrator privileges
3. Follow the on-screen instructions
4. Restart your computer after installation

## Configuration
After installation, you need to configure the product:
- Open Settings from the main menu
- Enter your license key
- Choose your preferred language
- Set up your workspace preferences

## System Requirements
Minimum requirements:
- Windows 10 or macOS 10.15+
- 8GB RAM
- 500MB free disk space
- Internet connection for activation
""")
    
    # Sample FAQ
    with open(f"{sample_docs_dir}/faq.md", "w") as f:
        f.write("""# Frequently Asked Questions

## How do I reset my password?
To reset your password:
1. Click on "Forgot Password" on the login page
2. Enter your email address
3. Check your email for a reset link
4. Follow the link and create a new password

## What payment methods do you accept?
We accept the following payment methods:
- Credit cards (Visa, Mastercard, AmEx)
- PayPal
- Bank transfer for enterprise customers

## How do I contact support?
You can contact our support team through:
- Email: support@example.com
- Live chat on our website (Mon-Fri 9AM-5PM)
- Phone: 1-800-EXAMPLE

## Can I get a refund?
Yes, we offer a 30-day money-back guarantee. Contact support with your order number to process a refund.
""")
    
    print("Sample documents created!\n")
    
    # Initialize pipeline
    pipeline = RAGDataPipeline()
    
    # Ingest documents
    pipeline.ingest_documents(sample_docs_dir)
    
    # Test retrieval
    print("\n" + "="*50)
    print("TESTING RETRIEVAL SYSTEM")
    print("="*50)
    
    test_queries = [
        "How do I install the product?",
        "What are the system requirements?",
        "How can I reset my password?",
        "What payment methods are available?"
    ]
    
    for query in test_queries:
        pipeline.test_retrieval(query, n_results=2)