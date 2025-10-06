#!/usr/bin/env python3
"""
Test RAG system with a PDF paper
Customized for user's Hermes model at http://127.0.0.1:1234
"""

import os
from pathlib import Path
from rag_phase1 import RAGDataPipeline, ChromaDBManager
from rag_phase2 import LMStudioClient, RAGSystem

def setup_pdf_test(pdf_path: str):
    """Set up RAG system for PDF testing"""
    print("🔬 PDF Paper RAG Test Setup")
    print("=" * 50)
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        print("💡 Please provide the correct path to your PDF file")
        return None
    
    print(f"📄 Found PDF: {pdf_path}")
    
    # Create a dedicated directory for the paper
    paper_dir = "./paper_test"
    os.makedirs(paper_dir, exist_ok=True)
    
    # Copy PDF to the test directory (or just use the original path)
    print(f"📂 Using PDF from: {pdf_path}")
    
    # Initialize Phase 1: Document processing
    print("\n📚 Processing PDF document...")
    pipeline = RAGDataPipeline(chroma_db_path="./paper_chroma_db")
    
    # Ingest the PDF
    num_chunks = pipeline.ingest_documents(pdf_path)
    print(f"✅ Created {num_chunks} chunks from PDF")
    
    return pipeline

def test_hermes_model():
    """Test connection to Hermes model"""
    print("\n🤖 Testing Hermes Model Connection...")
    
    try:
        # Custom LM Studio client with your specific URL
        llm_client = LMStudioClient(base_url="http://127.0.0.1:1234/v1")
        
        # Test the connection
        print("✅ Connected to Hermes model!")
        
        # Test a simple generation
        print("🧪 Testing model generation...")
        test_prompt = "What is the main topic of this document?"
        response = llm_client.generate(test_prompt, temperature=0.1, max_tokens=100)
        print(f"✅ Model response: {response[:100]}...")
        
        return llm_client
        
    except Exception as e:
        print(f"❌ Failed to connect to Hermes model: {e}")
        print("💡 Make sure:")
        print("   - LM Studio is running")
        print("   - Hermes-3-Llama-3.2-3B model is loaded")
        print("   - Server is started on http://127.0.0.1:1234")
        return None

def test_paper_queries(llm_client, pipeline):
    """Test various queries on the PDF paper"""
    print("\n🔍 Testing Paper Queries")
    print("=" * 50)
    
    # Initialize RAG system
    db_manager = ChromaDBManager(persist_directory="./paper_chroma_db")
    rag = RAGSystem(db_manager, llm_client)
    
    # Sample queries for academic papers
    paper_queries = [
        "What is the main research question or objective?",
        "What methodology was used in this study?",
        "What are the key findings or results?",
        "What are the main conclusions?",
        "What datasets were used?",
        "What are the limitations mentioned?",
        "What future work is suggested?",
        "What related work is cited?"
    ]
    
    print(f"📝 Testing {len(paper_queries)} academic queries...")
    
    for i, query in enumerate(paper_queries, 1):
        print(f"\n{'='*60}")
        print(f"QUERY {i}/{len(paper_queries)}: {query}")
        print(f"{'='*60}")
        
        try:
            result = rag.query(
                query,
                n_results=3,
                use_reranking=True,
                retrieve_n=8,
                analyze_query=True,
                structured_output=True,
                verify_citations=True
            )
            
            # Display results
            print(f"\n📋 Answer:")
            print(result['answer'])
            
            if result.get('citations'):
                print(f"\n📚 Sources: {', '.join(result['citations'])}")
            
            if 'structured_response' in result:
                structured = result['structured_response']
                print(f"🎯 Confidence: {structured.get('confidence', 'unknown')}")
            
            # Show citation report
            if 'citation_report' in result:
                print(f"\n{result['citation_report']}")
                
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    return rag

def interactive_paper_mode(rag):
    """Interactive mode for exploring the paper"""
    print("\n" + "="*60)
    print("📚 INTERACTIVE PAPER EXPLORATION")
    print("="*60)
    print("Ask questions about your paper!")
    print("Commands: 'quit' to exit, 'summary' for paper summary")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n💬 Your question about the paper: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye! 👋")
                break
            
            if user_input.lower() == 'summary':
                print("📄 Generating paper summary...")
                user_input = "Provide a comprehensive summary of this paper including the main research question, methodology, key findings, and conclusions."
            
            if not user_input:
                continue
            
            result = rag.query(
                user_input,
                n_results=5,
                use_reranking=True,
                retrieve_n=10,
                analyze_query=True,
                verify_citations=True
            )
            
            print(f"\n📋 Answer:")
            print(result['answer'])
            
            if result.get('citations'):
                print(f"\n📚 Sources: {', '.join(result['citations'])}")
                
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    print("🔬 PDF Paper RAG Testing with Hermes Model")
    print("=" * 60)
    
    # Get PDF path from user
    pdf_path = input("📄 Enter path to your PDF paper: ").strip()
    
    if not pdf_path:
        print("❌ No PDF path provided")
        return
    
    # Step 1: Set up PDF processing
    pipeline = setup_pdf_test(pdf_path)
    if not pipeline:
        return
    
    # Step 2: Test Hermes model connection
    llm_client = test_hermes_model()
    if not llm_client:
        return
    
    # Step 3: Test paper queries
    rag = test_paper_queries(llm_client, pipeline)
    
    # Step 4: Interactive mode
    print("\n" + "="*60)
    interactive_choice = input("🎮 Enter interactive mode? (y/n): ").strip().lower()
    if interactive_choice == 'y':
        interactive_paper_mode(rag)
    
    print("\n✅ PDF paper testing completed!")

if __name__ == "__main__":
    main()
