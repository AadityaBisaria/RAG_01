#!/usr/bin/env python3
"""
Simple script to run RAG on the Hotspot Paper
"""

from rag_phase1 import RAGDataPipeline, ChromaDBManager
from rag_phase2 import LMStudioClient, RAGSystem

def main():
    print("🔬 Hotspot Paper RAG System")
    print("=" * 40)
    
    # Step 1: Process the PDF
    print("\n📚 Processing Hotspot Paper...")
    pipeline = RAGDataPipeline(chroma_db_path="./hotspot_chroma_db", data_path="./data")
    num_chunks = pipeline.ingest_documents("./data/Hotspot_Paper.pdf")
    print(f"✅ Created {num_chunks} chunks from Hotspot Paper")
    
    # Step 2: Initialize RAG system with Hermes
    print("\n🤖 Initializing RAG system with Hermes model...")
    try:
        db_manager = ChromaDBManager(persist_directory="./hotspot_chroma_db")
        llm_client = LMStudioClient(base_url="http://127.0.0.1:1234/v1")
        rag = RAGSystem(db_manager, llm_client)
        print("✅ RAG system ready!")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("💡 Make sure LM Studio is running with Hermes model")
        return
    
    # Step 3: Test with hotspot paper questions
    print("\n🔍 Testing with Hotspot Paper questions...")
    
    hotspot_queries = [
        "What is the main research question in this hotspot paper?",
        "What methodology was used to detect hotspots?",
        "What are the key findings about hotspots?",
        "What datasets were used in this study?",
        "What are the limitations mentioned in the paper?",
        "What future work is suggested?",
        "How do hotspots relate to climate change?",
        "What statistical methods were applied?"
    ]
    
    for i, query in enumerate(hotspot_queries, 1):
        print(f"\n{'='*60}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*60}")
        
        try:
            result = rag.query(
                query,
                n_results=3,
                use_reranking=True,
                retrieve_n=8,
                analyze_query=True,
                verify_citations=True
            )
            
            print(f"\n📋 Answer:")
            print(result['answer'])
            
            if result.get('citations'):
                print(f"\n📚 Sources: {', '.join(result['citations'])}")
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    # Step 4: Interactive mode
    print("\n" + "="*60)
    interactive_choice = input("🎮 Enter interactive mode to ask your own questions? (y/n): ").strip().lower()
    if interactive_choice == 'y':
        print("\n💬 Ask questions about the Hotspot Paper!")
        print("Type 'quit' to exit")
        
        while True:
            try:
                user_input = input("\nYour question: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye! 👋")
                    break
                
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
    
    print("\n✅ Hotspot Paper RAG testing completed!")

if __name__ == "__main__":
    main()
