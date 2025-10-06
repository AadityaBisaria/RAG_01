#!/usr/bin/env python3
"""
Example usage of the enhanced RAG system with query understanding and citation verification
"""

from rag_phase1 import RAGDataPipeline, ChromaDBManager
from rag_phase2 import LMStudioClient, RAGSystem

def main():
    print("🚀 Enhanced RAG System Example")
    print("=" * 50)
    
    # Initialize the system
    print("\n📦 Initializing components...")
    
    # Phase 1: Data pipeline (works without LM Studio)
    pipeline = RAGDataPipeline()
    
    # Ingest documents from data folder
    print("\n📚 Ingesting documents...")
    pipeline.ingest_documents("./data")
    
    # Phase 2: RAG system (requires LM Studio)
    print("\n🤖 Initializing LLM client...")
    try:
        db_manager = ChromaDBManager()
        llm_client = LMStudioClient()
        rag = RAGSystem(db_manager, llm_client)
        print("\n✅ System initialized successfully!")
    except Exception as e:
        print(f"\n❌ Failed to initialize LLM client: {e}")
        print("\n💡 Make sure LM Studio is running:")
        print("   1. Open LM Studio")
        print("   2. Load a model (e.g., Llama-3.1-8B-Instruct)")
        print("   3. Go to Chat tab and click 'Start Server'")
        print("   4. Run this script again")
        print("\n🧪 Alternatively, test Phase 1 only:")
        print("   python test_without_llm.py")
        return
    
    # Example 1: Simple query with query analysis
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple Query with Analysis")
    print("="*60)
    
    result1 = rag.query(
        "How do I install this?",  # Ambiguous query
        analyze_query=True,
        use_reranking=True,
        verify_citations=True
    )
    
    print(f"Original query: {result1['query']}")
    print(f"Analyzed query: {result1.get('final_question', 'N/A')}")
    if 'query_analysis' in result1:
        analysis = result1['query_analysis']
        print(f"Query type: {analysis.get('query_type', 'unknown')}")
        print(f"Complexity: {analysis.get('complexity', 'unknown')}")
    
    # Example 2: Complex query with decomposition
    print("\n" + "="*60)
    print("EXAMPLE 2: Complex Query Decomposition")
    print("="*60)
    
    complex_query = "What are the installation steps and what payment methods do you accept?"
    result2 = rag.query(
        complex_query,
        analyze_query=True,
        use_reranking=True,
        retrieve_n=10,
        n_results=5,
        verify_citations=True
    )
    
    # Example 3: Structured output
    print("\n" + "="*60)
    print("EXAMPLE 3: Structured JSON Output")
    print("="*60)
    
    result3 = rag.query(
        "What are the system requirements?",
        structured_output=True,
        use_reranking=True,
        verify_citations=True
    )
    
    if 'structured_response' in result3:
        structured = result3['structured_response']
        print(f"Confidence: {structured.get('confidence', 'unknown')}")
        print(f"Information found: {structured.get('information_found', 'unknown')}")
    
    # Example 4: Citation verification report
    print("\n" + "="*60)
    print("EXAMPLE 4: Citation Verification")
    print("="*60)
    
    result4 = rag.query(
        "How can I reset my password?",
        verify_citations=True,
        use_reranking=True
    )
    
    if 'citation_report' in result4:
        print(result4['citation_report'])
    
    # Example 5: Batch testing with different configurations
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Testing")
    print("="*60)
    
    test_queries = [
        "What are the system requirements?",
        "How do I contact support?",
        "Can I get a refund?"
    ]
    
    # Test with all features enabled
    print("\n🧪 Testing with ALL FEATURES:")
    rag.batch_test(
        test_queries,
        n_results=3,
        use_reranking=True,
        analyze_query=True,
        structured_output=True,
        verify_citations=True
    )
    
    print("\n✅ All examples completed!")
    print("\n💡 Try the interactive mode for hands-on testing:")
    print("   rag.interactive_mode()")

if __name__ == "__main__":
    main()
