#!/usr/bin/env python3
"""
Test script for Hermes model setup
Verifies connection and model detection
"""

import requests
from rag_phase2 import LMStudioClient

def test_hermes_connection():
    """Test connection to Hermes model"""
    print("🤖 Testing Hermes Model Setup")
    print("=" * 40)
    
    # Test basic connection
    print("1. Testing basic connection...")
    try:
        response = requests.get("http://127.0.0.1:1234/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ Connected to LM Studio!")
            
            models = response.json()
            if models.get('data'):
                model_info = models['data'][0]
                model_name = model_info['id']
                print(f"📋 Model Name: {model_name}")
                print(f"📋 Model ID: {model_info.get('id', 'N/A')}")
                
                # Check if it matches expected Hermes model
                if "hermes" in model_name.lower() and "llama-3.2" in model_name.lower():
                    print("🎯 Perfect! Hermes-3-Llama-3.2 model detected!")
                elif "hermes" in model_name.lower():
                    print("🎯 Hermes model detected (version may vary)")
                elif "llama-3.2" in model_name.lower():
                    print("🎯 Llama-3.2 model detected")
                else:
                    print(f"⚠️  Different model detected: {model_name}")
                    print("   Expected: hermes-3-llama-3.2-3b")
                
                return True
            else:
                print("❌ No models loaded in LM Studio")
                return False
        else:
            print(f"❌ LM Studio responded with status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to LM Studio: {e}")
        print("💡 Make sure:")
        print("   - LM Studio is running")
        print("   - Server is started on http://127.0.0.1:1234")
        print("   - Hermes model is loaded")
        return False

def test_hermes_generation():
    """Test text generation with Hermes"""
    print("\n2. Testing text generation...")
    
    try:
        llm_client = LMStudioClient(base_url="http://127.0.0.1:1234/v1")
        
        # Test simple generation
        test_prompts = [
            "Hello, how are you?",
            "What is machine learning?",
            "Explain the concept of attention in neural networks."
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n   Test {i}: {prompt}")
            try:
                response = llm_client.generate(prompt, temperature=0.1, max_tokens=100)
                print(f"   ✅ Response: {response[:80]}...")
            except Exception as e:
                print(f"   ❌ Generation failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ LMStudioClient initialization failed: {e}")
        return False

def test_rag_integration():
    """Test RAG integration with Hermes"""
    print("\n3. Testing RAG integration...")
    
    try:
        from rag_phase1 import RAGDataPipeline, ChromaDBManager
        from rag_phase2 import RAGSystem
        
        # Initialize with sample data
        print("   📚 Setting up sample documents...")
        pipeline = RAGDataPipeline(chroma_db_path="./test_chroma_db")
        pipeline.ingest_documents("./sample_docs")
        
        # Initialize RAG system
        print("   🤖 Initializing RAG system...")
        db_manager = ChromaDBManager(persist_directory="./test_chroma_db")
        llm_client = LMStudioClient(base_url="http://127.0.0.1:1234/v1")
        rag = RAGSystem(db_manager, llm_client)
        
        # Test a simple query
        print("   🔍 Testing RAG query...")
        result = rag.query(
            "How do I install the product?",
            n_results=2,
            use_reranking=True,
            analyze_query=True
        )
        
        print(f"   ✅ RAG Response: {result['answer'][:100]}...")
        print(f"   📚 Citations: {result.get('citations', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG integration failed: {e}")
        return False

def main():
    print("🧪 Hermes Model Setup Test")
    print("=" * 50)
    
    # Test 1: Basic connection
    connection_ok = test_hermes_connection()
    
    if not connection_ok:
        print("\n❌ Connection test failed. Please check LM Studio setup.")
        return
    
    # Test 2: Text generation
    generation_ok = test_hermes_generation()
    
    # Test 3: RAG integration
    rag_ok = test_rag_integration()
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    print(f"✅ Connection: {'PASS' if connection_ok else 'FAIL'}")
    print(f"✅ Generation: {'PASS' if generation_ok else 'FAIL'}")
    print(f"✅ RAG Integration: {'PASS' if rag_ok else 'FAIL'}")
    
    if connection_ok and generation_ok and rag_ok:
        print("\n🎉 All tests passed! Your Hermes setup is ready!")
        print("\n🚀 Next steps:")
        print("   - Run: python test_pdf_paper.py")
        print("   - Provide path to your PDF paper")
        print("   - Start asking questions about your paper!")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
