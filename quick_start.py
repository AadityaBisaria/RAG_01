#!/usr/bin/env python3
"""
Quick Start Script for RAG System
This script guides you through setup and testing
"""

import sys
import subprocess
import requests

def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'chromadb',
        'sentence_transformers', 
        'langchain',
        'PyPDF2',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed!")
    return True

def check_lm_studio():
    """Check if LM Studio is running"""
    print("\n🤖 Checking LM Studio connection...")
    
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            if models.get('data'):
                model_name = models['data'][0]['id']
                print(f"✅ LM Studio connected! Active model: {model_name}")
                return True
            else:
                print("⚠️  LM Studio connected but no model loaded")
                print("💡 Load a model in LM Studio and try again")
                return False
        else:
            print("❌ LM Studio not responding")
            return False
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to LM Studio")
        return False

def run_phase1_test():
    """Run Phase 1 test (no LLM required)"""
    print("\n🧪 Running Phase 1 test...")
    try:
        import test_without_llm
        test_without_llm.test_phase1_only()
        return True
    except Exception as e:
        print(f"❌ Phase 1 test failed: {e}")
        return False

def run_full_example():
    """Run the full example"""
    print("\n🚀 Running full RAG example...")
    try:
        import example_usage
        example_usage.main()
        return True
    except Exception as e:
        print(f"❌ Full example failed: {e}")
        return False

def main():
    print("🎯 RAG System Quick Start")
    print("=" * 40)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install dependencies first:")
        print("   pip install -r requirements.txt")
        return
    
    # Step 2: Test Phase 1 (no LLM required)
    print("\n" + "="*40)
    print("PHASE 1 TEST (No LLM Required)")
    print("="*40)
    
    if not run_phase1_test():
        print("❌ Phase 1 test failed. Check the errors above.")
        return
    
    # Step 3: Check LM Studio
    print("\n" + "="*40)
    print("LM STUDIO SETUP CHECK")
    print("="*40)
    
    lm_studio_ready = check_lm_studio()
    
    if not lm_studio_ready:
        print("\n📋 LM Studio Setup Instructions:")
        print("1. Download LM Studio from: https://lmstudio.ai/")
        print("2. Install and open LM Studio")
        print("3. Go to 'Models' tab and download a model (e.g., Llama-3.1-8B-Instruct)")
        print("4. Go to 'Chat' tab and click 'Start Server'")
        print("5. Run this script again")
        
        print("\n🧪 You can still test Phase 1 functionality:")
        print("   python test_without_llm.py")
        return
    
    # Step 4: Run full example
    print("\n" + "="*40)
    print("FULL RAG SYSTEM TEST")
    print("="*40)
    
    if run_full_example():
        print("\n🎉 Success! Your RAG system is working!")
        print("\n💡 Next steps:")
        print("   - Add your own documents to ./sample_docs/")
        print("   - Try the interactive mode: rag.interactive_mode()")
        print("   - Experiment with different models in LM Studio")
    else:
        print("\n❌ Full example failed. Check the errors above.")

if __name__ == "__main__":
    main()
