#!/usr/bin/env python3
"""
Quick script to check which model is loaded in LM Studio
"""

import requests
import json

def check_model():
    """Check which model is currently loaded in LM Studio"""
    print("🔍 Checking LM Studio Model")
    print("=" * 30)
    
    try:
        # Get model information
        response = requests.get("http://127.0.0.1:1234/v1/models", timeout=5)
        
        if response.status_code == 200:
            models = response.json()
            
            print("✅ Connected to LM Studio!")
            print(f"📊 Response: {json.dumps(models, indent=2)}")
            
            if models.get('data') and len(models['data']) > 0:
                model = models['data'][0]
                model_name = model.get('id', 'Unknown')
                
                print(f"\n📋 Current Model:")
                print(f"   Name: {model_name}")
                print(f"   Object: {model.get('object', 'N/A')}")
                print(f"   Created: {model.get('created', 'N/A')}")
                print(f"   Owned By: {model.get('owned_by', 'N/A')}")
                
                # Check if it's the expected Hermes model
                if "hermes" in model_name.lower() and "llama-3.2" in model_name.lower():
                    print(f"\n🎯 Perfect! This matches your expected model:")
                    print(f"   Expected: hermes-3-llama-3.2-3b")
                    print(f"   Found: {model_name}")
                else:
                    print(f"\n⚠️  Model doesn't match expected name:")
                    print(f"   Expected: hermes-3-llama-3.2-3b")
                    print(f"   Found: {model_name}")
                
            else:
                print("❌ No models loaded in LM Studio")
                print("💡 Load a model in LM Studio and try again")
                
        else:
            print(f"❌ LM Studio responded with status {response.status_code}")
            print("💡 Make sure LM Studio server is running")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to LM Studio")
        print("💡 Make sure:")
        print("   - LM Studio is running")
        print("   - Server is started on http://127.0.0.1:1234")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_model()
