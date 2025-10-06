# RAG System Setup Guide

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up LM Studio (Required for LLM functionality)

**Download and Install LM Studio:**

1. Go to [https://lmstudio.ai/](https://lmstudio.ai/)
2. Download and install LM Studio for your OS
3. Open LM Studio

**Load a Model:**

1. In LM Studio, go to the "Models" tab
2. Search for and download a model (recommended: `Llama-3.1-8B-Instruct` or similar)
3. Go to the "Chat" tab
4. Click "Start Server" (this starts the local API server)
5. Note the server URL (default: `http://localhost:1234`)

**Alternative Models:**

- `Llama-3.1-8B-Instruct` (recommended, good balance)
- `Llama-3.1-7B-Instruct` (smaller, faster)
- `Mistral-7B-Instruct` (alternative option)

### 3. Verify LM Studio Connection

The system will automatically test the connection when you run it. You should see:

```
✓ Connected to LM Studio
  Active model: llama-3.1-8b-instruct
```

If you see connection errors, make sure:

- LM Studio is running
- Server is started (click "Start Server" in LM Studio)
- No firewall is blocking port 1234

## Running the System

### Option 1: Run the Example

```bash
python example_usage.py
```

### Option 2: Run Individual Components

```python
# Test Phase 1 (Data ingestion) - works without LM Studio
python rag_phase1.py

# Test Phase 2 (Full RAG system) - requires LM Studio
python rag_phase2.py
```

### Option 3: Interactive Mode

```python
from rag_phase1 import RAGDataPipeline, ChromaDBManager
from rag_phase2 import LMStudioClient, RAGSystem

# Initialize
pipeline = RAGDataPipeline()
pipeline.ingest_documents("./sample_docs")

db_manager = ChromaDBManager()
llm_client = LMStudioClient()
rag = RAGSystem(db_manager, llm_client)

# Start interactive mode
rag.interactive_mode()
```

## Troubleshooting

### LM Studio Connection Issues

- **Error**: "Could not connect to LM Studio"

  - **Solution**: Make sure LM Studio is running and server is started
  - **Check**: Go to LM Studio → Chat → Start Server

- **Error**: "No models loaded"
  - **Solution**: Download a model in LM Studio → Models tab

### Memory Issues

- **Error**: "CUDA out of memory" or similar
  - **Solution**: Use a smaller model or reduce batch sizes
  - **Alternative**: Use CPU-only mode in LM Studio

### Dependencies Issues

- **Error**: "Module not found"
  - **Solution**: Run `pip install -r requirements.txt`

### Sample Documents Not Found

- **Solution**: The system automatically creates sample documents if they don't exist
- **Location**: `./sample_docs/` directory

## Performance Tips

### For Better Performance:

1. **Use GPU**: Enable GPU acceleration in LM Studio
2. **Larger Models**: Use 8B+ models for better quality
3. **Batch Processing**: Process multiple queries together

### For Faster Testing:

1. **Smaller Models**: Use 7B models for faster responses
2. **Reduce Chunks**: Lower `n_results` parameter
3. **Disable Features**: Turn off reranking/compression for speed

## System Requirements

### Minimum:

- 8GB RAM
- 4GB free disk space
- Internet connection (for model download)

### Recommended:

- 16GB RAM
- 8GB+ VRAM (for GPU acceleration)
- 10GB free disk space

## Model Recommendations

### For Development/Testing:

- **Llama-3.1-7B-Instruct**: Fast, good quality
- **Mistral-7B-Instruct**: Alternative option

### For Production:

- **Llama-3.1-8B-Instruct**: Better quality
- **Llama-3.1-70B-Instruct**: Best quality (requires more resources)

## Next Steps

1. **Start Simple**: Run with default settings first
2. **Test Features**: Try different combinations of reranking, compression, etc.
3. **Add Your Data**: Replace sample documents with your own
4. **Customize**: Adjust chunk sizes, embedding models, etc.
5. **Deploy**: Consider using cloud APIs for production use
