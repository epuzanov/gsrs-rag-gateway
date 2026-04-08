# Ollama + Open WebUI Integration Guide with ERI Support

This guide shows you how to set up a **local** LLM environment with Ollama and Open WebUI, integrated with the GSRS RAG Gateway using the **ERI (External Retrieval Interface)** for answering questions about chemical substances.

## Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│   User      │────▶│  Open WebUI  │────▶│  RAG Gateway │────▶│  pgvector   │
│  Browser    │     │   (Frontend) │     │   (API)      │     │  Database   │
└─────────────┘     └───────┬──────┘     └─────────────┘     └─────────────┘
                            │                                     ▲
                            │ ERI Query                           │
                            ▼                                     │
                     ┌──────────────┐     ┌─────────────┐         │
                     │    Ollama    │────▶│  ERI Tool   │─────────┘
                     │  (Qwen3.5)   │     │  (Function) │
                     └──────────────┘     └─────────────┘
```

**Benefits:**
- ✅ **100% Local** - No API costs, full privacy
- ✅ **Qwen3.5 Models** - Latest powerful open-source LLMs from Alibaba
- ✅ **ERI Support** - Open WebUI native retrieval interface
- ✅ **Web Interface** - User-friendly chat interface
- ✅ **Multiple Models** - Easy model switching
- ✅ **RAG Enhanced** - Accurate answers from GSRS database

## Prerequisites

1. **Docker** and **Docker Compose**
2. **Ollama** installed locally
3. **GSRS RAG Gateway** repository

## Step 1: Install Ollama

### Linux

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### macOS

```bash
brew install ollama
```

### Windows

Download from [ollama.ai](https://ollama.ai)

### Verify Installation

```bash
ollama --version
```

## Step 2: Pull Qwen3.5 Models

```bash
# Qwen3.5 Chat model for answering questions (recommended: 7B or 14B)
ollama pull qwen3.5:7b

# Qwen3.5 Embedding model for RAG (recommended)
ollama pull qwen3-embedding:latest

# Alternative embedding models
# ollama pull nomic-embed-text
# ollama pull mxbai-embed-large

# Verify models
ollama list
```

**Recommended Qwen3.5 Models:**

| Purpose | Model | Size | Dimension | Command |
|---------|-------|------|-----------|---------|
| Chat | `qwen3.5:7b` | 7B | - | `ollama pull qwen3.5:7b` |
| Chat | `qwen3.5:14b` | 14B | - | `ollama pull qwen3.5:14b` |
| Chat | `qwen3.5:32b` | 32B | - | `ollama pull qwen3.5:32b` |
| Chat | `qwen3.5-coder:7b` | 7B | - | `ollama pull qwen3.5-coder:7b` |
| Embedding | `qwen3-embedding:latest` | ~7B | 1024 | `ollama pull qwen3-embedding` |
| Embedding | `nomic-embed-text` | 270MB | 768 | `ollama pull nomic-embed-text` |
| Embedding | `mxbai-embed-large` | 670MB | 1024 | `ollama pull mxbai-embed-large` |

### Using Qwen3.5 Embedding Model

**Yes, you can use `qwen3-embedding` as the embedding model!** This is the recommended embedding model for Qwen3.5 setups.

**Configuration for Qwen3.5 Embedding:**

```bash
# In your .env file:
EMBEDDING_API_KEY=ollama
EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1
EMBEDDING_MODEL=qwen3-embedding:latest
EMBEDDING_DIMENSION=1024
```

**Benefits of Qwen3.5 Embedding:**
- **Higher dimension** (1024 vs 768) - Better semantic representation
- **Multi-language support** - Optimized for 100+ languages
- **Better alignment** with Qwen3.5 chat models
- **State-of-the-art** performance on MTEB benchmark

**Note:** Qwen3.5 embedding model is larger (~7B parameters) and requires more VRAM (~14 GB) compared to nomic-embed-text (~270MB). For resource-constrained systems, consider using `nomic-embed-text` instead.

### Model Selection Guide

| Use Case | Chat Model | Embedding Model | Total VRAM |
|----------|------------|-----------------|------------|
| Low VRAM | `qwen3.5:3b` | `nomic-embed-text` | ~4 GB |
| Balanced | `qwen3.5:7b` | `nomic-embed-text` | ~8 GB |
| Best Quality | `qwen3.5:14b` | `qwen3-embedding` | ~24 GB |
| Maximum | `qwen3.5:32b` | `qwen3-embedding` | ~34 GB |

## Step 3: Configure the RAG Gateway

Create or edit `.env` file:

```bash
# Use Qwen embeddings via Ollama
EMBEDDING_API_KEY=ollama
EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

# Database (ChromaDB for local development)
DATABASE_URL=chroma://./chroma_data/chunks

# Authentication (for ERI)
API_USERNAME=admin
API_PASSWORD=admin123
```

## Step 4: Start All Services

```bash
# Navigate to project
cd gsrs-rag-gateway

# Start RAG Gateway with Open WebUI
docker-compose --profile ollama up -d

# Verify services
docker-compose ps
```

**Services started:**
- `gsrs-postgres` - PostgreSQL with pgvector (or ChromaDB for local)
- `gsrs-rag-gateway` - RAG Gateway API
- `gsrs-open-webui` - Open WebUI interface

## Step 5: Load Substance Data

### Option A: Load Sample Substances from GSRS Server (Recommended)

```bash
# Load specific substances by UUID (fast, recommended for testing)
python scripts/load_data.py \
  --uuids 0103a288-6eb6-4ced-b13a-849cd7edf028,80edf0eb-b6c5-4a9a-adde-28c7254046d9

# Verify data is loaded
curl -u admin:admin123 http://localhost:8000/statistics
```

### Option B: Load All Substances from GSRS Server

```bash
# Load first 100 substances (may take 10-30 minutes)
python scripts/load_data.py --all --max-results 100
```

### Option C: Load from .gsrs File

If you have substance data in `.gsrs` format:

```bash
python scripts/load_data.py data/substances.gsrs --batch-size 100
```

## Step 6: Configure Open WebUI with ERI

Open WebUI supports external tools that can be used to query the GSRS RAG Gateway. The recommended approach is to use a custom Python tool.

### Option A: Custom Python Tool (Recommended)

This approach uses Open WebUI's native tool system with a Python script.

#### 1. Create Tool File

The tool file is provided in the examples directory: `examples/gsrs_openwebui_tool.py`

**Tool Features:**
- Queries the GSRS RAG Gateway `/eri/query` endpoint
- Returns formatted search results
- Configurable via valves (base URL, credentials, top_k)
- Error handling for timeouts and connection issues


#### 2. Mount Tool File in Docker

Update `docker-compose.yaml` to mount the tool file:

```yaml
services:
  open-webui:
    volumes:
      - ./examples/gsrs_tool.py:/app/backend/tools/gsrs_tool.py
```

#### 3. Restart Open WebUI

```bash
docker-compose restart open-webui
```

#### 4. Enable Tool in Open WebUI

1. Open **http://localhost:3000**
2. Go to **Workspace** → **Tools**
3. Find **GSRS RAG Gateway Tool** in the list
4. Click to enable it
5. Configure valves if needed:
   - **rag_base_url**: `http://gsrs-rag-gateway:8000`
   - **api_username**: `admin`
   - **api_password**: `admin123`
   - **top_k**: `5`

#### 5. Use Tool in Chat

1. Start a new chat with Qwen model
2. The GSRS tool will be automatically available
3. Ask questions like:
   - "What is the CAS code for Aspirin?"
   - "Show me the molecular formula of Ibuprofen"
   - "What is the UNII code for Paracetamol?"

### Option B: Custom Function via Admin Panel (Advanced)

For more control, you can create a custom function through Open WebUI's Admin Panel.

**Note:** This requires admin access to Open WebUI.

#### 1. Access Admin Panel

1. Open **http://localhost:3000**
2. Click your profile icon → **Admin Panel** (or **Admin Settings**)
3. Go to **Functions** tab

#### 2. Create Custom Function

1. Click **Create Function** or **Add New**
2. Configure the function:
   - **Name**: `GSRS RAG Query`

3. Add the Python code from file:
`./examples/gsrs_rag_function.py`

4. Click **Save**

#### 3. Enable Function

1. In Admin Panel → Functions, find `GSRS RAG Query`
2. Toggle **Enabled** to ON
3. Optionally, assign to specific models

### Option B2: Alternative - Mount Function via Docker

If the Admin Panel doesn't show Functions tab, you can mount function files directly:

#### 1. Mount Tool File in Docker

Update `docker-compose.yaml` to mount the tool file:

```yaml
services:
  open-webui:
    volumes:
      - ./examples/gsrs_function.py:/app/backend/functions/gsrs_function.py
```

### Option C: Direct ERI API Usage

You can also use the ERI endpoint directly without tools, but using the Python tool (Option A) is recommended for better integration.

#### ERI Endpoint

```bash
POST /eri/query
Content-Type: application/json
Authorization: Basic admin:admin123
```

**Request:**
```json
{
    "query": "CAS code for Aspirin",
    "top_k": 5
}
```

**Response:**
```json
{
    "results": [
        {
            "id": "codes_cas_50-78-2",
            "text": "Code: 50-78-2 | System: CAS | Type: PRIMARY",
            "score": 0.92,
            "metadata": {
                "element_path": "codes_cas_50-78-2",
                "substance_uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028"
            }
        }
    ]
}
```

#### Test ERI Endpoint

```bash
curl -X POST http://localhost:8000/eri/query \
  -H "Content-Type: application/json" \
  -u admin:admin123 \
  -d '{"query": "CAS code for Aspirin", "top_k": 3}'
```

## Step 7: Ask Questions with ERI

### Using Open WebUI Interface

1. Open **http://localhost:3000**
2. Select model: `qwen3.5:7b` (or your preferred Qwen model)
3. Enable the GSRS ERI tool (toggle in chat input)
4. Type your question:

```
What is the CAS code for Aspirin?
```

5. The LLM will automatically use the ERI tool to query the GSRS database
6. Get AI-powered answer with citations!

### Example Questions

```
What is the molecular formula of Ibuprofen?
Show me all codes for this substance
What is the UNII code for Paracetamol?
List all substances with CHEMBL identifiers
What is the molecular weight of Caffeine?
Find proteins related to heart failure treatment
```

### Example Qwen Response

When you ask "What is the CAS code for Aspirin?", Qwen will use the ERI tool and respond:

> The CAS code for Aspirin is **50-78-2** [Source: codes_cas_50-78-2].
>
> Aspirin (also known as Acetylsalicylic acid) has the following identifiers:
> - **UNII**: WK2XYI10QM
> - **ChEMBL**: CHEMBL521
> - **ATC Code**: N02BA01

### Qwen-Specific Prompting Tips

For best results with Qwen models:

1. **Be specific**: Qwen excels at precise technical questions
2. **Use structured queries**: Qwen understands structured data well
3. **Multi-language**: Qwen supports queries in multiple languages

```
# English
What is the molecular weight of Ibuprofen?

# Chinese (Qwen supports Chinese natively)
布洛芬的分子量是多少？

# German
Was ist das Molekulargewicht von Ibuprofen?
```

### Example ERI Response

When you ask "What is the CAS code for Aspirin?", the ERI tool returns:

```json
{
    "query": "What is the CAS code for Aspirin?",
    "results": [
        {
            "element_path": "codes_cas_50-78-2",
            "substance_uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028",
            "chunk_text": "Code: 50-78-2 | System: CAS | Type: PRIMARY",
            "similarity_score": 0.92,
            "metadata": {
                "code_system": "CAS",
                "code_value": "50-78-2"
            }
        }
    ],
    "total_results": 1
}
```

The LLM then formats this into a natural language response as shown above.

## ERI Endpoint Implementation

The GSRS RAG Gateway provides a native ERI endpoint at `/eri/query`.

### ERI Endpoint

```bash
POST /eri/query
Content-Type: application/json
Authorization: Basic admin:admin123
```

**Request:**
```json
{
    "query": "CAS identifier for Ibuprofen",
    "top_k": 5,
    "filters": {}
}
```

**Response:**
```json
{
    "results": [
        {
            "id": "codes",
            "text": "Ibuprofen identifier in CAS: 139466-08-3.",
            "metadata": {
                "source_url": "https://gsrs.ncats.nih.gov/api/v1/substances(0103a288-6eb6-4ced-b13a-849cd7edf028)?view=full",
                "substance_uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028",
                "similarity_score": 0.92
            }
        }
    ]
}
```

### Using ERI Directly

```bash
curl -X POST http://localhost:8000/eri/query \
  -H "Content-Type: application/json" \
  -u admin:admin123 \
  -d '{"query": "CAS code for Aspirin", "top_k": 3}'
```

## Advanced ERI Configuration

### ERI with Filters

```json
{
    "query": "molecular weight",
    "top_k": 5,
    "filters": {
        "metadata": {
            "chunk_type": "property"
        }
    }
}
```

### ERI with Substance Class Filter

```bash
curl -X POST http://localhost:8000/eri/query \
  -u admin:admin123 \
  -H "Content-Type: application/json" \
  -d '{
    "query": "protein structure",
    "top_k": 5
  }'
```

### ERI Response Transformation

If you need to transform the response for compatibility with other systems:

```python
def transform_eri_response(response: dict) -> dict:
    """Transform GSRS response to standard ERI format."""
    return {
        "results": [
            {
                "id": r["element_path"],
                "text": r["chunk_text"],
                "score": r["similarity_score"],
                "metadata": {
                    "substance_uuid": r["substance_uuid"],
                    **r.get("metadata", {})
                }
            }
            for r in response.get("results", [])
        ]
    }
```

## Troubleshooting

### ERI Tool Not Appearing

```bash
# Check if tool is loaded
curl http://localhost:3000/api/tools

# Restart Open WebUI
docker-compose restart open-webui
```

### ERI Query Returns No Results

```bash
# Check if data is loaded
curl -u admin:admin123 http://localhost:8000/statistics

# Test ERI endpoint directly
curl -u admin:admin123 -X POST http://localhost:8000/eri/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 1}'
```

### Authentication Errors

```bash
# Verify credentials
curl -u admin:admin123 http://localhost:8000/health

# Change password in .env
API_USERNAME=your-username
API_PASSWORD=your-password
```

### Network Issues Between Containers

```bash
# Test connectivity from Open WebUI container
docker-compose exec open-webui wget -qO- http://gsrs-rag-gateway:8000/health

# Check Docker network
docker-compose ps
docker network ls
```

## Performance Tips

### 1. Optimize ERI Query Parameters

```json
{
    "query": "aspirin",
    "top_k": 3
}
```

### 2. Use Embedding Caching

The RAG Gateway caches embeddings for repeated queries.

### 3. Batch Multiple Queries

For complex questions, the LLM may make multiple ERI calls. This is normal.

## Complete Setup Script

Save as `setup-eri.sh`:

```bash
#!/bin/bash
set -e

echo "=== GSRS RAG Gateway - ERI Setup (Qwen3.5) ==="

# 1. Install Ollama (Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# 2. Pull Qwen models
echo "Pulling Qwen models..."
ollama pull qwen3.5:7b

# Check for sufficient VRAM for qwen3-embedding
AVAILABLE_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo "0")
if [ "$AVAILABLE_VRAM" -gt 20000 ]; then
    echo "Sufficient VRAM detected (${AVAILABLE_VRAM}MB). Pulling qwen3-embedding..."
    ollama pull qwen3-embedding:latest
    EMBEDDING_MODEL="qwen3-embedding:latest"
    EMBEDDING_DIMENSION=1024
else
    echo "Limited VRAM detected. Using nomic-embed-text instead..."
    ollama pull nomic-embed-text
    EMBEDDING_MODEL="nomic-embed-text"
    EMBEDDING_DIMENSION=768
fi

# 3. Create .env
echo "Creating .env file..."
cat > .env << EOF
DATABASE_URL=chroma://./chroma_data/chunks
EMBEDDING_API_KEY=ollama
EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1
EMBEDDING_MODEL=${EMBEDDING_MODEL}
EMBEDDING_DIMENSION=${EMBEDDING_DIMENSION}
API_USERNAME=admin
API_PASSWORD=admin123
EOF

# 4. Start services
echo "Starting Docker services..."
docker-compose --profile ollama up -d

# 5. Wait for services
echo "Waiting for services to start..."
sleep 10

# 6. Load sample substances from GSRS server
echo "Loading sample substances from GSRS server..."
python scripts/load_data.py \
  --uuids 0103a288-6eb6-4ced-b13a-849cd7edf028,80edf0eb-b6c5-4a9a-adde-28c7254046d9

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Open WebUI: http://localhost:3000"
echo "ERI Endpoint: http://localhost:8000/eri/query"
echo "Swagger UI: http://localhost:8000/docs"
echo ""
echo "Example: Ask 'What is the CAS code for Aspirin?' in Open WebUI"
echo "Model: qwen3.5:7b"
```

Make executable and run:
```bash
chmod +x setup-eri.sh
./setup-eri.sh
```

## Next Steps

- [ChatGPT Integration](chatgpt.md) - Cloud-based alternative
- [API Reference](../api-reference.md) - Complete API documentation
- [Chunking Guide](chunking.md) - Understand how substances are chunked
- [Data Loading](../data-loading.md) - Load your own substance data

## Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Open WebUI Documentation](https://docs.openwebui.com)
- [Open WebUI ERI Guide](https://docs.openwebui.com/features/external-retrieval/)
- [GSRS RAG Gateway API](http://localhost:8000/docs)
