# GSRS RAG Gateway

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/epuzanov/gsrs-rag-gateway/actions/workflows/tests.yml/badge.svg)](https://github.com/epuzanov/gsrs-rag-gateway/actions)

Retrieval-Augmented Generation (RAG) Gateway für GSRS (Global Substance Registration System) Substanzen mit **pgvector** oder **ChromaDB** als Vektordatenbank.

## Features

- 🧩 **Intelligentes Chunking**: Automatische Aufteilung von GSRS Substance JSON Dokumenten in elementbasierte Chunks
- 🔍 **Vektorsuche**: Semantische Suche mit pgvector (Production) oder ChromaDB (Development)
- 🎯 **Element Path IDs**: Eindeutige IDs für Chunks basierend auf Element-Pfaden
- 📊 **Metadata-Retention**: Vollständige Metadaten für jedes Element in Embeddings
- 🔄 **SubstanceClass Filter**: Filterung nach Substanztyp (chemical, protein, nucleicAcid, etc.)
- 🎨 **Embedding Provider**: OpenAI API, Azure OpenAI, Ollama und OpenAI-kompatible APIs
- 🔐 **Authentication**: HTTP Basic Auth und API Key Unterstützung
- 🗄️ **Multi-Backend**: pgvector (PostgreSQL) oder ChromaDB (lokal, serverless)
- 🐳 **Docker Deployment**: Einfache Bereitstellung mit Docker Compose
- 📥 **Bulk-Loading**: Ladeskript für JSONL-Dateien
- ✅ **Unit Tests**: Vollständige Testabdeckung mit pytest

## Architektur

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   GSRS JSON     │────▶│  ChunkerService  │────▶│ EmbeddingService│
│   (Substance)   │     │  (gsrs.model)    │     │ (OpenAI/Ollama) │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Vector Database Backend                          │
│  ┌─────────────────┐                   ┌─────────────────┐          │
│  │   pgvector      │  (Production)     │    ChromaDB     │          │
│  │  (PostgreSQL)   │◀─────────────────▶│  (Development)  │          │
│  └─────────────────┘                   └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                            │
│  ┌─────────────────┐                   ┌─────────────────┐          │
│  │  /ingest        │                   │   /query        │          │
│  │  /ingest/batch  │◀───────┬─────────▶│   /statistics   │          │
│  │  /substances/*  │        │          │   /health       │          │
│  └─────────────────┘        │          └─────────────────┘          │
│                             │                                        │
│  ┌─────────────────┐        │                                        │
│  │  Auth Layer     │◀───────┘                                        │
│  │  (Basic/API Key)│                                                 │
│  └─────────────────┘                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Schnellstart

### Option 1: Production mit PostgreSQL + pgvector (Docker)

```bash
# .env Datei erstellen und konfigurieren
cp .env.example .env
# Bearbeite .env und setze:
#   DATABASE_URL=postgresql://gsrs:your_password@postgres:5432/gsrs_rag
#   EMBEDDING_API_KEY=sk-your-key

# Alle Services starten (PostgreSQL + RAG Gateway)
docker-compose --profile postgres up -d

# Mit Open WebUI (für Ollama Integration)
docker-compose --profile postgres --profile ollama up -d
```

### Option 2: Development mit ChromaDB (Docker)

```bash
# .env Datei erstellen
cp .env.example .env
# DATABASE_URL ist bereits auf ChromaDB vorkonfiguriert

# ChromaDB + RAG Gateway starten
docker-compose --profile chroma up -d
```

### Option 3: Local Development (ohne Docker)

```bash
# Virtuelle Umgebung erstellen
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder: venv\Scripts\activate  # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt

# .env Datei erstellen
cp .env.example .env

# EMBEDDING_API_KEY setzen
export EMBEDDING_API_KEY="sk-your-key"

# Gateway starten
uvicorn app.main:app --reload
```

### Authentifizierung

```bash
# Standard Admin Benutzer
# Username: admin
# Password: admin123 (oder API_PASSWORD in .env ändern)

# API mit Authentifizierung nutzen
curl -u admin:admin123 http://localhost:8000/health
```

### Gesundheit prüfen

```bash
curl http://localhost:8000/health
```

### Beispieldaten laden

```bash
# Substanzen von GSRS Server laden
python scripts/load_data.py \
  --uuids 0103a288-6eb6-4ced-b13a-849cd7edf028,80edf0eb-b6c5-4a9a-adde-28c7254046d9

# Statistiken prüfen
curl -u admin:admin123 http://localhost:8000/statistics
```

## API Endpoints

### Health Check

```bash
GET /health
```

### Substance ingestieren

```bash
POST /ingest
Content-Type: application/json
Authorization: Basic YWRtaW46YWRtaW4xMjM=  # admin:admin123

{
    "substance": { /* GSRS Substance JSON */ }
}
```

### Batch Ingest

```bash
POST /ingest/batch
Content-Type: application/json

{
    "substances": [ /* Array of GSRS Substance JSON */ ]
}
```

### Semantische Suche

```bash
POST /query
Content-Type: application/json

{
    "query": "CAS code for Aspirin",
    "top_k": 5,
    "filters": {}  // optionale Metadata-Filter
}
```

### Substance löschen

```bash
DELETE /substances/{substance_uuid}
Authorization: Basic YWRtaW46YWRtaW4xMjM=
```

### Verfügbare Embedding Modelle

```bash
GET /models
```

### Substance Klassen

```bash
GET /substance-classes
```

### Statistiken

```bash
GET /statistics
```

## Konfiguration

Umgebungsvariablen (`.env` Datei):

```bash
# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
# Database URL - Schema bestimmt Backend automatisch:
# - PostgreSQL: postgresql://user:pass@host:port/dbname
# - ChromaDB: chroma://./chroma_data/substance_chunks

# Für ChromaDB (Development/Testing - Default):
DATABASE_URL=chroma://./chroma_data/substance_chunks

# Für PostgreSQL (Production - uncomment):
# DATABASE_URL=postgresql://gsrs:your_secure_password@localhost:5432/gsrs_rag

# =============================================================================
# EMBEDDING API CONFIGURATION
# =============================================================================
# Funktioniert mit OpenAI, Azure OpenAI, Ollama und OpenAI-kompatiblen APIs

# OpenAI (Production):
EMBEDDING_API_KEY=sk-your-api-key-here
EMBEDDING_BASE_URL=https://api.openai.com/v1

# Azure OpenAI (uncomment für Azure):
# EMBEDDING_API_KEY=your-azure-key
# EMBEDDING_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment

# Ollama (Local/Development - uncomment für lokale Embeddings):
# EMBEDDING_API_KEY=ollama
# EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1

# =============================================================================
# EMBEDDING MODEL CONFIGURATION
# =============================================================================
# OpenAI Modelle:
#   - text-embedding-3-small (1536 dim, empfohlen)
#   - text-embedding-3-large (3072 dim, höchste Qualität)
#   - text-embedding-ada-002 (1536 dim, legacy)

# Ollama Modelle:
#   - nomic-embed-text (768 dim, leichtgewichtig)
#   - mxbai-embed-large (1024 dim, hohe Qualität)

EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# Für Ollama (uncomment für lokale Embeddings):
# EMBEDDING_MODEL=nomic-embed-text
# EMBEDDING_DIMENSION=768

# =============================================================================
# AUTHENTICATION CONFIGURATION (HTTP Basic Auth)
# =============================================================================
# In Production ändern!
API_USERNAME=admin
API_PASSWORD=admin123

# =============================================================================
# API CONFIGURATION
# =============================================================================
API_HOST=0.0.0.0
API_PORT=8000
DEFAULT_TOP_K=5
```

### Embedding Provider

#### OpenAI (und kompatible APIs)

```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
```

**Unterstützte Modelle:**
- `text-embedding-3-small` (1536 dim) - Schnell und effizient
- `text-embedding-3-large` (3072 dim) - Höchste Qualität
- `text-embedding-ada-002` (1536 dim) - Legacy

**Azure OpenAI:**
```bash
EMBEDDING_API_KEY=your-azure-key
EMBEDDING_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
```

#### Ollama (Lokale Modelle)

```bash
EMBEDDING_API_KEY=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768
EMBEDDING_BASE_URL=http://localhost:11434/v1
```

**Unterstützte Modelle:**
- `nomic-embed-text` (768 dim)
- `mxbai-embed-large` (1024 dim)
- `all-minilm` (384 dim)
- Und alle anderen Ollama Embedding Modelle

## Daten laden

### Aus JSON Dateien

```bash
curl -X POST http://localhost:8000/ingest \
    -H "Content-Type: application/json" \
    -u admin:admin123 \
    -d @substance.json
```

### Aus .gsrs Dateien (JSONL.gz)

```bash
python scripts/load_data.py data/substances.gsrs --batch-size 100
```

### Sample Daten herunterladen

```bash
python scripts/download_samples.py
```

## Integration mit LLMs

### ChatGPT Integration

```bash
# Zusätzliche Abhängigkeiten installieren
pip install -r requirements-examples.txt

# API Key setzen
export OPENAI_API_KEY="sk-..."

# Frage stellen
python examples/chatgpt_integration.py "What is the molecular formula of Aspirin?"
```

### Ollama Integration (Lokal)

```bash
# Ollama installieren: https://ollama.ai
ollama pull llama3.1

# Embedding Modell für RAG (wird im Gateway verwendet)
ollama pull nomic-embed-text

# Frage stellen
python examples/ollama_integration.py "What is the CAS code for Ibuprofen?"
```

### Open WebUI Integration

```bash
# Mit Open WebUI Profil starten
docker-compose --profile ollama up -d

# Open WebUI im Browser öffnen
# http://localhost:3000
```

## Chunking Strategie

Jedes GSRS Substance Dokument wird in Chunks basierend auf Element-Pfaden aufgeteilt:

### Beispiel

**Input JSON:**
```json
{
    "uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028",
    "substanceClass": "chemical",
    "codes": [
        {
            "code": "WK2XYI10QM",
            "codeSystem": "FDA UNII"
        },
        {
            "code": "CHEMBL521",
            "codeSystem": "ChEMBL"
        }
    ]
}
```

**Erstellte Chunks:**
| Element Path | Chunk Text | Metadata |
|--------------|------------|----------|
| `root_codes_0_code` | code: WK2XYI10QM | {codeSystem: FDA UNII} |
| `root_codes_0_codeSystem` | codeSystem: FDA UNII | {} |
| `root_codes_1_code` | code: CHEMBL521 | {codeSystem: ChEMBL} |
| `root_codes_1_codeSystem` | codeSystem: ChEMBL | {} |

## Entwicklung

### Lokale Entwicklung ohne Docker

```bash
# PostgreSQL mit pgvector installieren
# Ubuntu: sudo apt install postgresql-16-pgvector

# Virtuelle Umgebung erstellen
python -m venv venv
source venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt

# .env Datei erstellen
cp .env.example .env

# OPENAI_API_KEY setzen (für OpenAI Embeddings)
export OPENAI_API_KEY="sk-..."

# Datenbank erstellen
createdb -U postgres gsrs_rag

# App starten
uvicorn app.main:app --reload
```

### Tests

```bash
# Unit Tests für Vector Database Backends
python -m pytest tests/ -v

# Nur ChromaDB Tests
python -m pytest tests/test_vector_db.py -v

# Chunking Tests
python -m pytest tests/test_chunking.py -v
```

## Projektstruktur

```
gsrs-rag-gateway/
├── app/
│   ├── __init__.py
│   ├── config.py              # Konfiguration
│   ├── main.py                # FastAPI App
│   ├── models/
│   │   └── db.py              # SQLAlchemy Modelle
│   ├── db/
│   │   ├── base.py            # Vector Database Interface
│   │   ├── factory.py         # Backend Factory
│   │   └── backends/
│   │       ├── chroma.py      # ChromaDB Backend
│   │       └── pgvector.py    # pgvector Backend
│   └── services/
│       ├── __init__.py
│       ├── chunking.py        # ChunkerService
│       ├── embedding.py       # EmbeddingService
│       └── vector_database.py # VectorDatabaseService
├── scripts/
│   ├── load_data.py           # Ladeskript für .gsrs Dateien
│   ├── download_samples.py    # Sample Daten Downloader
│   └── test_gateway.py        # Quick Test Script
├── examples/
│   ├── chatgpt_integration.py
│   ├── ollama_integration.py
│   └── embedding_examples.py
├── tests/
│   ├── test_chunking.py
│   └── test_vector_db.py
├── docs/
│   ├── quickstart.md
│   ├── configuration.md
│   ├── authentication.md
│   ├── data-loading.md
│   ├── troubleshooting.md
│   ├── vector-databases.md
│   └── guides/
│       ├── chunking.md
│       ├── chatgpt.md
│       └── ollama-open-webui.md
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── requirements-examples.txt
├── pyproject.toml
└── README.md
```

## API Dokumentation

Die vollständige API Dokumentation ist unter Swagger UI verfügbar:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Fehlerbehebung

### Datenbank Verbindung fehlgeschlagen

```bash
# PostgreSQL Logs prüfen
docker-compose logs postgres

# Verbindung testen
docker-compose exec postgres pg_isready -U gsrs
```

### Embedding API Fehler (OpenAI)

```bash
# API Key prüfen
echo $EMBEDDING_API_KEY

# API Verfügbarkeit testen
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $EMBEDDING_API_KEY"
```

### Embedding API Fehler (Ollama)

```bash
# Ollama Status prüfen
curl http://localhost:11434/api/tags

# Modell pullen
ollama pull nomic-embed-text
```

### Vektorsuche gibt keine Ergebnisse

```bash
# Prüfen ob Daten geladen sind
curl http://localhost:8000/statistics

# Substance Klassen prüfen
curl http://localhost:8000/substance-classes
```

## Lizenz

MIT License - siehe [LICENSE](LICENSE) Datei.

## Beiträge

Beiträge sind willkommen! Bitte erstelle ein Issue oder Pull Request für Verbesserungen.

## Links

- [GitHub Repository](https://github.com/epuzanov/gsrs-rag-gateway)
- [GSRS Model Library](https://github.com/epuzanov/gsrs.model)
- [GSRS Specification](https://github.com/IHEC/gsrs)
- [pgvector](https://github.com/pgvector/pgvector)
- [ChromaDB](https://docs.trychroma.com/)
