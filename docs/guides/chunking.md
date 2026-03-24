# GSRS RAG Gateway - Chunking Documentation

Comprehensive documentation for the GSRS substance chunking service.

## Overview

The GSRS RAG Gateway automatically chunks GSRS (Global Substance Registration System) substance documents into meaningful, searchable units using the `gsrs.model` library. Each chunk preserves the substance context while focusing on specific elements or sections.

## Chunking Service

The chunking is handled by the `ChunkerService` class, which uses the `Substance.to_embedding_chunks()` method from the `gsrs.model` library.

```python
from app.services.chunking import ChunkerService

# Initialize chunker
chunker = ChunkerService()

# Chunk a substance document
substance = {...}  # GSRS substance JSON
chunks = chunker.chunk_substance(substance)
```

## VectorDocument Structure

Each chunk is represented as a `VectorDocument` with the following structure:

```python
from app.models import VectorDocument

@dataclass
class VectorDocument:
    chunk_id: str                    # Unique chunk identifier (e.g., "root_names_0_name")
    document_id: UUID                # Substance UUID
    section: str                     # Section name (e.g., "names", "codes", "structure")
    text: str                        # Text content for embeddings
    embedding: List[float]           # Vector embedding (set by EmbeddingService)
    chunk_metadata: Dict[str, Any]   # Additional metadata (chunk_type, attributes, etc.)
    source_url: Optional[str] = None # Optional source URL
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | str | Unique identifier following path convention (e.g., `root_codes_0_code`) |
| `document_id` | UUID | The substance UUID this chunk belongs to |
| `section` | str | The section type (e.g., "names", "codes", "structure", "protein") |
| `text` | str | The actual text content for embedding generation |
| `embedding` | List[float] | Vector embedding (populated by EmbeddingService) |
| `chunk_metadata` | Dict[str, Any] | Additional metadata including chunk_type, attributes, etc. |
| `source_url` | Optional[str] | Optional source URL or reference |

## Substance Classes

The chunking service supports all GSRS substance classes via `gsrs.model`:

| Class | Description | Specific Sections |
|-------|-------------|-------------------|
| `concept` | Base substance class for concepts | None (common sections only) |
| `chemical` | Small molecule chemicals with structure | structure, moieties |
| `protein` | Proteins and monoclonal antibodies | protein (subunits, glycosylation, disulfide links) |
| `nucleicAcid` | RNA/DNA therapeutics | nucleicAcid (subunits, sugars, linkages) |
| `polymer` | Synthetic and biosynthetic polymers | polymer (classification, monomers, structure) |
| `mixture` | Mixtures of multiple components | mixture (components, parent substance) |
| `structurallyDiverse` | Natural products from organisms | structurallyDiverse (source material) |
| `specifiedSubstanceG1` | Specified substances with constituents | specifiedSubstance (constituents) |

## Common Sections

These sections are chunked for all substance classes:

| Section | Description | Example chunk_id |
|---------|-------------|------------------|
| `root` | Substance overview/summary | `root` |
| `names` | Substance names | `root_names_0_name`, `root_names_0_type` |
| `codes` | External identifiers (CAS, UNII, etc.) | `root_codes_0_code`, `root_codes_0_codeSystem` |
| `properties` | Physical/chemical properties | `root_properties_molecularWeight` |
| `relationships` | Relationships to other substances | `root_relationships_0_relatedSubstance` |
| `references` | Source references | `root_references_0_id` |
| `notes` | Validation notes and warnings | `root_notes_0_comment` |

## Class-Specific Chunking

### Chemical Substances

```python
# Structure chunk
VectorDocument(
    chunk_id="root_structure_smiles",
    document_id=UUID("12345678-1234-5678-1234-567812345678"),
    section="structure",
    text="SMILES: CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    chunk_metadata={
        "chunk_type": "structure_smiles",
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    }
)

# Moieties chunk
VectorDocument(
    chunk_id="root_moieties_0_molecularFormula",
    document_id=UUID("12345678-1234-5678-1234-567812345678"),
    section="moieties",
    text="Molecular Formula: C13H18O2",
    chunk_metadata={
        "chunk_type": "moiety_formula",
        "moiety_index": 0
    }
)
```

### Protein Substances

```python
# Protein subunit
VectorDocument(
    chunk_id="root_protein_subunits_0_sequence",
    document_id=UUID("12345678-1234-5678-1234-567812345678"),
    section="protein",
    text="Protein Subunit 1: Heavy Chain | Sequence: QVQLVQSGAEVKKPG...",
    chunk_metadata={
        "chunk_type": "protein_subunit_sequence",
        "subunit_index": 0,
        "sequence_length": 446
    }
)

# Disulfide link
VectorDocument(
    chunk_id="root_protein_disulfideLinks_0_subunit1",
    document_id=UUID("12345678-1234-5678-1234-567812345678"),
    section="protein",
    text="Disulfide Link: Subunit 1 Pos 220 ↔ Subunit 3 Pos 214",
    chunk_metadata={
        "chunk_type": "disulfide_link",
        "subunit1_position": 220,
        "subunit3_position": 214
    }
)
```

### Nucleic Acid Substances

```python
# Nucleic acid subunit
VectorDocument(
    chunk_id="root_nucleicAcid_subunits_0_sequence",
    document_id=UUID("12345678-1234-5678-1234-567812345678"),
    section="nucleicAcid",
    text="Nucleic Acid Subunit 1: Length: 21 | Sequence: ACCUCACCAAGGCCAGCACUU",
    chunk_metadata={
        "chunk_type": "nucleic_acid_subunit_sequence",
        "subunit_index": 0,
        "sequence_length": 21
    }
)
```

### Polymer Substances

```python
# Polymer monomer
VectorDocument(
    chunk_id="root_polymer_mononers_0_name",
    document_id=UUID("12345678-1234-5678-1234-567812345678"),
    section="polymer",
    text="Polymer Monomer: Propylene | Mole Ratio: 100",
    chunk_metadata={
        "chunk_type": "polymer_monomer",
        "monomer_index": 0
    }
)
```

### Mixture Substances

```python
# Mixture component
VectorDocument(
    chunk_id="root_mixture_components_0_name",
    document_id=UUID("12345678-1234-5678-1234-567812345678"),
    section="mixture",
    text="Mixture Component 1: ZUCLOMIPHENE CITRATE | Strength: 30-50%",
    chunk_metadata={
        "chunk_type": "mixture_component",
        "component_index": 0
    }
)
```

## Chunk ID Naming Convention

Chunk IDs follow a consistent path-based naming pattern:

```
root_<section>_<index>_<field>
```

### Examples

| chunk_id | Description |
|----------|-------------|
| `root` | Substance root/overview |
| `root_names_0_name` | First name entry, name field |
| `root_names_0_type` | First name entry, type field |
| `root_codes_0_code` | First code entry, code value |
| `root_codes_0_codeSystem` | First code entry, code system |
| `root_structure_smiles` | Structure section, SMILES field |
| `root_protein_subunits_0_sequence` | First protein subunit, sequence |
| `root_moieties_0_molecularFormula` | First moiety, molecular formula |

## Usage

### Basic Usage

```python
from app.services.chunking import ChunkerService

# Initialize chunker
chunker = ChunkerService()

# Chunk a substance document
substance = {
    "uuid": "12345678-1234-5678-1234-567812345678",
    "substanceClass": "chemical",
    "names": [{"name": "Aspirin", "type": "COMMON"}],
    "codes": [{"code": "50-78-2", "codeSystem": "CAS"}]
}

chunks = chunker.chunk_substance(substance)

# Process chunks
for chunk in chunks:
    print(f"{chunk.chunk_id} ({chunk.section}): {chunk.text[:50]}...")
```

### From JSON String

```python
from app.services.chunking import chunk_substance_json

json_str = '{"uuid": "...", "substanceClass": "chemical", ...}'
chunks = chunk_substance_json(json_str)
```

### With Embeddings

```python
from app.services.chunking import ChunkerService
from app.services.embedding import EmbeddingService
from app.services.vector_database import VectorDatabaseService

chunker = ChunkerService()
embedding_service = EmbeddingService(api_key="...")
db_service = VectorDatabaseService(database_url="...")

# Chunk substance
substance = {...}
chunks = chunker.chunk_substance(substance)

# Generate embeddings
texts = [chunk.text for chunk in chunks]
embeddings = embedding_service.embed_batch(texts)

# Store in database
db_service.upsert_chunks(chunks, embeddings)
```

### Accessing Metadata

```python
for chunk in chunks:
    # Access chunk metadata
    chunk_type = chunk.chunk_metadata.get("chunk_type")
    
    # Filter by section
    if chunk.section == "codes":
        print(f"Code: {chunk.text}")
    
    # Filter by chunk_type
    if chunk.chunk_metadata.get("chunk_type") == "names":
        print(f"Name: {chunk.text}")
```

## Testing

The chunking service has comprehensive test coverage:

```bash
# Run chunking tests
python -m pytest tests/test_chunking.py -v

# Run specific test class
python -m pytest tests/test_chunking.py::TestChunkerService -v
```

### Test Coverage

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestChunkerService` | 6 | ChunkerService functionality |
| `test_chunker_initialization` | 1 | Service initialization |
| `test_chemical_substance_chunking` | 1 | Chemical substance chunking |
| `test_chunk_substance_json` | 1 | JSON string convenience function |
| `test_concept_substance_chunking` | 1 | Concept substance class |
| `test_names_chunking` | 1 | Names section chunking |
| `test_codes_chunking` | 1 | Codes section chunking |

## Best Practices

### 1. Chunk Size

The `gsrs.model` library creates appropriately sized chunks:
- Overview chunks: ~100-200 characters
- Section chunks: ~50-300 characters
- Subunit/Component chunks: ~100-500 characters

### 2. Metadata Usage

Use `chunk_metadata` for filtering and faceted search:

```python
# Filter by chunk_type
name_chunks = [c for c in chunks if c.chunk_metadata.get("chunk_type") == "name"]

# Filter by section
code_chunks = [c for c in chunks if c.section == "codes"]

# Access specific metadata
for chunk in chunks:
    if "sequence_length" in chunk.chunk_metadata:
        print(f"Sequence length: {chunk.chunk_metadata['sequence_length']}")
```

### 3. Embedding Generation

Always generate embeddings after chunking:

```python
# Generate embeddings for all chunks
texts = [chunk.text for chunk in chunks]
embeddings = embedding_service.embed_batch(texts)

# Assign embeddings to chunks
for chunk, embedding in zip(chunks, embeddings):
    chunk.set_embedding(embedding)
```

### 4. Database Storage

Store chunks with their embeddings:

```python
# Upsert chunks with embeddings
count = db_service.upsert_chunks(chunks, embeddings)
print(f"Stored {count} chunks")
```

## Troubleshooting

### No Chunks Created

Check that the substance has required fields:

```python
assert "uuid" in substance
assert "substanceClass" in substance

# Validate with gsrs.model
from gsrs.model import Substance
gsrs_substance = Substance.model_validate(substance)  # Should not raise
```

### Invalid Chunk Data

Verify the substance JSON is valid GSRS format:

```python
from gsrs.model import Substance

try:
    gsrs_substance = Substance.model_validate(substance)
    chunks = gsrs_substance.to_embedding_chunks()
except ValidationError as e:
    print(f"Invalid substance: {e}")
```

### Missing Metadata

Some chunks may have empty metadata:

```python
for chunk in chunks:
    metadata = chunk.chunk_metadata or {}
    chunk_type = metadata.get("chunk_type", "unknown")
```

## Integration with gsrs.model

The chunking service relies on the `gsrs.model` library:

```python
from gsrs.model import Substance

# Parse substance
gsrs_substance = Substance.model_validate(substance_json)

# Get embedding-ready chunks
raw_chunks = gsrs_substance.to_embedding_chunks()

# Each raw chunk is a dict with:
# - chunk_id: str
# - document_id: UUID
# - section: str
# - text: str
# - metadata: Dict[str, Any]
```

## Related Documentation

- [Data Loading Guide](../data-loading.md) - How to load substances
- [API Reference](../api-reference.md) - API endpoints
- [Vector Databases](../vector-databases.md) - Database backends
- [Embedding Service](embedding.md) - Generating embeddings
