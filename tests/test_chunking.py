"""
GSRS RAG Gateway - Chunking Service Tests

Tests for the ChunkerService using gsrs.model library's
Substance.to_embedding_chunks() method.
"""
import pytest
from app.services import ChunkerService, chunk_substance_json
from app.models import VectorDocument


class TestChunkerService:
    """Tests for ChunkerService using gsrs.model."""

    def test_chunker_initialization(self):
        """Test chunker can be initialized."""
        chunker = ChunkerService()
        assert chunker is not None

    def test_chemical_substance_chunking(self):
        """Test chemical substance chunking with minimal valid data."""
        substance = {
            "uuid": "12345678-1234-5678-1234-567812345678",
            "substanceClass": "chemical",
            "names": [
                {
                    "name": "Aspirin",
                    "type": "COMMON",
                    "displayName": True,
                    "preferred": True,
                    "languages": ["en"]
                }
            ],
            "references": [
                {
                    "docType": "journal article",
                    "id": "12345"
                }
            ],
            "version": "1.0",
            "status": "Active",
            "structure": {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "stereochemistry": "ACHIRAL",
                "opticalActivity": "NONE",
                "atropisomerism": "No"
            },
            "moieties": [
                {
                    "smiles": "CC(=O)O",
                    "stereochemistry": "ACHIRAL",
                    "opticalActivity": "NONE",
                    "atropisomerism": "No"
                }
            ]
        }

        chunker = ChunkerService()
        chunks = chunker.chunk_substance(substance)

        assert len(chunks) > 0
        # Check that chunks are VectorDocument instances
        assert all(isinstance(c, VectorDocument) for c in chunks)

        # Check for root chunk
        root_chunks = [c for c in chunks if c.section == 'summary']
        assert len(root_chunks) > 0
        assert "Aspirin" in root_chunks[0].text

    def test_chunk_substance_json(self):
        """Test chunk_substance_json convenience function."""
        import json

        substance = {
            "uuid": "12345678-1234-5678-1234-567812345678",
            "substanceClass": "chemical",
            "names": [
                {
                    "name": "Test Substance",
                    "type": "COMMON",
                    "displayName": True,
                    "preferred": True,
                    "languages": ["en"]
                }
            ],
            "references": [
                {
                    "docType": "journal article",
                    "id": "12345"
                }
            ],
            "version": "1.0",
            "status": "Active",
            "structure": {
                "smiles": "CCO",
                "stereochemistry": "ACHIRAL",
                "opticalActivity": "NONE",
                "atropisomerism": "No"
            },
            "moieties": [
                {
                    "smiles": "CCO",
                    "stereochemistry": "ACHIRAL",
                    "opticalActivity": "NONE",
                    "atropisomerism": "No"
                }
            ]
        }

        chunks = chunk_substance_json(json.dumps(substance))
        assert len(chunks) > 0
        assert all(isinstance(c, VectorDocument) for c in chunks)

    def test_concept_substance_chunking(self):
        """Test concept substance chunking."""
        substance = {
            "uuid": "12345678-1234-5678-1234-567812345678",
            "substanceClass": "concept",
            "names": [
                {
                    "name": "Test Concept",
                    "type": "COMMON",
                    "displayName": True,
                    "preferred": True,
                    "languages": ["en"]
                }
            ],
            "references": [
                {
                    "docType": "journal article",
                    "id": "12345"
                }
            ],
            "version": "1.0",
            "status": "Active"
        }

        chunker = ChunkerService()
        chunks = chunker.chunk_substance(substance)

        assert len(chunks) > 0
        assert all(isinstance(c, VectorDocument) for c in chunks)

        # Check for root chunk
        root_chunks = [c for c in chunks if c.section == 'summary']
        assert len(root_chunks) > 0
        assert "Test Concept" in root_chunks[0].text

    def test_names_chunking(self):
        """Test that names are properly chunked."""
        substance = {
            "uuid": "12345678-1234-5678-1234-567812345678",
            "substanceClass": "chemical",
            "names": [
                {
                    "name": "Aspirin",
                    "type": "COMMON",
                    "displayName": True,
                    "preferred": True,
                    "languages": ["en"]
                },
                {
                    "name": "Acetylsalicylic acid",
                    "type": "SYSTEMATIC",
                    "languages": ["en"]
                }
            ],
            "references": [
                {
                    "docType": "journal article",
                    "id": "12345"
                }
            ],
            "version": "1.0",
            "status": "Active",
            "structure": {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "stereochemistry": "ACHIRAL",
                "opticalActivity": "NONE",
                "atropisomerism": "No"
            },
            "moieties": [
                {
                    "smiles": "CC(=O)O",
                    "stereochemistry": "ACHIRAL",
                    "opticalActivity": "NONE",
                    "atropisomerism": "No"
                }
            ]
        }

        chunker = ChunkerService()
        chunks = chunker.chunk_substance(substance)

        # Should have chunks for names
        name_chunks = [c for c in chunks if 'name' in c.section.lower()]
        assert len(name_chunks) > 0

        # Check that both names appear in chunks
        all_text = ' '.join([c.text for c in chunks])
        assert 'Aspirin' in all_text
        assert 'Acetylsalicylic acid' in all_text

    def test_codes_chunking(self):
        """Test that codes are properly chunked."""
        substance = {
            "uuid": "12345678-1234-5678-1234-567812345678",
            "substanceClass": "chemical",
            "names": [
                {
                    "name": "Test Substance",
                    "type": "COMMON",
                    "displayName": True,
                    "preferred": True,
                    "languages": ["en"]
                }
            ],
            "codes": [
                {
                    "code": "ASA-101",
                    "codeSystem": "INTERNAL",
                    "type": "SYSTEM"
                }
            ],
            "references": [
                {
                    "docType": "journal article",
                    "id": "12345"
                }
            ],
            "version": "1.0",
            "status": "Active",
            "structure": {
                "smiles": "CCO",
                "stereochemistry": "ACHIRAL",
                "opticalActivity": "NONE",
                "atropisomerism": "No"
            },
            "moieties": [
                {
                    "smiles": "CCO",
                    "stereochemistry": "ACHIRAL",
                    "opticalActivity": "NONE",
                    "atropisomerism": "No"
                }
            ]
        }

        chunker = ChunkerService()
        chunks = chunker.chunk_substance(substance)

        # Should have chunks for codes
        code_chunks = [c for c in chunks if 'code' in c.section.lower()]
        assert len(code_chunks) > 0

        # Check that code appears in chunks
        all_text = ' '.join([c.text for c in chunks])
        assert 'ASA-101' in all_text
