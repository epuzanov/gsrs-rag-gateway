"""
GSRS RAG Gateway - Chunking Service Tests

Tests for the ChunkerService using gsrs.model library's
Substance.to_embedding_chunks() method.
"""
import unittest

from gsrs.model import Substance
from gsrs.services.ai import SubstanceChunker

from app.models import VectorDocument


class TestChunkerService(unittest.TestCase):
    """Tests for ChunkerService using gsrs.model."""

    @classmethod
    def setUpClass(cls):
        cls._chunker = SubstanceChunker(class_=VectorDocument)

    @staticmethod
    def _to_substance(payload):
        return Substance.model_validate(payload)

    def test_chunker_initialization(self):
        """Test chunker can be initialized."""
        self.assertIsNotNone(self._chunker)

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
                    "languages": ["en"],
                }
            ],
            "references": [{"docType": "journal article", "id": "12345"}],
            "version": "1.0",
            "status": "Active",
            "structure": {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "stereochemistry": "ACHIRAL",
                "opticalActivity": "NONE",
                "atropisomerism": "No",
            },
            "moieties": [
                {
                    "smiles": "CC(=O)O",
                    "stereochemistry": "ACHIRAL",
                    "opticalActivity": "NONE",
                    "atropisomerism": "No",
                }
            ],
        }

        chunks = self._chunker.chunk(self._to_substance(substance))

        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(c, VectorDocument) for c in chunks))

        root_chunks = [c for c in chunks if c.section == "summary"]
        self.assertGreater(len(root_chunks), 0)
        self.assertIn("Aspirin", root_chunks[0].text)

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
                    "languages": ["en"],
                }
            ],
            "references": [{"docType": "journal article", "id": "12345"}],
            "version": "1.0",
            "status": "Active",
        }

        chunks = self._chunker.chunk(self._to_substance(substance))

        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(c, VectorDocument) for c in chunks))

        root_chunks = [c for c in chunks if c.section == "summary"]
        self.assertGreater(len(root_chunks), 0)
        self.assertIn("Test Concept", root_chunks[0].text)

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
                    "languages": ["en"],
                },
                {
                    "name": "Acetylsalicylic acid",
                    "type": "SYSTEMATIC",
                    "languages": ["en"],
                },
            ],
            "references": [{"docType": "journal article", "id": "12345"}],
            "version": "1.0",
            "status": "Active",
            "structure": {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "stereochemistry": "ACHIRAL",
                "opticalActivity": "NONE",
                "atropisomerism": "No",
            },
            "moieties": [
                {
                    "smiles": "CC(=O)O",
                    "stereochemistry": "ACHIRAL",
                    "opticalActivity": "NONE",
                    "atropisomerism": "No",
                }
            ],
        }

        chunks = self._chunker.chunk(self._to_substance(substance))

        name_chunks = [c for c in chunks if "name" in c.section.lower()]
        self.assertGreater(len(name_chunks), 0)

        all_text = " ".join([c.text for c in chunks])
        self.assertIn("Aspirin", all_text)
        self.assertIn("Acetylsalicylic acid", all_text)

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
                    "languages": ["en"],
                }
            ],
            "codes": [{"code": "ASA-101", "codeSystem": "INTERNAL", "type": "SYSTEM"}],
            "references": [{"docType": "journal article", "id": "12345"}],
            "version": "1.0",
            "status": "Active",
            "structure": {
                "smiles": "CCO",
                "stereochemistry": "ACHIRAL",
                "opticalActivity": "NONE",
                "atropisomerism": "No",
            },
            "moieties": [
                {
                    "smiles": "CCO",
                    "stereochemistry": "ACHIRAL",
                    "opticalActivity": "NONE",
                    "atropisomerism": "No",
                }
            ],
        }

        chunks = self._chunker.chunk(self._to_substance(substance))

        code_chunks = [c for c in chunks if "code" in c.section.lower()]
        self.assertGreater(len(code_chunks), 0)

        all_text = " ".join([c.text for c in chunks])
        self.assertIn("ASA-101", all_text)


if __name__ == "__main__":
    unittest.main()

