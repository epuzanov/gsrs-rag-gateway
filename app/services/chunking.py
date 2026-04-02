"""
GSRS RAG Gateway - Chunking Service
Chunks GSRS Substance JSON into substance-level and element-level chunks
using the gsrs.model library's Substance.to_embedding_chunks() method.

Substance Classes:
- concept: Base substance class for all other substance classes
- chemical: Small molecule chemicals with structure
- protein: Proteins including monoclonal antibodies
- nucleicAcid: RNA/DNA therapeutics
- polymer: Synthetic and biosynthetic polymers
- mixture: Mixtures of multiple components
- structurallyDiverse: Natural products from organisms, minerals, etc.
- specifiedSubstanceG1: Specified substances with constituents
"""
import json
from typing import Dict, List, Any

from gsrs.model import Substance
from gsrs.utils import SubstanceChunker
from app.models import VectorDocument


class ChunkerService:
    """
    Chunks GSRS Substance JSON documents using gsrs.model library.

    This chunker uses the Substance.to_embedding_chunks() method from the
    gsrs.model library to create embedding-ready chunks from GSRS Substance
    JSON documents.
    """

    def __init__(self):
        self._chunker = SubstanceChunker()

    def chunk_substance(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """
        Split a substance document into chunks using gsrs.model.

        Args:
            substance: The substance JSON document

        Returns:
            List of VectorDocument objects
        """
        # Validate and parse the substance using gsrs.model
        gsrs_substance = Substance.model_validate(substance)

        # Get embedding chunks from gsrs.model
        gsrs_chunks: List[Dict[str, Any]] = self._chunker.chunk(gsrs_substance)

        # Convert gsrs.model chunks to our Chunk format
        chunks = []
        for gsrs_chunk in gsrs_chunks:
            gsrs_chunk['chunk_metadata'] = gsrs_chunk.pop('metadata', {})
            chunk = VectorDocument(**gsrs_chunk)
            chunks.append(chunk)
        return chunks


def chunk_substance_json(json_str: str) -> List[VectorDocument]:
    """
    Convenience function to chunk a substance JSON string.

    Args:
        json_str: JSON string of the substance

    Returns:
        List of VectorDocument objects
    """
    substance = json.loads(json_str)
    chunker = ChunkerService()
    return chunker.chunk_substance(substance)
