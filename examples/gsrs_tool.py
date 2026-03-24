"""
title: GSRS RAG Gateway Tool
author: Egor Puzanov
version: 0.2
description: Query GSRS substance database using the RAG Gateway ERI endpoint
"""

from pydantic import BaseModel, Field
from typing import Optional
import httpx
import json


class Tools:
    class Valves(BaseModel):
        rag_base_url: str = Field(
            default="http://gsrs-rag-gateway:8000",
            description="Base URL for the GSRS RAG Gateway API"
        )
        api_username: str = Field(
            default="admin",
            description="Username for RAG Gateway authentication"
        )
        api_password: str = Field(
            default="admin123",
            description="Password for RAG Gateway authentication"
        )
        top_k: int = Field(
            default=5,
            description="Number of retrieval results to fetch"
        )

    def __init__(self):
        self.valves = self.Valves()

    def gsrs_substance_query(
        self,
        query: str = Field(
            ...,
            description="Search query about chemical substances (e.g., 'CAS code for Aspirin', 'molecular weight of Ibuprofen')"
        ),
        top_k: Optional[int] = Field(
            default=None,
            description="Number of results to return (default: uses valve setting)"
        )
    ) -> str:
        """
        Query the GSRS substance database for chemical substance information.
        
        Use this tool when the user asks about:
        - Chemical codes (CAS, UNII, ChEMBL, etc.)
        - Molecular properties (weight, formula, etc.)
        - Substance names and synonyms
        - Protein or nucleic acid structures
        - Any GSRS substance data
        
        Args:
            query: The search query about a chemical substance
            top_k: Optional number of results (default: 5)
        
        Returns:
            Formatted search results from the GSRS database
        """
        try:
            # Use top_k from parameter or valve
            k = top_k if top_k is not None else self.valves.top_k
            
            # Query RAG Gateway ERI endpoint
            with httpx.Client() as client:
                response = client.post(
                    f"{self.valves.rag_base_url}/eri/query",
                    json={"query": query, "top_k": k},
                    timeout=30,
                    auth=(self.valves.api_username, self.valves.api_password)
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if not results:
                        return "No results found in the GSRS database."
                    
                    # Format results
                    formatted = f"Found {len(results)} result(s) for '{query}':\n\n"
                    for i, r in enumerate(results, 1):
                        formatted += f"{i}. {r.get('text', 'N/A')}\n"
                        formatted += f"   Score: {r.get('score', 0):.2f}\n"
                        metadata = r.get('metadata', {})
                        if 'source_url' in metadata:
                            formatted += f"   Source: {metadata['source_url']}\n"
                        if 'document_id' in metadata:
                            formatted += f"   Substance UUID: {metadata['document_id']}\n"
                        if 'section' in metadata:
                            formatted += f"   Source: {metadata['section']}\n"
                        formatted += "\n"
                    
                    return formatted
                else:
                    return f"Error querying GSRS database: HTTP {response.status_code}"
                    
        except httpx.TimeoutException:
            return "Timeout: The GSRS database query took too long."
        except httpx.ConnectError as e:
            return f"Connection error: Could not connect to GSRS RAG Gateway at {self.valves.rag_base_url}. Make sure the service is running."
        except Exception as e:
            return f"Error querying GSRS database: {str(e)}"
