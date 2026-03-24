"""
title: GSRS RAG Query
author: Egor Puzanov
version: 0.1
"""

from pydantic import BaseModel, Field
from typing import Optional
import httpx


class Filter:
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
        min_query_length: int = Field(
            default=5,
            description="Minimum query length to trigger RAG lookup"
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """Query GSRS RAG Gateway and add context to the prompt."""
        messages = body.get("messages", [])
        if not messages:
            return body

        query = messages[-1].get("content", "")

        if len(query) < self.valves.min_query_length:
            return body

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.valves.rag_base_url}/eri/query",
                    json={"query": query, "top_k": self.valves.top_k},
                    timeout=30,
                    auth=(self.valves.api_username, self.valves.api_password)
                )

                if response.status_code == 200:
                    results = response.json().get("results", [])
                    if results:
                        context = "GSRS Database Context:\n\n"
                        for i, r in enumerate(results, 1):
                            context += f"[{i}] {r['text']}\n"
                            metadata = r.get('metadata', {})
                            if 'source_url' in metadata:
                                context += f"   Source: {metadata['source_url']}\n"
                            if 'document_id' in metadata:
                                context += f"   Substance UUID: {metadata['document_id']}\n"
                            if 'section' in metadata:
                                context += f"   Section: {metadata['section']}\n"
                            context += "\n"
                        body["messages"][-1]["content"] = f"{context}\n\nUser Question: {query}"
        except Exception as e:
            print(f"RAG error: {e}")

        return body