"""
GSRS RAG Gateway - Data Loader Script Tests

Tests for the load_data.py script functionality.
"""
import pytest
import gzip
import json
import tempfile
import os
from pathlib import Path
from scripts.load_data import (
    parse_gsrs_file,
    fetch_substance_by_uuid,
    fetch_all_substance_uuids,
    ingest_batch
)


class TestParseGsrsFile:
    """Tests for .gsrs file parsing."""

    def test_parse_gsrs_file_basic(self):
        """Test parsing basic .gsrs file."""
        substances = [
            {"uuid": "uuid-1", "substanceClass": "chemical"},
            {"uuid": "uuid-2", "substanceClass": "protein"}
        ]

        with tempfile.NamedTemporaryFile(suffix=".gsrs", delete=False) as f:
            temp_path = f.name
            with gzip.open(f, 'wt', encoding='utf-8') as gz:
                for sub in substances:
                    gz.write('\t\t' + json.dumps(sub) + '\n')

        try:
            parsed = list(parse_gsrs_file(temp_path))
            assert len(parsed) == 2
            assert parsed[0]["uuid"] == "uuid-1"
            assert parsed[1]["uuid"] == "uuid-2"
        finally:
            os.unlink(temp_path)

    def test_parse_gsrs_file_with_empty_lines(self):
        """Test parsing .gsrs file with empty lines."""
        with tempfile.NamedTemporaryFile(suffix=".gsrs", delete=False) as f:
            temp_path = f.name
            with gzip.open(f, 'wt', encoding='utf-8') as gz:
                gz.write('\t\t{"uuid": "uuid-1"}\n')
                gz.write('\t\t\n')  # Empty line with tabs
                gz.write('\n')  # Empty line
                gz.write('\t\t{"uuid": "uuid-2"}\n')

        try:
            parsed = list(parse_gsrs_file(temp_path))
            assert len(parsed) == 2
        finally:
            os.unlink(temp_path)

    def test_parse_gsrs_file_with_invalid_json(self):
        """Test parsing .gsrs file with invalid JSON lines."""
        with tempfile.NamedTemporaryFile(suffix=".gsrs", delete=False) as f:
            temp_path = f.name
            with gzip.open(f, 'wt', encoding='utf-8') as gz:
                gz.write('\t\t{"uuid": "uuid-1"}\n')
                gz.write('\t\t{invalid json}\n')  # Invalid JSON
                gz.write('\t\t{"uuid": "uuid-2"}\n')

        try:
            parsed = list(parse_gsrs_file(temp_path))
            # Should skip invalid lines
            assert len(parsed) == 2
        finally:
            os.unlink(temp_path)

    def test_parse_gsrs_file_single_tab(self):
        """Test parsing .gsrs file with single leading tab."""
        with tempfile.NamedTemporaryFile(suffix=".gsrs", delete=False) as f:
            temp_path = f.name
            with gzip.open(f, 'wt', encoding='utf-8') as gz:
                gz.write('\t{"uuid": "uuid-1"}\n')  # Single tab

        try:
            parsed = list(parse_gsrs_file(temp_path))
            assert len(parsed) == 1
            assert parsed[0]["uuid"] == "uuid-1"
        finally:
            os.unlink(temp_path)


class TestIngestBatch:
    """Tests for batch ingestion."""

    def test_ingest_batch_api_error(self):
        """Test ingestion with unavailable API."""
        substances = [{"uuid": "test-uuid"}]
        result = ingest_batch(substances, "http://invalid-url-12345:9999")

        assert result["successful"] == 0
        assert result["failed"] == 1
        assert len(result["errors"]) > 0


class TestFetchSubstanceByUuid:
    """Tests for fetching substances from GSRS API."""

    @pytest.mark.asyncio
    async def test_fetch_valid_substance(self):
        """Test fetching a valid substance from GSRS API."""
        import httpx

        # Known valid UUID from GSRS
        test_uuid = "0103a288-6eb6-4ced-b13a-849cd7edf028"  # Ibuprofen

        async with httpx.AsyncClient(timeout=30.0) as session:
            substance = await fetch_substance_by_uuid(test_uuid, session)

        assert substance is not None
        assert substance["uuid"] == test_uuid
        assert substance["substanceClass"] == "chemical"

    @pytest.mark.asyncio
    async def test_fetch_invalid_substance(self):
        """Test fetching an invalid substance UUID."""
        import httpx

        invalid_uuid = "00000000-0000-0000-0000-000000000000"

        async with httpx.AsyncClient(timeout=30.0) as session:
            substance = await fetch_substance_by_uuid(invalid_uuid, session)

        assert substance is None

    @pytest.mark.asyncio
    async def test_fetch_multiple_substances_parallel(self):
        """Test fetching multiple substances in parallel."""
        import httpx

        test_uuids = [
            "0103a288-6eb6-4ced-b13a-849cd7edf028",  # Ibuprofen
            "80edf0eb-b6c5-4a9a-adde-28c7254046d9",  # Chemical
        ]

        async with httpx.AsyncClient(timeout=30.0) as session:
            import asyncio
            tasks = [fetch_substance_by_uuid(uuid, session) for uuid in test_uuids]
            results = await asyncio.gather(*tasks)

        valid_results = [r for r in results if r is not None]
        assert len(valid_results) == 2


class TestFetchAllSubstanceUuids:
    """Tests for fetching all substance UUIDs."""

    @pytest.mark.asyncio
    async def test_fetch_uuids_limited(self):
        """Test fetching limited number of UUIDs."""
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as session:
            uuids = await fetch_all_substance_uuids(session, max_results=10)

        assert len(uuids) <= 10
        assert all(isinstance(u, str) for u in uuids)

    @pytest.mark.asyncio
    async def test_fetch_uuids_format(self):
        """Test that fetched UUIDs have correct format."""
        import httpx
        import re

        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )

        async with httpx.AsyncClient(timeout=30.0) as session:
            uuids = await fetch_all_substance_uuids(session, max_results=5)

        for uuid in uuids:
            assert uuid_pattern.match(uuid), f"Invalid UUID format: {uuid}"
