"""
GSRS RAG Gateway - Data Loader Script Tests

Tests for the load_data.py script functionality.
"""
import gzip
import json
import os
import tempfile

import pytest

from scripts.load_data import (
    fetch_all_substance_uuids,
    fetch_substance_by_uuid,
    ingest_batch,
    load_from_file,
    load_substances_from_api,
    parse_gsrs_file,
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
                gz.write('\t\t\n')
                gz.write('\n')
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
                gz.write('\t\t{invalid json}\n')
                gz.write('\t\t{"uuid": "uuid-2"}\n')

        try:
            parsed = list(parse_gsrs_file(temp_path))
            assert len(parsed) == 2
        finally:
            os.unlink(temp_path)

    def test_parse_gsrs_file_single_tab(self):
        """Test parsing .gsrs file with single leading tab."""
        with tempfile.NamedTemporaryFile(suffix=".gsrs", delete=False) as f:
            temp_path = f.name
            with gzip.open(f, 'wt', encoding='utf-8') as gz:
                gz.write('\t{"uuid": "uuid-1"}\n')

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

    def test_ingest_batch_disables_certificate_validation(self, monkeypatch):
        """Test ingestion can disable TLS certificate validation."""
        seen = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"successful": 1, "failed": 0, "total_chunks": 1, "errors": []}

        class FakeClient:
            def __init__(self, *, timeout, verify):
                seen["timeout"] = timeout
                seen["verify"] = verify

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, endpoint, json):
                seen["endpoint"] = endpoint
                return FakeResponse()

        monkeypatch.setattr("scripts.load_data.httpx.Client", FakeClient)

        result = ingest_batch([{"uuid": "test-uuid"}], "https://gateway.example", verify_ssl=False)

        assert result["successful"] == 1
        assert seen["verify"] is False
        assert seen["endpoint"] == "https://gateway.example/ingest/batch"

    def test_load_from_file_disables_certificate_validation(self, monkeypatch):
        """Test file loading passes disabled TLS verification to health and ingest calls."""
        seen = {"verify": []}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"status": "healthy", "database_connected": True}

        class FakeClient:
            def __init__(self, *, timeout, verify):
                seen["verify"].append(verify)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def get(self, url):
                seen["health_url"] = url
                return FakeResponse()

        monkeypatch.setattr("scripts.load_data.httpx.Client", FakeClient)

        captured = {}

        def fake_ingest_batch(substances, api_url, timeout=300, verify_ssl=True):
            captured["verify_ssl"] = verify_ssl
            return {"successful": len(substances), "failed": 0, "total_chunks": 2, "errors": []}

        monkeypatch.setattr("scripts.load_data.ingest_batch", fake_ingest_batch)

        with tempfile.NamedTemporaryFile(suffix=".gsrs", delete=False) as f:
            temp_path = f.name
            with gzip.open(f, 'wt', encoding='utf-8') as gz:
                gz.write('\t\t{"uuid": "uuid-1"}\n')

        try:
            result = load_from_file(
                temp_path,
                batch_size=1,
                api_url="https://gateway.example",
                verify_ssl=False,
            )
        finally:
            os.unlink(temp_path)

        assert result["successful"] == 1
        assert seen["verify"] == [False]
        assert captured["verify_ssl"] is False
        assert seen["health_url"] == "https://gateway.example/health"


class TestFetchSubstanceByUuid:
    """Tests for fetching substances from GSRS API."""

    @pytest.mark.asyncio
    async def test_fetch_valid_substance(self):
        """Test fetching a valid substance from GSRS API."""
        import httpx

        test_uuid = "0103a288-6eb6-4ced-b13a-849cd7edf028"

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
            "0103a288-6eb6-4ced-b13a-849cd7edf028",
            "80edf0eb-b6c5-4a9a-adde-28c7254046d9",
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

    @pytest.mark.asyncio
    async def test_load_substances_from_api_disables_certificate_validation(self, monkeypatch):
        """Test API loading passes disabled TLS verification to both async and sync clients."""
        seen = {"async_verify": [], "sync_verify": []}

        class FakeAsyncResponse:
            def __init__(self, payload, status_code=200):
                self._payload = payload
                self.status_code = status_code

            def json(self):
                return self._payload

        class FakeAsyncClient:
            def __init__(self, *, timeout, verify):
                seen["async_verify"].append(verify)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, url, timeout=30.0, params=None):
                substance_uuid = url.split("(")[-1].split(")")[0]
                return FakeAsyncResponse({"uuid": substance_uuid})

        class FakeSyncResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"status": "healthy"}

        class FakeSyncClient:
            def __init__(self, *, timeout, verify):
                seen["sync_verify"].append(verify)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def get(self, url):
                seen["health_url"] = url
                return FakeSyncResponse()

        monkeypatch.setattr("scripts.load_data.httpx.AsyncClient", FakeAsyncClient)
        monkeypatch.setattr("scripts.load_data.httpx.Client", FakeSyncClient)

        captured = {}

        def fake_ingest_batch(substances, api_url, timeout=300, verify_ssl=True):
            captured["verify_ssl"] = verify_ssl
            return {
                "successful": len(substances),
                "failed": 0,
                "total_chunks": len(substances),
                "errors": [],
            }

        monkeypatch.setattr("scripts.load_data.ingest_batch", fake_ingest_batch)

        result = await load_substances_from_api(
            ["uuid-1", "uuid-2"],
            batch_size=2,
            api_url="https://gateway.example",
            verify_ssl=False,
        )

        assert result["downloaded"] == 2
        assert result["successful"] == 2
        assert seen["async_verify"] == [False]
        assert seen["sync_verify"] == [False]
        assert captured["verify_ssl"] is False
