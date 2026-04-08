"""
GSRS RAG Gateway - Data Loader Script Tests

Tests for the load_data.py script functionality.
"""
import gzip
import json
import os
import re
import tempfile
import unittest
from unittest.mock import patch

from scripts.load_data import (
    fetch_all_substance_uuids,
    fetch_substance_by_uuid,
    ingest_batch,
    load_from_file,
    load_substances_from_api,
    parse_gsrs_file,
)


class TestParseGsrsFile(unittest.TestCase):
    """Tests for .gsrs file parsing."""

    def test_parse_gsrs_file_basic(self):
        """Test parsing basic .gsrs file."""
        substances = [
            {"uuid": "uuid-1", "substanceClass": "chemical"},
            {"uuid": "uuid-2", "substanceClass": "protein"},
        ]

        with tempfile.NamedTemporaryFile(suffix=".gsrs", delete=False) as f:
            temp_path = f.name
            with gzip.open(f, "wt", encoding="utf-8") as gz:
                for sub in substances:
                    gz.write("\t\t" + json.dumps(sub) + "\n")

        try:
            parsed = list(parse_gsrs_file(temp_path))
            self.assertEqual(len(parsed), 2)
            self.assertEqual(parsed[0]["uuid"], "uuid-1")
            self.assertEqual(parsed[1]["uuid"], "uuid-2")
        finally:
            os.unlink(temp_path)

    def test_parse_gsrs_file_with_empty_lines(self):
        """Test parsing .gsrs file with empty lines."""
        with tempfile.NamedTemporaryFile(suffix=".gsrs", delete=False) as f:
            temp_path = f.name
            with gzip.open(f, "wt", encoding="utf-8") as gz:
                gz.write("\t\t{\"uuid\": \"uuid-1\"}\n")
                gz.write("\t\t\n")
                gz.write("\n")
                gz.write("\t\t{\"uuid\": \"uuid-2\"}\n")

        try:
            parsed = list(parse_gsrs_file(temp_path))
            self.assertEqual(len(parsed), 2)
        finally:
            os.unlink(temp_path)

    def test_parse_gsrs_file_with_invalid_json(self):
        """Test parsing .gsrs file with invalid JSON lines."""
        with tempfile.NamedTemporaryFile(suffix=".gsrs", delete=False) as f:
            temp_path = f.name
            with gzip.open(f, "wt", encoding="utf-8") as gz:
                gz.write("\t\t{\"uuid\": \"uuid-1\"}\n")
                gz.write("\t\t{invalid json}\n")
                gz.write("\t\t{\"uuid\": \"uuid-2\"}\n")

        try:
            parsed = list(parse_gsrs_file(temp_path))
            self.assertEqual(len(parsed), 2)
        finally:
            os.unlink(temp_path)

    def test_parse_gsrs_file_single_tab(self):
        """Test parsing .gsrs file with single leading tab."""
        with tempfile.NamedTemporaryFile(suffix=".gsrs", delete=False) as f:
            temp_path = f.name
            with gzip.open(f, "wt", encoding="utf-8") as gz:
                gz.write("\t{\"uuid\": \"uuid-1\"}\n")

        try:
            parsed = list(parse_gsrs_file(temp_path))
            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0]["uuid"], "uuid-1")
        finally:
            os.unlink(temp_path)


class TestIngestBatch(unittest.TestCase):
    """Tests for batch ingestion."""

    def test_ingest_batch_api_error(self):
        """Test ingestion with unavailable API."""
        substances = [{"uuid": "test-uuid"}]
        result = ingest_batch(substances, "http://invalid-url-12345:9999")

        self.assertEqual(result["successful"], 0)
        self.assertEqual(result["failed"], 1)
        self.assertGreater(len(result["errors"]), 0)

    def test_ingest_batch_disables_certificate_validation(self):
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

        with patch("scripts.load_data.httpx.Client", FakeClient):
            result = ingest_batch([{"uuid": "test-uuid"}], "https://gateway.example", verify_ssl=False)

        self.assertEqual(result["successful"], 1)
        self.assertFalse(seen["verify"])
        self.assertEqual(seen["endpoint"], "https://gateway.example/ingest/batch")

    def test_load_from_file_disables_certificate_validation(self):
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

        captured = {}

        def fake_ingest_batch(substances, api_url, timeout=300, verify_ssl=True):
            captured["verify_ssl"] = verify_ssl
            return {"successful": len(substances), "failed": 0, "total_chunks": 2, "errors": []}

        with patch("scripts.load_data.httpx.Client", FakeClient), patch(
            "scripts.load_data.ingest_batch", fake_ingest_batch
        ):
            with tempfile.NamedTemporaryFile(suffix=".gsrs", delete=False) as f:
                temp_path = f.name
                with gzip.open(f, "wt", encoding="utf-8") as gz:
                    gz.write("\t\t{\"uuid\": \"uuid-1\"}\n")

            try:
                result = load_from_file(
                    temp_path,
                    batch_size=1,
                    api_url="https://gateway.example",
                    verify_ssl=False,
                )
            finally:
                os.unlink(temp_path)

        self.assertEqual(result["successful"], 1)
        self.assertEqual(seen["verify"], [False])
        self.assertFalse(captured["verify_ssl"])
        self.assertEqual(seen["health_url"], "https://gateway.example/health")


class TestFetchSubstanceByUuid(unittest.IsolatedAsyncioTestCase):
    """Tests for fetching substances from GSRS API."""

    async def test_fetch_valid_substance(self):
        """Test fetching a valid substance from GSRS API."""
        test_uuid = "0103a288-6eb6-4ced-b13a-849cd7edf028"

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"uuid": test_uuid, "substanceClass": "chemical"}

        class FakeSession:
            async def get(self, url, timeout=30.0):
                self.last_url = url
                return FakeResponse()

        session = FakeSession()
        substance = await fetch_substance_by_uuid(test_uuid, session)

        self.assertIsNotNone(substance)
        if substance is not None:
            self.assertEqual(substance["uuid"], test_uuid)
            self.assertEqual(substance["substanceClass"], "chemical")

    async def test_fetch_invalid_substance(self):
        """Test fetching an invalid substance UUID."""
        invalid_uuid = "00000000-0000-0000-0000-000000000000"

        class FakeResponse:
            status_code = 404

            @staticmethod
            def json():
                return {}

        class FakeSession:
            async def get(self, url, timeout=30.0):
                return FakeResponse()

        session = FakeSession()
        substance = await fetch_substance_by_uuid(invalid_uuid, session)

        self.assertIsNone(substance)

    async def test_fetch_multiple_substances_parallel(self):
        """Test fetching multiple substances in parallel."""
        import asyncio

        test_uuids = [
            "0103a288-6eb6-4ced-b13a-849cd7edf028",
            "80edf0eb-b6c5-4a9a-adde-28c7254046d9",
        ]

        class FakeResponse:
            def __init__(self, payload, status_code=200):
                self._payload = payload
                self.status_code = status_code

            def json(self):
                return self._payload

        class FakeSession:
            async def get(self, url, timeout=30.0):
                substance_uuid = url.split("(")[-1].split(")")[0]
                return FakeResponse({"uuid": substance_uuid})

        session = FakeSession()
        tasks = [fetch_substance_by_uuid(uuid, session) for uuid in test_uuids]
        results = await asyncio.gather(*tasks)

        valid_results = [r for r in results if r is not None]
        self.assertEqual(len(valid_results), 2)


class TestFetchAllSubstanceUuids(unittest.IsolatedAsyncioTestCase):
    """Tests for fetching all substance UUIDs."""

    async def test_fetch_uuids_limited(self):
        """Test fetching limited number of UUIDs."""

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "results": [
                        {"uuid": "00000000-0000-0000-0000-000000000001"},
                        {"uuid": "00000000-0000-0000-0000-000000000002"},
                        {"uuid": "00000000-0000-0000-0000-000000000003"},
                    ]
                }

        class FakeSession:
            async def get(self, url, params=None, timeout=30.0):
                return FakeResponse()

        uuids = await fetch_all_substance_uuids(FakeSession(), max_results=10)

        self.assertLessEqual(len(uuids), 10)
        self.assertTrue(all(isinstance(u, str) for u in uuids))

    async def test_fetch_uuids_format(self):
        """Test that fetched UUIDs have correct format."""
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "results": [
                        {"uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028"},
                        {"uuid": "80edf0eb-b6c5-4a9a-adde-28c7254046d9"},
                    ]
                }

        class FakeSession:
            async def get(self, url, params=None, timeout=30.0):
                return FakeResponse()

        uuids = await fetch_all_substance_uuids(FakeSession(), max_results=5)

        for uuid in uuids:
            self.assertRegex(uuid, uuid_pattern)

    async def test_load_substances_from_api_disables_certificate_validation(self):
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

        captured = {}

        def fake_ingest_batch(substances, api_url, timeout=300, verify_ssl=True):
            captured["verify_ssl"] = verify_ssl
            return {
                "successful": len(substances),
                "failed": 0,
                "total_chunks": len(substances),
                "errors": [],
            }

        with patch("scripts.load_data.httpx.AsyncClient", FakeAsyncClient), patch(
            "scripts.load_data.httpx.Client", FakeSyncClient
        ), patch("scripts.load_data.ingest_batch", fake_ingest_batch):
            result = await load_substances_from_api(
                ["uuid-1", "uuid-2"],
                batch_size=2,
                api_url="https://gateway.example",
                verify_ssl=False,
            )

        self.assertEqual(result["downloaded"], 2)
        self.assertEqual(result["successful"], 2)
        self.assertEqual(seen["async_verify"], [False])
        self.assertEqual(seen["sync_verify"], [False])
        self.assertFalse(captured["verify_ssl"])


if __name__ == "__main__":
    unittest.main()
