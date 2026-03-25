#!/usr/bin/env python3
"""
GSRS RAG Gateway - Data Loader Script

Loads substances into the RAG database from multiple sources:
1. *.gsrs files (JSONL.gz format with two leading tabs per line)
2. UUID list - load specific substances from GSRS server
3. No parameters - load all substances from GSRS server (batch download)

Usage:
    # Load from .gsrs file
    python scripts/load_data.py data/substances.gsrs --batch-size 100

    # Load specific substances by UUID from GSRS server
    python scripts/load_data.py --uuids 0103a288-6eb6-4ced-b13a-849cd7edf028,80edf0eb-b6c5-4a9a-adde-28c7254046d9

    # Load all substances from GSRS server
    python scripts/load_data.py --all

    # Dry run (parse only, don't upload)
    python scripts/load_data.py data/substances.gsrs --dry-run

    # Disable TLS certificate validation for HTTPS endpoints
    python scripts/load_data.py --all --insecure
"""

import argparse
import asyncio
import gzip
import httpx
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GSRS API Configuration
GSRS_BASE_URL = "https://gsrs.ncats.nih.gov/api/v1"
GSRS_SEARCH_URL = f"{GSRS_BASE_URL}/substances/search"
GSRS_SUBSTANCE_URL = f"{GSRS_BASE_URL}/substances"


def create_client(timeout: float, verify_ssl: bool = True) -> httpx.Client:
    """Create a sync HTTP client with optional certificate validation."""
    return httpx.Client(timeout=timeout, verify=verify_ssl)


def create_async_client(timeout: float, verify_ssl: bool = True) -> httpx.AsyncClient:
    """Create an async HTTP client with optional certificate validation."""
    return httpx.AsyncClient(timeout=timeout, verify=verify_ssl)


def parse_gsrs_file(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """
    Parse a .gsrs file (JSONL.gz format).

    Each line starts with two tab characters.

    Args:
        file_path: Path to the .gsrs file

    Yields:
        Substance JSON documents
    """
    logger.info(f"Opening file: {file_path}")

    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        line_number = 0

        for line in f:
            line_number += 1

            # Strip leading tabs (two tab characters as per spec)
            stripped_line = line.lstrip('\t\t')

            if not stripped_line.strip():
                continue

            try:
                substance = json.loads(stripped_line)
                yield substance

                if line_number % 1000 == 0:
                    logger.info(f"Processed {line_number} lines...")

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_number}: {e}")
                continue

    logger.info(f"Finished parsing {line_number} lines")


async def fetch_substance_by_uuid(
    uuid: str,
    session: httpx.AsyncClient
) -> Optional[Dict[str, Any]]:
    """
    Fetch a single substance from GSRS API by UUID.

    Args:
        uuid: Substance UUID
        session: Async HTTP client session

    Returns:
        Substance JSON document or None if failed
    """
    url = f"{GSRS_SUBSTANCE_URL}({uuid})?view=full"
    try:
        response = await session.get(url, timeout=30.0)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to fetch {uuid}: HTTP {response.status_code}")
            return None
    except Exception as e:
        logger.warning(f"Failed to fetch {uuid}: {e}")
        return None


async def fetch_all_substance_uuids(
    session: httpx.AsyncClient,
    max_results: int = 10000
) -> List[str]:
    """
    Fetch all substance UUIDs from GSRS API.

    Args:
        session: Async HTTP client session
        max_results: Maximum number of results to fetch

    Returns:
        List of substance UUIDs
    """
    uuids = []
    page = 1
    page_size = 1000

    while len(uuids) < max_results:
        url = f"{GSRS_SEARCH_URL}"
        params = {
            "page": page,
            "size": page_size,
            "fields": "uuid"
        }

        try:
            response = await session.get(url, params=params, timeout=30.0)
            if response.status_code != 200:
                logger.warning(f"Search page {page} failed: HTTP {response.status_code}")
                break

            data = response.json()
            results = data.get("results", [])

            if not results:
                break

            for item in results:
                uuid = item.get("uuid")
                if uuid:
                    uuids.append(uuid)

            logger.info(f"Fetched page {page}, total UUIDs: {len(uuids)}")

            # Check if there are more pages
            if len(results) < page_size:
                break

            page += 1

        except Exception as e:
            logger.warning(f"Error fetching page {page}: {e}")
            break

    logger.info(f"Total UUIDs fetched: {len(uuids)}")
    return uuids


async def load_substances_from_api(
    uuids: List[str],
    batch_size: int = 100,
    api_url: str = "http://localhost:8000",
    dry_run: bool = False,
    verify_ssl: bool = True,
) -> Dict[str, Any]:
    """
    Load substances from GSRS API and ingest to RAG Gateway.

    Args:
        uuids: List of substance UUIDs to load
        batch_size: Batch size for ingestion
        api_url: RAG Gateway API URL
        dry_run: If True, only download without uploading
        verify_ssl: Whether to verify TLS certificates for HTTP requests

    Returns:
        Summary statistics
    """
    stats = {
        "total_substances": 0,
        "downloaded": 0,
        "successful": 0,
        "failed": 0,
        "total_chunks": 0,
        "errors": []
    }

    # Download substances from GSRS
    logger.info(f"Downloading {len(uuids)} substances from GSRS API...")

    async with create_async_client(timeout=30.0, verify_ssl=verify_ssl) as session:
        tasks = [fetch_substance_by_uuid(uuid, session) for uuid in uuids]
        results = await asyncio.gather(*tasks)

    substances = [s for s in results if s is not None]
    stats["downloaded"] = len(substances)
    stats["total_substances"] = len(uuids)

    logger.info(f"Downloaded {len(substances)}/{len(uuids)} substances")

    if dry_run:
        stats["successful"] = len(substances)
        return stats

    # Check API availability
    try:
        with create_client(timeout=10, verify_ssl=verify_ssl) as client:
            health_response = client.get(f"{api_url}/health")
            health_response.raise_for_status()
            health = health_response.json()
            logger.info(f"API Health: {health['status']}")
    except Exception as e:
        logger.error(f"API not available: {e}")
        stats["errors"].append(f"API not available: {e}")
        return stats

    # Ingest in batches
    batch = []
    for substance in substances:
        batch.append(substance)

        if len(batch) >= batch_size:
            result = ingest_batch(batch, api_url, verify_ssl=verify_ssl)
            stats["successful"] += result.get("successful", 0)
            stats["failed"] += result.get("failed", 0)
            stats["total_chunks"] += result.get("total_chunks", 0)
            stats["errors"].extend(result.get("errors", []))
            logger.info(
                f"Batch complete: {result.get('successful', 0)} successful, "
                f"{result.get('total_chunks', 0)} chunks"
            )
            batch = []

    # Process remaining
    if batch:
        result = ingest_batch(batch, api_url, verify_ssl=verify_ssl)
        stats["successful"] += result.get("successful", 0)
        stats["failed"] += result.get("failed", 0)
        stats["total_chunks"] += result.get("total_chunks", 0)
        stats["errors"].extend(result.get("errors", []))

    return stats


def ingest_batch(
    substances: List[Dict[str, Any]],
    api_url: str,
    timeout: int = 300,
    verify_ssl: bool = True,
) -> Dict[str, Any]:
    """
    Load substances to the RAG Gateway API.

    Args:
        substances: List of substance documents
        api_url: Base URL of the RAG Gateway API
        timeout: Request timeout in seconds
        verify_ssl: Whether to verify TLS certificates for HTTP requests

    Returns:
        Response from the API
    """
    endpoint = f"{api_url}/ingest/batch"

    try:
        with create_client(timeout=timeout, verify_ssl=verify_ssl) as client:
            response = client.post(endpoint, json={"substances": substances})
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {
            "successful": 0,
            "failed": len(substances),
            "total_chunks": 0,
            "errors": [str(e)]
        }


def load_from_file(
    file_path: str,
    batch_size: int = 100,
    api_url: str = "http://localhost:8000",
    dry_run: bool = False,
    verify_ssl: bool = True,
) -> Dict[str, Any]:
    """
    Load substances from a .gsrs file.

    Args:
        file_path: Path to the .gsrs file
        batch_size: Batch size for ingestion
        api_url: RAG Gateway API URL
        dry_run: If True, only parse without uploading
        verify_ssl: Whether to verify TLS certificates for HTTP requests

    Returns:
        Summary statistics
    """
    stats = {
        "total_substances": 0,
        "successful": 0,
        "failed": 0,
        "total_chunks": 0,
        "errors": []
    }

    # Check API availability if not dry run
    if not dry_run:
        try:
            with create_client(timeout=10, verify_ssl=verify_ssl) as client:
                health_response = client.get(f"{api_url}/health")
                health_response.raise_for_status()
                health = health_response.json()
                logger.info(f"API Health: {health['status']}")
                if not health['database_connected']:
                    logger.error("Database is not connected!")
                    stats["errors"].append("Database not connected")
                    return stats
        except httpx.HTTPError as e:
            logger.error(f"API not available: {e}")
            stats["errors"].append(f"API not available: {e}")
            return stats

    # Parse and load substances
    batch = []

    logger.info(f"Starting load with batch size: {batch_size}")

    for substance in parse_gsrs_file(file_path):
        batch.append(substance)

        if len(batch) >= batch_size:
            stats["total_substances"] += len(batch)

            if dry_run:
                logger.info(f"[DRY RUN] Would process batch of {len(batch)} substances")
                stats["successful"] += len(batch)
            else:
                result = ingest_batch(batch, api_url, verify_ssl=verify_ssl)
                stats["successful"] += result.get("successful", 0)
                stats["failed"] += result.get("failed", 0)
                stats["total_chunks"] += result.get("total_chunks", 0)
                stats["errors"].extend(result.get("errors", []))
                logger.info(
                    f"Batch complete: {result.get('successful', 0)} successful, "
                    f"{result.get('total_chunks', 0)} chunks"
                )

            batch = []

    # Process remaining
    if batch:
        stats["total_substances"] += len(batch)

        if dry_run:
            logger.info(f"[DRY RUN] Would process final batch of {len(batch)} substances")
            stats["successful"] += len(batch)
        else:
            result = ingest_batch(batch, api_url, verify_ssl=verify_ssl)
            stats["successful"] += result.get("successful", 0)
            stats["failed"] += result.get("failed", 0)
            stats["total_chunks"] += result.get("total_chunks", 0)
            stats["errors"].extend(result.get("errors", []))

    return stats


def print_summary(stats: Dict[str, Any]):
    """Print load summary."""
    print("\n" + "=" * 60)
    print("LOAD SUMMARY")
    print("=" * 60)

    if "downloaded" in stats:
        print(f"Substances downloaded: {stats['downloaded']}/{stats['total_substances']}")

    print(f"Total substances processed: {stats['total_substances']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total chunks created: {stats['total_chunks']}")

    if stats["errors"]:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats["errors"][:10]:
            print(f"  - {error}")
        if len(stats["errors"]) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Load GSRS substances into RAG Gateway"
    )

    # Input source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "file",
        nargs="?",
        type=str,
        help="Path to the .gsrs file (JSONL.gz format)"
    )
    source_group.add_argument(
        "--uuids",
        type=str,
        help="Comma-separated list of substance UUIDs to load from GSRS server"
    )
    source_group.add_argument(
        "--all",
        action="store_true",
        help="Load all substances from GSRS server"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of substances to process per batch (default: 100)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="RAG Gateway API URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only parse/download, don't upload to API"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10000,
        help="Maximum number of substances to fetch when using --all (default: 10000)"
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate validation for GSRS and RAG Gateway HTTPS requests"
    )

    args = parser.parse_args()
    verify_ssl = not args.insecure

    # Validate input
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            sys.exit(1)

    # Run appropriate loader
    if args.file:
        # Load from file
        stats = load_from_file(
            args.file,
            batch_size=args.batch_size,
            api_url=args.api_url,
            dry_run=args.dry_run,
            verify_ssl=verify_ssl,
        )
    elif args.uuids:
        # Load specific UUIDs from API
        uuid_list = [u.strip() for u in args.uuids.split(",") if u.strip()]
        if not uuid_list:
            logger.error("No valid UUIDs provided")
            sys.exit(1)

        logger.info(f"Loading {len(uuid_list)} substances from GSRS API...")
        stats = asyncio.run(load_substances_from_api(
            uuid_list,
            batch_size=args.batch_size,
            api_url=args.api_url,
            dry_run=args.dry_run,
            verify_ssl=verify_ssl,
        ))
    elif args.all:
        # Load all substances from API
        logger.info(f"Fetching all substance UUIDs from GSRS API (max: {args.max_results})...")

        async def fetch_and_load():
            async with create_async_client(timeout=30.0, verify_ssl=verify_ssl) as session:
                uuids = await fetch_all_substance_uuids(session, args.max_results)

            if not uuids:
                logger.error("No UUIDs fetched")
                return {
                    "total_substances": 0,
                    "downloaded": 0,
                    "successful": 0,
                    "failed": 0,
                    "total_chunks": 0,
                    "errors": ["No UUIDs fetched"]
                }

            return await load_substances_from_api(
                uuids,
                batch_size=args.batch_size,
                api_url=args.api_url,
                dry_run=args.dry_run,
                verify_ssl=verify_ssl,
            )

        stats = asyncio.run(fetch_and_load())
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Load from .gsrs file")
        print("  python scripts/load_data.py data/substances.gsrs")
        print()
        print("  # Load specific substances by UUID")
        print("  python scripts/load_data.py --uuids 0103a288-6eb6-4ced-b13a-849cd7edf028,80edf0eb-b6c5-4a9a-adde-28c7254046d9")
        print()
        print("  # Load all substances from GSRS server")
        print("  python scripts/load_data.py --all")
        print()
        print("  # Disable TLS certificate validation")
        print("  python scripts/load_data.py --all --insecure")
        sys.exit(1)

    print_summary(stats)

    if stats["failed"] > 0 or stats["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
