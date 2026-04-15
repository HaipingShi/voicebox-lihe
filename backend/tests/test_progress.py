"""
Test script to debug model download progress tracking.
"""

import asyncio
import json
import logging
import time
from typing import List, Dict

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

from utils.progress import ProgressManager, get_progress_manager
from utils.hf_progress import HFProgressTracker, create_hf_progress_callback


def test_progress_manager_basic():
    """Test 1: Basic ProgressManager functionality."""
    logger.info("=" * 60)
    logger.info("Test 1: ProgressManager Basic Operations")
    logger.info("=" * 60)

    pm = ProgressManager()

    # Test update_progress
    pm.update_progress(
        model_name="test-model",
        current=50,
        total=100,
        filename="test.bin",
        status="downloading"
    )

    # Test get_progress
    progress = pm.get_progress("test-model")
    logger.info("Progress stored: %s", progress)
    assert progress is not None
    assert progress["progress"] == 50.0
    assert progress["filename"] == "test.bin"
    assert progress["status"] == "downloading"

    # Test mark_complete
    pm.mark_complete("test-model")
    progress = pm.get_progress("test-model")
    logger.info("Marked complete: %s", progress)
    assert progress["status"] == "complete"
    assert progress["progress"] == 100.0

    logger.info("Test 1 PASSED")
    return True


async def test_progress_manager_sse():
    """Test 2: ProgressManager SSE streaming."""
    logger.info("=" * 60)
    logger.info("Test 2: ProgressManager SSE Streaming")
    logger.info("=" * 60)

    pm = ProgressManager()
    collected_events: List[Dict] = []

    # Simulate SSE client
    async def sse_client():
        """Simulates a frontend SSE connection."""
        logger.info("SSE client: Subscribing to test-model-sse...")
        async for event in pm.subscribe("test-model-sse"):
            # Parse SSE event
            if event.startswith("data: "):
                data = json.loads(event[6:])
                logger.info(
                    "SSE client: Received event: %s - %s%%",
                    data['status'],
                    f"{data.get('progress', 0):.1f}",
                )
                collected_events.append(data)

                # Stop when complete
                if data.get("status") in ("complete", "error"):
                    break
            elif event.startswith(": heartbeat"):
                logger.debug("SSE client: Received heartbeat")

    # Simulate download progress updates (from backend thread)
    async def simulate_download():
        """Simulates backend sending progress updates."""
        logger.info("Backend: Starting simulated download...")
        await asyncio.sleep(0.2)  # Let SSE client subscribe first

        # Send progress updates
        for i in range(0, 101, 20):
            logger.info("Backend: Updating progress to %d%%", i)
            pm.update_progress(
                model_name="test-model-sse",
                current=i,
                total=100,
                filename=f"file_{i}.bin",
                status="downloading" if i < 100 else "downloading"
            )
            await asyncio.sleep(0.1)

        # Mark complete
        logger.info("Backend: Marking download complete")
        pm.mark_complete("test-model-sse")

    # Run SSE client and download simulation concurrently
    await asyncio.gather(
        sse_client(),
        simulate_download()
    )

    # Verify we got events
    logger.info("Collected %d events", len(collected_events))
    assert len(collected_events) > 0, "Should have received at least one event"
    assert collected_events[-1]["status"] == "complete", "Last event should be 'complete'"

    logger.info("Test 2 PASSED")
    return True


def test_hf_progress_tracker():
    """Test 3: HFProgressTracker tqdm patching."""
    logger.info("=" * 60)
    logger.info("Test 3: HFProgressTracker tqdm Patching")
    logger.info("=" * 60)

    captured_progress: List[tuple] = []

    def progress_callback(downloaded: int, total: int, filename: str):
        """Capture progress updates."""
        captured_progress.append((downloaded, total, filename))
        logger.debug("Progress callback: %d/%d bytes (%s)", downloaded, total, filename)

    tracker = HFProgressTracker(progress_callback)

    # Simulate a download with tqdm
    with tracker.patch_download():
        try:
            from tqdm import tqdm

            # Simulate downloading a file
            logger.info("Simulating download with tqdm...")
            total_size = 1000
            with tqdm(total=total_size, desc="model.bin", unit="B", unit_scale=True) as pbar:
                for chunk in range(0, total_size, 100):
                    pbar.update(100)
                    time.sleep(0.01)

            logger.info("Captured %d progress updates", len(captured_progress))
            assert len(captured_progress) > 0, "Should have captured progress updates"

            # Verify progress increases
            last_downloaded = 0
            for downloaded, total, filename in captured_progress:
                assert downloaded >= last_downloaded, "Downloaded bytes should increase"
                assert total == total_size, "Total should be consistent"
                last_downloaded = downloaded

            logger.info("Test 3 PASSED")
            return True

        except ImportError:
            logger.warning("tqdm not available, skipping test")
            return None


async def test_full_integration():
    """Test 4: Full integration test."""
    logger.info("=" * 60)
    logger.info("Test 4: Full Integration (ProgressManager + HFProgressTracker)")
    logger.info("=" * 60)

    pm = get_progress_manager()
    collected_events: List[Dict] = []

    # SSE client
    async def sse_client():
        logger.info("SSE client: Subscribing...")
        async for event in pm.subscribe("integration-test"):
            if event.startswith("data: "):
                data = json.loads(event[6:])
                logger.info(
                    "SSE client: %s - %s%% - %s",
                    data['status'],
                    f"{data.get('progress', 0):.1f}",
                    data.get('filename', ''),
                )
                collected_events.append(data)
                if data.get("status") in ("complete", "error"):
                    break

    # Simulate backend download with HFProgressTracker
    async def simulate_real_download():
        await asyncio.sleep(0.2)  # Let SSE subscribe

        logger.info("Backend: Starting download with HFProgressTracker...")

        # Set up tracking (like the real backend does)
        progress_callback = create_hf_progress_callback("integration-test", pm)
        tracker = HFProgressTracker(progress_callback)

        # Initialize progress
        pm.update_progress(
            model_name="integration-test",
            current=0,
            total=1,
            filename="",
            status="downloading"
        )

        # Simulate download with tqdm patching
        with tracker.patch_download():
            try:
                from tqdm import tqdm

                # Simulate multi-file download (like HuggingFace does)
                files = [
                    ("model.safetensors", 5000),
                    ("config.json", 1000),
                    ("tokenizer.json", 500),
                ]

                for filename, size in files:
                    logger.info("Backend: Downloading %s...", filename)
                    with tqdm(total=size, desc=filename, unit="B") as pbar:
                        for chunk in range(0, size, 500):
                            chunk_size = min(500, size - chunk)
                            pbar.update(chunk_size)
                            await asyncio.sleep(0.05)

                # Mark complete
                logger.info("Backend: Download complete")
                pm.mark_complete("integration-test")

            except ImportError:
                logger.error("tqdm not available")
                pm.mark_error("integration-test", "tqdm not available")

    # Run both
    await asyncio.gather(
        sse_client(),
        simulate_real_download()
    )

    # Verify
    logger.info("Collected %d events", len(collected_events))
    if len(collected_events) > 0:
        logger.info("First event: %s", collected_events[0])
        logger.info("Last event: %s", collected_events[-1])
        assert collected_events[-1]["status"] == "complete", "Should end with 'complete'"
        logger.info("Test 4 PASSED")
        return True
    else:
        logger.error("Test 4 FAILED - No events received")
        return False


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Voicebox Progress Tracking Test Suite")
    logger.info("=" * 60)

    results = []

    # Test 1: Basic operations
    try:
        results.append(("Basic Operations", test_progress_manager_basic()))
    except Exception as e:
        logger.error("Test 1 FAILED: %s", e)
        results.append(("Basic Operations", False))

    # Test 2: SSE streaming
    try:
        results.append(("SSE Streaming", await test_progress_manager_sse()))
    except Exception as e:
        logger.error("Test 2 FAILED: %s", e)
        results.append(("SSE Streaming", False))

    # Test 3: tqdm patching
    try:
        results.append(("tqdm Patching", test_hf_progress_tracker()))
    except Exception as e:
        logger.error("Test 3 FAILED: %s", e)
        results.append(("tqdm Patching", False))

    # Test 4: Full integration
    try:
        results.append(("Full Integration", await test_full_integration()))
    except Exception as e:
        logger.error("Test 4 FAILED: %s", e)
        results.append(("Full Integration", False))

    # Summary
    logger.info("=" * 60)
    logger.info("Test Results Summary")
    logger.info("=" * 60)

    for name, result in results:
        status = "PASS" if result else ("SKIP" if result is None else "FAIL")
        logger.info("  %s\t%s", status, name)

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)

    logger.info("")
    logger.info("  Total: %d tests", len(results))
    logger.info("  Passed: %d", passed)
    logger.info("  Failed: %d", failed)
    logger.info("  Skipped: %d", skipped)
    logger.info("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
