"""
Test Qwen TTS model download with SSE progress monitoring.

This specifically tests the MLX TTS backend download progress tracking,
which requires tqdm to be patched BEFORE mlx_audio is imported.

Usage:
    cd backend && python -m tests.test_qwen_download

Prerequisites:
    - Server must be running: cd backend && python main.py
    - Delete model first for fresh download test:
      curl -X DELETE http://localhost:8000/models/qwen-tts-0.6B
"""

import asyncio
import json
import logging
import httpx
import time
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


async def monitor_sse_stream(model_name: str, timeout: int = 600) -> List[Dict]:
    """
    Monitor SSE stream for a model download.

    Args:
        model_name: Name of the model to monitor
        timeout: Maximum time to wait for download (seconds)

    Returns:
        List of SSE events received
    """
    events: List[Dict] = []
    url = f"http://localhost:8000/models/progress/{model_name}"
    last_progress = -1

    logger.info("Connecting to SSE endpoint: %s", url)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("GET", url) as response:
                logger.info("SSE connected, status: %s", response.status_code)

                if response.status_code != 200:
                    logger.error("SSE endpoint returned %s", response.status_code)
                    return events

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            events.append(data)

                            # Print progress (only when it changes significantly)
                            progress = data.get('progress', 0)
                            status = data.get('status', 'unknown')
                            filename = data.get('filename', '')
                            current = data.get('current', 0)
                            total = data.get('total', 0)

                            # Print every 5% change or status change
                            if abs(progress - last_progress) >= 5 or status in ('complete', 'error'):
                                current_mb = current / (1024 * 1024)
                                total_mb = total / (1024 * 1024)
                                logger.info(
                                    "%s %s%% (%.1fMB / %.1fMB) %s",
                                    status,
                                    f"{progress:6.1f}",
                                    current_mb,
                                    total_mb,
                                    filename[:50],
                                )
                                last_progress = progress

                            # Stop if complete or error
                            if status in ("complete", "error"):
                                if status == "complete":
                                    logger.info("Download complete!")
                                else:
                                    logger.error("Download error: %s", data.get('error', 'unknown'))
                                break

                        except json.JSONDecodeError as e:
                            logger.warning("Error parsing JSON: %s", e)

                    elif line.startswith(": heartbeat"):
                        # Heartbeat every 1 second, don't spam
                        pass

    except asyncio.CancelledError:
        logger.info("SSE monitor cancelled")
    except Exception as e:
        logger.error("SSE error: %s", e)

    return events


async def trigger_download(model_name: str) -> bool:
    """Trigger a model download via the API."""
    url = "http://localhost:8000/models/download"

    logger.info("Triggering download for: %s", model_name)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json={"model_name": model_name})
            result = response.json()
            logger.info("Response: %s - %s", response.status_code, result)
            return response.status_code == 200
    except Exception as e:
        logger.error("Error triggering download: %s", e)
        return False


async def delete_model(model_name: str) -> bool:
    """Delete a model from cache."""
    url = f"http://localhost:8000/models/{model_name}"

    logger.info("Deleting model: %s", model_name)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.delete(url)
            if response.status_code == 200:
                logger.info("Model deleted")
                return True
            elif response.status_code == 404:
                logger.info("Model not found (already deleted)")
                return True
            else:
                logger.warning("Delete response: %s - %s", response.status_code, response.text)
                return False
    except Exception as e:
        logger.error("Error deleting model: %s", e)
        return False


async def check_model_status(model_name: str) -> Optional[Dict]:
    """Check the status of a model."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get("http://localhost:8000/models/status")
            if response.status_code == 200:
                data = response.json()
                for model in data.get("models", []):
                    if model["model_name"] == model_name:
                        return model
    except Exception as e:
        logger.warning("Error checking model status: %s", e)
    return None


async def check_server() -> bool:
    """Check if the server is running."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://localhost:8000/health")
            return response.status_code == 200
    except Exception:
        return False


async def main():
    logger.info("=" * 70)
    logger.info("Qwen TTS Model Download Progress Test")
    logger.info("=" * 70)
    logger.info("This test verifies that MLX TTS download progress tracking works.")
    logger.info("It specifically tests the tqdm patching for mlx_audio.tts imports.")

    # Check if server is running
    logger.info("Checking if server is running...")
    if not await check_server():
        logger.error("Server is not running on http://localhost:8000")
        logger.info("Please start the server first:")
        logger.info("   cd backend && python main.py")
        return False

    logger.info("Server is running")

    # Test model
    model_name = "qwen-tts-0.6B"  # Note: 0.6B currently maps to 1.7B on MLX

    # Check current status
    logger.info("Checking status of %s...", model_name)
    status = await check_model_status(model_name)
    if status:
        logger.info("Downloaded: %s", status.get('downloaded', False))
        logger.info("Downloading: %s", status.get('downloading', False))
        logger.info("Loaded: %s", status.get('loaded', False))
        if status.get('size_mb'):
            logger.info("Size: %.1f MB", status['size_mb'])
    else:
        logger.warning("Could not get model status")

    # Ask if user wants to delete first
    logger.info("-" * 70)
    if status and status.get('downloaded'):
        logger.info("Model is already downloaded. Delete it for a fresh download test?")
        logger.info("   [y] Yes, delete and download fresh")
        logger.info("   [n] No, just test SSE connection")
        logger.info("   [q] Quit")

        choice = input("\nChoice [y/n/q]: ").strip().lower()

        if choice == 'q':
            logger.info("Exiting...")
            return True

        if choice == 'y':
            if not await delete_model(model_name):
                logger.warning("Failed to delete model. Continue anyway? [y/n]")
                if input().strip().lower() != 'y':
                    return False
    else:
        logger.info("Model not downloaded. Will perform fresh download test.")
        input("Press Enter to continue...")

    # Run the test
    logger.info("=" * 70)
    logger.info("Starting Download Test")
    logger.info("=" * 70)

    async def run_test():
        # Start SSE monitor in background FIRST
        monitor_task = asyncio.create_task(monitor_sse_stream(model_name, timeout=600))

        # Wait for SSE to connect
        await asyncio.sleep(1)

        # Trigger download
        success = await trigger_download(model_name)

        if not success:
            logger.error("Failed to trigger download")
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            return []

        # Wait for SSE monitor to complete
        logger.info("Waiting for download to complete (this may take several minutes)...")
        events = await monitor_task

        return events

    start_time = time.time()
    events = await run_test()
    elapsed = time.time() - start_time

    # Results
    logger.info("=" * 70)
    logger.info("Test Results")
    logger.info("=" * 70)

    logger.info("Elapsed time: %.1f seconds", elapsed)
    logger.info("Total SSE events received: %d", len(events))

    if not events:
        logger.error("FAILED - No SSE events received!")
        logger.error("Possible causes:")
        logger.error("  1. SSE endpoint not working")
        logger.error("  2. tqdm not patched before mlx_audio import")
        logger.error("  3. Progress callbacks not firing")
        logger.error("  4. Model already fully downloaded")
        logger.error("Debug steps:")
        logger.error("  1. Check server logs for [DEBUG] messages")
        logger.error("  2. Look for 'tqdm patched' before 'mlx_audio.tts import'")
        logger.error(
            "  3. Delete model: curl -X DELETE http://localhost:8000/models/%s",
            model_name,
        )
        return False

    # Analyze events
    first_event = events[0]
    last_event = events[-1]

    logger.info("First event:")
    logger.info("   Status: %s", first_event.get('status'))
    logger.info("   Progress: %.1f%%", first_event.get('progress', 0))

    logger.info("Last event:")
    logger.info("   Status: %s", last_event.get('status'))
    logger.info("   Progress: %.1f%%", last_event.get('progress', 0))

    # Check for expected behaviors
    has_progress_updates = len(events) > 2
    has_increasing_progress = False
    has_complete = any(e.get('status') == 'complete' for e in events)
    has_100_percent = any(e.get('progress', 0) >= 100 for e in events)

    # Check if progress increased over time
    if len(events) >= 2:
        progress_values = [e.get('progress', 0) for e in events]
        has_increasing_progress = progress_values[-1] > progress_values[0]

    logger.info("Checks:")
    logger.info(
        "   %s Multiple progress updates received (%d events)",
        "PASS" if has_progress_updates else "FAIL",
        len(events),
    )
    logger.info(
        "   %s Progress increased over time",
        "PASS" if has_increasing_progress else "FAIL",
    )
    logger.info(
        "   %s Reached 100%% progress",
        "PASS" if has_100_percent else "FAIL",
    )
    logger.info(
        "   %s Received 'complete' status",
        "PASS" if has_complete else "FAIL",
    )

    # Overall result
    success = has_progress_updates and has_complete

    if success:
        logger.info("=" * 70)
        logger.info("TEST PASSED - Qwen TTS download progress tracking works!")
        logger.info("=" * 70)
    else:
        logger.error("=" * 70)
        logger.error("TEST FAILED - Progress tracking has issues")
        logger.error("=" * 70)
        logger.error("Check the server logs for debug output.")

    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
