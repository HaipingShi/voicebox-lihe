"""
Test real model download with SSE progress monitoring.
"""

import asyncio
import json
import logging
import httpx
import time
from typing import List, Dict

logger = logging.getLogger(__name__)


async def monitor_sse_stream(model_name: str, timeout: int = 300):
    """Monitor SSE stream for a model download."""
    events: List[Dict] = []
    url = f"http://localhost:8000/models/progress/{model_name}"

    logger.info("Connecting to SSE endpoint: %s", url)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("GET", url) as response:
            logger.info("SSE connected, status: %s", response.status_code)

            if response.status_code != 200:
                logger.error("SSE endpoint returned %s", response.status_code)
                return events

            async for line in response.aiter_lines():
                if not line:
                    continue

                logger.debug("Raw SSE: %s...", line[:100])

                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        logger.info(
                            "%s %s%% %s",
                            data['status'],
                            f"{data.get('progress', 0):6.1f}",
                            data.get('filename', ''),
                        )
                        events.append(data)

                        # Stop if complete or error
                        if data.get("status") in ("complete", "error"):
                            logger.info("Download %s!", data['status'])
                            break

                    except json.JSONDecodeError as e:
                        logger.error("Error parsing JSON: %s", e)
                        logger.debug("Line was: %s", line)

                elif line.startswith(": heartbeat"):
                    logger.debug("heartbeat")

    return events


async def trigger_download(model_name: str):
    """Trigger a model download via the API."""
    url = "http://localhost:8000/models/download"

    logger.info("Triggering download for: %s", model_name)

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(url, json={"model_name": model_name})
        logger.info("Response: %s - %s", response.status_code, response.json())
        return response.status_code == 200


async def check_server():
    """Check if the server is running."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://localhost:8000/health")
            return response.status_code == 200
    except Exception as e:
        logger.error("Server not running: %s", e)
        return False


async def main():
    logger.info("=" * 60)
    logger.info("Real Model Download Progress Test")
    logger.info("=" * 60)

    # Check if server is running
    logger.info("Checking if server is running...")
    if not await check_server():
        logger.error("Server is not running on http://localhost:8000")
        logger.info("Please start the server first:")
        logger.info("  cd backend && python main.py")
        return False

    logger.info("Server is running")

    # Choose a small model for testing
    model_name = "whisper-base"  # ~150MB, faster to download
    logger.info("Using model: %s", model_name)

    # Option to delete model first if it exists
    logger.info("Do you want to delete the model first to force a fresh download? (y/n)")
    # For automated testing, skip deletion prompt
    # delete_first = input().strip().lower() == 'y'
    delete_first = False

    if delete_first:
        logger.info("Deleting %s...", model_name)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.delete(f"http://localhost:8000/models/{model_name}")
            logger.info("Delete response: %s", response.status_code)

    logger.info("=" * 60)
    logger.info("Starting Test")
    logger.info("=" * 60)

    # Start monitoring SSE stream BEFORE triggering download
    async def run_test():
        # Start SSE monitor in background
        monitor_task = asyncio.create_task(monitor_sse_stream(model_name))

        # Wait a bit to ensure SSE is connected
        await asyncio.sleep(1)

        # Trigger download
        success = await trigger_download(model_name)

        if not success:
            logger.error("Failed to trigger download")
            monitor_task.cancel()
            return False

        # Wait for SSE monitor to complete
        events = await monitor_task

        return events

    events = await run_test()

    # Results
    logger.info("=" * 60)
    logger.info("Test Results")
    logger.info("=" * 60)

    if not events:
        logger.error("FAILED - No SSE events received!")
        logger.error("Possible causes:")
        logger.error("  1. SSE endpoint not working")
        logger.error("  2. Progress updates not being sent")
        logger.error("  3. Model already downloaded (no progress to report)")
        logger.info("Try deleting the model first to force a fresh download:")
        logger.info("  curl -X DELETE http://localhost:8000/models/%s", model_name)
        return False

    logger.info("Received %d SSE events", len(events))
    logger.info("First event: %s", events[0])
    logger.info("Last event: %s", events[-1])

    # Check if we got meaningful progress
    has_progress = any(e.get('progress', 0) > 0 for e in events)
    has_complete = any(e.get('status') == 'complete' for e in events)

    if has_progress:
        logger.info("Progress updates received")
    else:
        logger.warning("No progress updates (might be already downloaded)")

    if has_complete:
        logger.info("Download completed successfully")
    else:
        logger.warning("Download did not complete")

    success = has_progress and has_complete

    if success:
        logger.info("TEST PASSED - Progress tracking works!")
    else:
        logger.warning("TEST INCONCLUSIVE - Try with a fresh download")

    return success


if __name__ == "__main__":
    asyncio.run(main())
