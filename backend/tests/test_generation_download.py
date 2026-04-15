"""
Test TTS generation with SSE progress monitoring.
This test captures the exact SSE events triggered during generation
to identify UX issues where users see download progress even when
the model is already cached.
"""

import asyncio
import json
import logging
import httpx
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


async def monitor_sse_stream(model_name: str, timeout: int = 120):
    """Monitor SSE stream for a model during generation."""
    events: List[Dict] = []
    url = f"http://localhost:8000/models/progress/{model_name}"

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
                            logger.info(
                                "SSE Event: %s %s%% %s",
                                data['status'],
                                f"{data.get('progress', 0):6.1f}",
                                data.get('filename', ''),
                            )
                            events.append({**data})

                            # Stop if complete or error
                            if data.get("status") in ("complete", "error"):
                                logger.info("Model %s!", data['status'])
                                break

                        except json.JSONDecodeError as e:
                            logger.error("Error parsing JSON: %s", e)
                            logger.debug("Line was: %s", line)

                    elif line.startswith(": heartbeat"):
                        logger.debug("heartbeat")

    except asyncio.TimeoutError:
        logger.error("SSE monitoring timed out")
    except Exception as e:
        logger.error("SSE error: %s", e)

    return events


async def trigger_generation(profile_id: str, text: str, model_size: str = "1.7B"):
    """Trigger TTS generation via the API."""
    url = "http://localhost:8000/generate"

    logger.info("Triggering generation...")
    logger.info("Profile: %s", profile_id)
    logger.info("Text: %s...", text[:50])
    logger.info("Model: %s", model_size)

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                url,
                json={
                    "profile_id": profile_id,
                    "text": text,
                    "language": "en",
                    "model_size": model_size,
                },
            )

            logger.info("Response: %s", response.status_code)

            if response.status_code == 200:
                result = response.json()
                logger.info("Generation successful!")
                logger.info("Generation ID: %s", result.get('id'))
                logger.info("Duration: %.2fs", result.get('duration', 0))
                return True, result
            elif response.status_code == 202:
                # Model is being downloaded
                result = response.json()
                logger.info("Model download in progress")
                logger.info("Detail: %s", result)
                return False, result
            else:
                logger.error("Error: %s", response.text)
                return False, None

    except Exception as e:
        logger.error("Exception: %s", e)
        return False, None


async def get_first_profile():
    """Get the first available voice profile."""
    url = "http://localhost:8000/profiles"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url)
            if response.status_code == 200:
                profiles = response.json()
                if profiles:
                    return profiles[0]["id"]
    except Exception as e:
        logger.error("Error getting profiles: %s", e)

    return None


async def check_server():
    """Check if the server is running."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://localhost:8000/health")
            return response.status_code == 200
    except Exception as e:
        logger.error("Server not running: %s", e)
        return False


async def test_generation_with_cached_model():
    """
    Test Case 1: Generation when model is already cached.

    This should NOT show any download progress events.
    If it does, that's the UX bug we're trying to fix.
    """
    logger.info("=" * 80)
    logger.info("TEST CASE 1: Generation with Cached Model")
    logger.info("=" * 80)
    logger.info("Expected: No download progress events (or minimal/instant completion)")
    logger.info("Actual UX Issue: Users see 'started' and 'finished' events even for cached models")
    logger.info("=" * 80)

    model_size = "1.7B"
    model_name = f"qwen-tts-{model_size}"

    # Get a profile
    profile_id = await get_first_profile()
    if not profile_id:
        logger.error("No voice profiles found. Please create a profile first.")
        return False

    logger.info("Using profile: %s", profile_id)

    # Start SSE monitor BEFORE triggering generation
    monitor_task = asyncio.create_task(monitor_sse_stream(model_name, timeout=30))

    # Wait for SSE to connect
    await asyncio.sleep(1)

    # Trigger generation
    test_text = "Hello, this is a test of the voice generation system."
    success, result = await trigger_generation(profile_id, test_text, model_size)

    if not success and result and result.get("downloading"):
        logger.warning("Model is being downloaded. Waiting for download to complete...")
        # Wait for SSE monitor to capture download events
        events = await monitor_task
        return events

    # Wait a bit more to catch any progress events
    await asyncio.sleep(3)

    # Cancel SSE monitor
    monitor_task.cancel()
    try:
        events = await monitor_task
    except asyncio.CancelledError:
        events = []

    return events


async def test_generation_with_fresh_download():
    """
    Test Case 2: Generation when model needs to be downloaded.

    This SHOULD show download progress events.
    """
    logger.info("=" * 80)
    logger.info("TEST CASE 2: Generation with Model Download")
    logger.info("=" * 80)
    logger.info("Expected: Download progress events from 0%% to 100%%")
    logger.info("=" * 80)

    # Use a different model size to force download
    model_size = "0.6B"  # Smaller model for faster testing
    model_name = f"qwen-tts-{model_size}"

    # Get a profile
    profile_id = await get_first_profile()
    if not profile_id:
        logger.error("No voice profiles found. Please create a profile first.")
        return False

    logger.info("Using profile: %s", profile_id)
    logger.info("Note: This will download the model if not cached")

    # Start SSE monitor BEFORE triggering generation
    monitor_task = asyncio.create_task(monitor_sse_stream(model_name, timeout=300))

    # Wait for SSE to connect
    await asyncio.sleep(1)

    # Trigger generation
    test_text = "This should trigger a model download if the model is not cached."
    success, result = await trigger_generation(profile_id, test_text, model_size)

    if not success and result and result.get("downloading"):
        logger.info("Model download initiated. Monitoring progress...")
        # Wait for download to complete
        events = await monitor_task

        # Try generation again
        logger.info("Retrying generation after download...")
        await asyncio.sleep(2)
        success, result = await trigger_generation(profile_id, test_text, model_size)

        if success:
            logger.info("Generation successful after download")

        return events

    # If model was already cached
    await asyncio.sleep(3)
    monitor_task.cancel()
    try:
        events = await monitor_task
    except asyncio.CancelledError:
        events = []

    return events


async def main():
    logger.info("=" * 80)
    logger.info("TTS Generation Progress Test")
    logger.info("=" * 80)
    logger.info("Purpose: Capture exact SSE events during generation to identify UX issues")
    logger.info("=" * 80)

    # Check if server is running
    logger.info("Checking if server is running...")
    if not await check_server():
        logger.error("Server is not running on http://localhost:8000")
        logger.info("Please start the server first:")
        logger.info("  cd backend && python main.py")
        return False

    logger.info("Server is running")

    # Test Case 1: Cached model
    logger.info("-" * 40)
    events_cached = await test_generation_with_cached_model()

    # Results for Test Case 1
    logger.info("=" * 80)
    logger.info("TEST CASE 1 RESULTS: Generation with Cached Model")
    logger.info("=" * 80)

    if not events_cached:
        logger.info("GOOD: No SSE progress events received")
        logger.info("This is the expected behavior for a cached model.")
    else:
        logger.warning("ISSUE FOUND: Received %d SSE events:", len(events_cached))
        logger.info("Event Timeline:")
        for i, event in enumerate(events_cached, 1):
            logger.info("  %d. %s", i, event)

        logger.warning("This explains the UX issue!")
        logger.warning("Users see progress events even when the model is already cached,")
        logger.warning("making them think the model is downloading again.")

    logger.info("=" * 80)
    logger.info("Test Complete!")
    logger.info("=" * 80)

    return True


if __name__ == "__main__":
    asyncio.run(main())
