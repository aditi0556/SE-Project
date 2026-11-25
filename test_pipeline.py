import asyncio
import json
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

from agent import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

APP_NAME = "sdlc"
USER_ID = "aditi"

SAMPLE_INPUT = (
    "We are developing a cloud-based remote patient monitoring system that "
    "collects real-time vital signs (heart rate, blood pressure, ECG, SpO2) "
    "from wearable medical sensors, transmits them securely, stores patient health "
    "records, and alerts healthcare providers in case of emergency anomalies. "
    "It must integrate with hospital Electronic Health Record (EHR) systems via FHIR APIs, "
    "comply with HIPAA and medical device safety regulations, and guarantee high availability "
    "and zero data loss."
)


async def main():
    print("=" * 70)
    print("Starting SDLC & Compliance Multi-Agent Pipeline Test")
    print("=" * 70)
    print(f"User Input Statement:\n{SAMPLE_INPUT}\n")

    session_service = InMemorySessionService()
    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=str(uuid.uuid4()),
    )
    print(f"Created Session ID: {session.id}\n")

    message = types.Content(
        role="user",
        parts=[types.Part(text=SAMPLE_INPUT)],
    )

    print("Running pipeline through SequentialAgent...")
    max_retries = 3
    delay = 3

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"\nRetry attempt {attempt + 1}/{max_retries}...")
            async for event in runner.run_async(
                user_id=USER_ID,
                session_id=session.id,
                new_message=message,
            ):
                author = event.author

                for call in event.get_function_calls():
                    print(f"[{author}] TOOL CALL -> {call.name}({str(call.args)[:100]})")

                for resp in event.get_function_responses():
                    print(f"[{author}] TOOL RESPONSE <- {resp.name}")

                if event.is_final_response():
                    print(f"[{author}] FINISHED (result saved to session state)")
            break
        except Exception as e:
            print(f"Warning: encountered error during run: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(delay)
            delay *= 2

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 70)

    # Inspect final session state
    updated_session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session.id,
    )
    print("\nSESSION STATE KEYS:", list(updated_session.state.keys()))

    for key in ["requirements", "compliance", "sdlc_result"]:
        if key in updated_session.state:
            print(f"\n{'='*30} {key.upper()} {'='*30}")
            val = updated_session.state[key]
            if isinstance(val, (dict, list)):
                print(json.dumps(val, indent=2))
            else:
                print(val)


if __name__ == "__main__":
    asyncio.run(main())
