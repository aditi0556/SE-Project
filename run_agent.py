import asyncio
import uuid
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
load_dotenv()
from agent import root_agent

APP_NAME = "sdlc"
USER_ID = "aditi"


session_service = InMemorySessionService()

runner = Runner(
    app_name=APP_NAME,
    agent=root_agent,
    session_service=session_service,
)


async def create_session():
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=str(uuid.uuid4()),
    )

    return session


async def run_with_retry(session_id: str, user_input: str):
    message = types.Content(
        role="user",
        parts=[
            types.Part(text=user_input)
        ],
    )

    max_retries = 5
    delay = 2

    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1}")

            async for event in runner.run_async(
                user_id=USER_ID,
                session_id=session_id,
                new_message=message,
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        print(
                            "\nFINAL RESPONSE:\n",
                            event.content.parts[0].text,
                        )

            return

        except Exception as e:

            error = str(e)

            # Retry only temporary errors
            if "503" not in error and "429" not in error:
                raise

            if attempt == max_retries - 1:
                print("Maximum retries reached.")
                raise

            print(
                f"Temporary model error: {error}"
            )

            print(
                f"Retrying in {delay} seconds..."
            )

            await asyncio.sleep(delay)

            delay *= 2


async def main():

    session = await create_session()

    print("Session created:")
    print(session.id)

    while True:

        user_input = input("\nYou: ")

        if user_input.lower() in {"exit", "quit"}:
            break

        await run_with_retry(
            session_id=session.id,
            user_input=user_input,
        )


if __name__ == "__main__":
    asyncio.run(main())