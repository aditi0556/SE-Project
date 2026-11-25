import asyncio
import json
import os
from pathlib import Path
from typing import AsyncGenerator
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn

load_dotenv()

from agent import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

app = FastAPI(title="SDLC Multi-Agent Analysis System")

APP_NAME = "sdlc"
USER_ID = "aditi"
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"


@app.get("/")
async def serve_frontend():
    return FileResponse(TEMPLATES_DIR / "index.html")


async def pipeline_stream_generator(user_input: str) -> AsyncGenerator[str, None]:
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

    yield f"data: {json.dumps({'type': 'status', 'message': f'Session created: {session.id}. Running agents...'})}\n\n"

    message = types.Content(
        role="user",
        parts=[types.Part(text=user_input)],
    )

    last_started_author = None
    dispatched_keys = set()

    try:
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session.id,
            new_message=message,
        ):
            author = getattr(event, "author", None)

            # Signal agent start
            if author and author != last_started_author:
                last_started_author = author
                yield f"data: {json.dumps({'type': 'agent_start', 'agent': author})}\n\n"

            # Check if this agent produced its final response
            if event.is_final_response():
                updated_session = await session_service.get_session(
                    app_name=APP_NAME,
                    user_id=USER_ID,
                    session_id=session.id,
                )

                # Requirement Analyser
                if "requirements" in updated_session.state and "requirements" not in dispatched_keys:
                    dispatched_keys.add("requirements")
                    yield f"data: {json.dumps({'type': 'agent_result', 'agent': 'requirement_analyser', 'key': 'requirements', 'data': updated_session.state['requirements']})}\n\n"

                # Compliance Agent
                if "compliance" in updated_session.state and "compliance" not in dispatched_keys:
                    dispatched_keys.add("compliance")
                    yield f"data: {json.dumps({'type': 'agent_result', 'agent': 'compliance_agent', 'key': 'compliance', 'data': updated_session.state['compliance']})}\n\n"

                # SDLC Determiner
                if "sdlc_result" in updated_session.state and "sdlc_result" not in dispatched_keys:
                    dispatched_keys.add("sdlc_result")
                    yield f"data: {json.dumps({'type': 'agent_result', 'agent': 'sdlc_determiner', 'key': 'sdlc_result', 'data': updated_session.state['sdlc_result']})}\n\n"

        # Final check for any keys not yet dispatched
        updated_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session.id,
        )

        for key, agent_name in [
            ("requirements", "requirement_analyser"),
            ("compliance", "compliance_agent"),
            ("sdlc_result", "sdlc_determiner"),
        ]:
            if key in updated_session.state and key not in dispatched_keys:
                dispatched_keys.add(key)
                yield f"data: {json.dumps({'type': 'agent_result', 'agent': agent_name, 'key': key, 'data': updated_session.state[key]})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@app.post("/api/analyze")
async def analyze_endpoint(request: Request):
    payload = await request.json()
    statement = payload.get("statement", "").strip()
    if not statement:
        return StreamingResponse(
            (chunk for chunk in [f"data: {json.dumps({'type': 'error', 'message': 'Statement cannot be empty'})}\n\n"]),
            media_type="text/event-stream",
        )

    return StreamingResponse(
        pipeline_stream_generator(statement),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
