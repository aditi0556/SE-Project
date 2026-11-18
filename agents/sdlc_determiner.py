from google.adk.agents import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from config import get_model
from pydantic import BaseModel
from typing import List


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "sdlc_knowledge"

class SDLCOutput(BaseModel):
    selected_sdlc: str
    rationale: str
    matched_factors: List[str]
    retrieved_knowledge: List[str]

qdrant_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="/home/aditi-sinha/.local/bin/uvx",
            args=["mcp-server-qdrant"],
            env={
                "PATH": "/home/aditi-sinha/.local/bin:/usr/local/bin:/usr/bin:/bin",
                "QDRANT_URL": QDRANT_URL,
                "COLLECTION_NAME": COLLECTION_NAME,
            },
        ),
        timeout=30,
    ),
)

sdlc_determiner = Agent(
    name="sdlc_determiner",
    model=get_model(),

    instruction="""
    You are an SDLC Determining Agent.

    The Requirement Analyser and Compliance Agent have already analyzed the project.

    The generated requirements are available in session state under the key 'requirements'.
    The compliance verification results are available in session state under the key 'compliance'.

    IMPORTANT:

    1. You MUST call qdrant-find before deciding the SDLC model.

    2. Use the retrieved knowledge as reference for the characteristics
       of the SDLC models (including Waterfall, V-Model, Spiral, Agile, RAD, and Incremental and prototype models).

    3. Compare the retrieved SDLC knowledge against the sdlc_factors
       produced by the Requirement Analyser AND the compliance verification results
       (including verification status, compliance notes, and safety/regulatory constraints).

    4. Consider:
       - Requirement stability
       - Requirement uncertainty
       - Technical uncertainty
       - Risk level
       - Safety and regulatory criticality
       - Regulatory constraints and mandatory approval checkpoints
       - Need for user feedback
       - Need for prototyping
       - Integration complexity

    5. Do NOT select a model merely because of the domain label.

    7. Do NOT invent requirements, technologies, regulations, or
       constraints that were not provided.

    9. Select the SDLC model whose documented characteristics best
       match the project's actual factors.

    10. In matched_factors, list the project factors and compliance constraints that influenced
        the decision.

    11. In retrieved_knowledge, include the important pieces of
        knowledge retrieved from Qdrant that were used in the decision.

    Return only the structured output.
    """,

    tools=[qdrant_tools],
    output_schema=SDLCOutput,
    output_key="sdlc_result",
)