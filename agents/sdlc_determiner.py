from google.adk.agents import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "sdlc_knowledge"

qdrant_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uvx",
            args=["mcp-server-qdrant"],
            env={
                "QDRANT_URL":QDRANT_URL,
                "COLLECTION_NAME":COLLECTION_NAME,
            },
        ),
        timeout=30,
    ),
)

sdlc_determiner = Agent(
    name="sdlc_determiner",
    model="gemini-flash-latest",

    instruction="""
        You are an SDLC Determining Agent.

        The Requirement Analyser has already analyzed the project.

        The generated requirements are available in session state
        under the key 'requirements'.

        IMPORTANT:
        1. You MUST call qdrant-find before deciding the SDLC model.
        2. Use the retrieved knowledge as reference for SDLC characteristics.
        3. Base your decision primarily on the ACTUAL requirements provided.
        4. Do NOT assume requirements are stable unless stability is explicitly
        stated in the project description.
        5. Do NOT invent technologies, regulations, standards, constraints,
        or requirements that were not provided.
        6. If the project has high technical/business risk, uncertain or
        evolving requirements, and requires repeated risk analysis or
        prototyping, strongly consider the Spiral model.
        7. Changing requirements alone does not automatically mean Agile.
        Evaluate the type and level of risk as well.

        Consider only:
        - Waterfall
        - Incremental
        - Spiral
        - RAD
        - Agile
        - V-Model

        Provide:
        1. Recommended SDLC model
        2. Reasons for selecting it

        The final recommendation must be based on the requirements and
        the retrieved knowledge, not on assumptions.
        """,

    tools=[qdrant_tools],
    output_key="sdlc_result",
)