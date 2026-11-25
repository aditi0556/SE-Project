from google.adk.agents import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from pydantic import BaseModel, Field
from typing import List
from google.adk.models.lite_llm import LiteLlm
from config import get_model
import os

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "healthcare_knowledge"

class SDLCFactors(BaseModel):
    requirement_stability: str
    requirement_uncertainty: str
    technical_uncertainty: str
    risk_level: str
    safety_criticality: str
    regulatory_constraints: str
    need_for_user_feedback: str
    need_for_prototyping: str
    integration_complexity: str


class RequirementsOutput(BaseModel):
    functional_requirements: List[str]
    non_functional_requirements: List[str]
    sdlc_factors: SDLCFactors


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

requirement_analyser = Agent(
    name="requirement_analyser",
    model=get_model(),
    description=(
        "Analyzes a user's project description and identifies "
        "functional requirements, non-functional requirements, "
        "risks, uncertainties, and requirement stability."
    ),
    instruction="""
        You are a Requirement Analyser Agent for healthcare software.

        You MUST call the Qdrant retrieval tool before generating the final output.

        Use the retrieved healthcare knowledge only to identify relevant
        healthcare-domain considerations. Do not determine or recommend an
        SDLC model.

        Analyze the project description and perform the following tasks:

        1. FUNCTIONAL REQUIREMENTS
        Identify functional requirements that are:
        - Explicitly stated in the project description, OR
        - Clearly and strongly implied by the project description.

        Do NOT invent technologies, features, integrations, workflows,
        standards, regulations, or implementation details.

        2. NON-FUNCTIONAL REQUIREMENTS
        Identify relevant non-functional requirements from:
        - Security
        - Performance
        - Reliability
        - Scalability
        - Data integrity

        Include a non-functional requirement only when it is explicitly
        stated or strongly justified by the project description or retrieved
        healthcare knowledge.

        3. SDLC-RELEVANT FACTORS
        Analyze the following factors:
        - Requirement stability
        - Requirement uncertainty
        - Technical uncertainty
        - Risk level
        - Safety criticality
        - Need for prototyping
        - Integration complexity

        For every factor, determine whether it is:
        - Explicitly stated
        - Strongly implied
        - Derived from retrieved healthcare knowledge
        - Unknown / insufficient information

        Do not assume a factor merely because the system belongs to the
        healthcare domain.

        4. EXPLICIT VS IMPLIED
        Clearly distinguish between:
        - Explicit requirements: directly stated by the user.
        - Strongly implied requirements: not directly stated but logically
            necessary or strongly supported by the scenario.
        - Domain-derived considerations: obtained from the retrieved
            healthcare knowledge base.

        Do not present domain-derived considerations as if they were
        requirements stated by the user.

        5. HEALTHCARE KNOWLEDGE
        Use retrieved knowledge to provide relevant domain considerations.

        If retrieved knowledge mentions regulations, standards, or compliance
        requirements, include them only as domain considerations unless the
        project description indicates that they apply.

        6. UNCERTAINTY
        If the project description does not provide enough information to
        determine a factor, mark it as "Unknown" rather than guessing.

        7. SDLC
        Do NOT select, recommend, rank, or justify any SDLC model.
        The output will be passed to a separate SDLC Determination Agent.

        Return ONLY the structured output.
        """,
   tools=[qdrant_tools],
   output_schema=RequirementsOutput,
   output_key="requirements"
)