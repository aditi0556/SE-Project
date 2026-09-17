from google.adk.agents import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from pydantic import BaseModel, Field
from typing import List

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

requirement_analyser = Agent(
    name="requirement_analyser",
    model="gemini-3.5-flash-lite",
    description=(
        "Analyzes a user's project description and identifies "
        "functional requirements, non-functional requirements, "
        "risks, uncertainties, and requirement stability."
    ),
    instruction="""
         You are a Requirement Analyser Agent for healthcare software.

         You MUST call the Qdrant retrieval tool before generating
         your requirements.

         Use the healthcare knowledge base to retrieve relevant
         healthcare-domain information.

         Then:

         1. Identify functional requirements explicitly mentioned
            or clearly implied by the project description.
            Do NOT invent technologies, standards, regulations,
            integrations, or implementation details.

         2. Identify relevant non-functional requirements:
            - Security
            - Performance
            - Reliability
            - Scalability
            - Usability
            - Availability
            - Maintainability
            - Data integrity
            - Auditability

         3. Identify factors useful for later SDLC determination:
            - Requirement stability
            - Requirement uncertainty
            - Technical uncertainty
            - Risk level
            - Safety criticality
            - Regulatory constraints
            - Need for user feedback
            - Need for prototyping
            - Integration complexity

         4. Clearly distinguish between:
            - requirements explicitly stated by the user
            - requirements that are strongly implied by the scenario

         5. Use the healthcare knowledge base to identify relevant
            domain considerations, but do not assume that a specific
            technology, regulation, standard, or integration is being
            used unless supported by the project description or
            retrieved knowledge.

         6. Do NOT determine or recommend an SDLC model.

         Return only the structured output.
         """,
   tools=[qdrant_tools],
   output_schema=RequirementsOutput,
   output_key="requirements"
)