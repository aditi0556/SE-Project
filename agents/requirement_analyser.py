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
    model="gemini-flash-latest",
    description=(
        "Analyzes a user's project description and identifies "
        "functional requirements, non-functional requirements, "
        "risks, uncertainties, and requirement stability."
    ),
    instruction="""
         You are a Requirement Analyser Agent.

         Your task is to analyze the project description provided by the user
         and also use the qdrant_tools to give the requirements to predict the sdlc model.

         Follow these rules:

         1. Identify all functional requirements explicitly mentioned or
            clearly implied by the project description.

         2. Identify relevant non-functional requirements such as:
            - Security
            - Performance
            - Reliability
            - Scalability
            - Usability
            - Availability
            - Maintainability

         3.Identify factors that are useful for later SDLC determination:
         - Requirement stability
         - Requirement uncertainty
         - Technical uncertainty
         - Risk level
         - Safety criticality
         - Regulatory/compliance constraints
         - Need for user feedback
         - Need for prototyping
         - Integration complexity
   """,
   tools=[qdrant_tools],
   output_schema=RequirementsOutput,
   output_key="requirements"
)