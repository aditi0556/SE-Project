from google.adk.agents import SequentialAgent
from .agents.requirement_analyser import requirement_analyser
from .agents.sdlc_determiner import sdlc_determiner


root_agent = SequentialAgent(
    name="software_engineering_orchestrator",
    sub_agents=[
        requirement_analyser,
        sdlc_determiner,
    ],
)