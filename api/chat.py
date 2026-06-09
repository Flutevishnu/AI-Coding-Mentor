import logging

from fastapi import APIRouter, status
from core.config import settings
from pydantic import BaseModel
from schemas.request import ChatActionRequest
from schemas.response import ChatResponse
from agents.mentor import workflow
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
memory = InMemorySaver()
mentor_graph = workflow.compile(checkpointer=memory)

chat_router = APIRouter(tags=["Chat"])


@chat_router.post(
    "/chat",
    status_code=status.HTTP_200_OK,
    summary="Chat with the AI Coding Mentor",
    description="Send a message to the AI Coding Mentor and receive a response. The mentor can assist with coding questions, debugging, and learning resources."
)   

async def chat(request: ChatActionRequest) -> ChatResponse:
    
    config = RunnableConfig(configurable={"thread_id": request.session_id})
    logger.info("chat request received: session_id=%s action_type=%s", request.session_id, request.action_type)

    if request.problem_description:
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "problem_description": request.problem_description,
            "current_phase": "IDEATION",
            "code_submission": None
        }
        output = mentor_graph.invoke(initial_state, config=config)

    elif request.action_type == "chat":
        msg = {"messages": [HumanMessage(content=request.message)]}
        output = mentor_graph.invoke(msg, config=config)

    elif request.action_type == "submit_code":
        msg = {"messages": [HumanMessage(content="Here is my Latest Code Submission")]}
        mentor_graph.update_state(config, {"code_submission": request.code})
        output = mentor_graph.invoke(msg, config=config)
        current_state = mentor_graph.get_state(config)
        logger.info("mentor_graph state after submit_code: %s", current_state)

    else:
        logger.warning("Unsupported action_type received: %s", request.action_type)
        raise ValueError(f"Unsupported action_type: {request.action_type}")

    return ChatResponse(
        response=output["messages"][-1].content,
        current_phase=output["current_phase"]
    )
