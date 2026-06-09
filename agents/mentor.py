import os
import operator
from pydantic import BaseModel, Field
from typing import Literal, Annotated, List, Dict, Any, TypedDict, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
from core.config import settings


class MentorState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    problem_description: str
    current_phase: Literal["IDEATION", "CODING"]
    code_submission: Optional[str]


class IntentClassification(BaseModel):
    intent: Literal["chat", "stuck", "submit"]
    reasoning: str

class ApprovePlan(BaseModel):
    """Call this ONLY when the user has clearly stated:
    (1) a valid algorithmic approach, AND
    (2) the correct time and space complexity.
    Both conditions must be met before calling."""
    rationale: str = Field(
        description="One sentence: what the user got right that earns the unlock."
    )


def get_llm(model: str, api_base: str, temperature: float = 0.4):
    return ChatOpenAI(
        model=model, 
        temperature=temperature,
        openai_api_key=settings.NOVITA_API_KEY,
        base_url=api_base,
        max_retries=5,
        timeout=60,
    )


def ideation_node(state: MentorState):
    """Socratic mentor. Asks one question at a time. Gate: ApprovePlan tool."""

    llm = get_llm(model="google/gemma-4-26b-a4b-it",api_base="https://api.novita.ai/openai", temperature=0.4).bind_tools([ApprovePlan])

    system = SystemMessage(content=(
        f"You are an expert DSA interviewer mentoring a candidate on this problem:\n"
        f"<problem>{state['problem_description']}</problem>\n\n"
        "RULES — follow them strictly:\n"
        "1. The code editor is LOCKED. The candidate must earn it by thinking first.\n"
        "2. NEVER write code. NEVER reveal the algorithm name unprompted.\n"
        "3. Ask EXACTLY ONE guiding question per turn. Not two. Not a question and a hint. One.\n"
        "4. Use the Socratic method: ask questions that lead them to discover the answer.\n"
        "5. When the candidate has clearly stated BOTH a valid approach AND correct "
        "   time/space complexity — call `ApprovePlan` immediately. Not before.\n"
        "6. If they are close but incomplete, ask a follow-up to close the gap."
    ))

    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]} 


def route_after_ideation(state: MentorState) -> Literal["transition_node", "__end__"]:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    for tc in tool_calls:
        if tc["name"] == "ApprovePlan":
            return "transition_node"
    return END


def transition_node(state: MentorState):
    """
    Fires when ApprovePlan is called.
    - Satisfies the ToolMessage the LLM expects
    - Sets current_phase to CODING
    - Sends unlock message to client
    - Goes to END so client can render the editor before coding starts
    """
    last_msg = state["messages"][-1]
    tool_call = last_msg.tool_calls[0]

    tool_msg = ToolMessage(
        content="Plan approved. Transitioning to coding phase.",
        tool_call_id=tool_call["id"]
    )
    
    ai_msg = AIMessage(
        content=(
            "Perfect. Your approach is solid and your complexity analysis is correct. "
            "I've unlocked the code editor — let's see your implementation. "
            "I'm here if you get stuck."
        )
    )

    return {
        "current_phase": "CODING",
        "messages": [tool_msg, ai_msg]
    }


def intent_router(state: MentorState) -> Literal["chat_node", "hint_node", "review_node"]:
    """
    Thin classifier that runs at the start of every coding-phase turn.
    
    Deterministic short-circuits run first (no LLM cost).
    Only falls through to the tiny LLM call when genuinely ambiguous.
    """

    # Short-circuit 1: submit button was clicked — deterministic, no LLM needed
    if state.get("code_submission"):
        return "review_node"

    llm = get_llm(model="google/gemma-4-26b-a4b-it", api_base="https://api.novita.ai/openai",temperature=0.0).with_structured_output(IntentClassification)

    system = SystemMessage(content=(
        "Classify the intent of the user's last message in a DSA coding session.\n\n"
        "Intents:\n"
        "- chat: thinking out loud, asking clarifying questions, general conversation\n"
        "- stuck: explicitly asking for help, saying they don't know, requesting a hint\n"
        "- submit: saying their code is done / ready for review\n\n"
        f"Problem context: {state['problem_description']}"
    ))

    last_message = state["messages"][-1]
    result = llm.invoke([system, last_message])

    intent_to_node = {
        "chat": "chat_node",
        "stuck": "hint_node",
        "submit": "review_node",
    }
    return intent_to_node[result.intent]


def coding_entry(state: MentorState):
    """Empty pass-through node. Just an anchor for the coding phase edge."""
    return {}


def chat_node(state: MentorState):
    """
    Normal coding conversation.
    Encourages, asks clarifying questions, points out bugs without fixing them.
    """
    llm = get_llm(model="google/gemma-4-26b-a4b-it", api_base="https://api.novita.ai/openai", temperature=0.4)

    system = SystemMessage(content=(
        f"You are an expert DSA mentor. The candidate is coding this problem:\n"
        f"<problem>{state['problem_description']}</problem>\n\n"
        "RULES:\n"
        "1. Do NOT write code for them. Do NOT give away the solution.\n"
        "2. Encourage their thinking. Ask ONE clarifying question if needed.\n"
        "3. If you spot a bug, point to the area — don't fix it.\n"
        "4. Keep responses short. One thought per turn."
    ))

    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}



def hint_node(state: MentorState):
    """
    Triggered when:
      - User explicitly says they're stuck (intent_router → "stuck")

    
    No tool_call on last message here — intent_router routes directly.
    """
    llm = get_llm(model="google/gemma-4-26b-a4b-it", api_base="https://api.novita.ai/openai", temperature=0.5)

    # Pull what the user is stuck on from their last message for context
    last_message_content = state["messages"][-1].content

    system = SystemMessage(content=(
        f"You are a DSA mentor. The candidate is solving:\n"
        f"<problem>{state['problem_description']}</problem>\n\n"
        f"The candidate appears stuck. Their last message: \"{last_message_content}\"\n\n"
        "Give ONE hint. Rules:\n"
        "- Do NOT write code.\n"
        "- Do NOT give the full answer.\n"
        "- Give the next smallest concrete thing to think about.\n"
        "- Be warm and encouraging.\n"
        "- Maximum 2-3 sentences."
    ))

    response = llm.invoke([system] + state["messages"])

    return {
        "messages": [response],
    }


def review_node(state: MentorState):
    """
    Triggered when:
      - Submit button clicked (code_submission set in state)
      - User says code is done (intent_router → "submit")
    
    Gives structured feedback. Clears code_submission after review.
    No tool_call on last message — intent_router routes directly.
    """
    llm = get_llm(model="google/gemma-4-26b-a4b-it", api_base="https://api.novita.ai/openai", temperature=0.2)

    # Use submitted code from state if available, otherwise ask for it
    code = state.get("code_submission")

    if not code:
        # User said they're done but didn't use the submit button
        response = AIMessage(content=(
            "It looks like you're ready for a review! "
            "Please use the Submit button in the editor so I can see your code."
        ))
        return {"messages": [response]}

    system = SystemMessage(content=(
        f"You are a senior engineer reviewing a DSA solution.\n"
        f"Problem: {state['problem_description']}\n\n"
        f"Code submitted:\n```\n{code}\n```\n\n"
        "Provide feedback in this exact structure:\n\n"
        "**Correctness** — does it solve the problem? Any bugs or missed edge cases?\n"
        "**Complexity** — actual time and space complexity with reasoning.\n"
        "**Style** — readability, naming, structure.\n"
        "**One thing to improve** — the single highest-priority change.\n\n"
        "Be direct and specific. No filler. No praise for the sake of praise."
    ))

    response = llm.invoke([system] + state["messages"])

    return {
        "messages": [response],
        "code_submission": None,  
    }



def route_on_phase(state: MentorState) -> Literal["ideation_node", "coding_entry"]:
    return "coding_entry" if state["current_phase"] == "CODING" else "ideation_node"
 
workflow = StateGraph(MentorState)


workflow.add_node("ideation_node", ideation_node)
workflow.add_node("transition_node", transition_node)
workflow.add_node("coding_entry", coding_entry)
workflow.add_node("chat_node", chat_node)
workflow.add_node("hint_node", hint_node)
workflow.add_node("review_node", review_node)


workflow.add_conditional_edges(START, route_on_phase)


workflow.add_conditional_edges(
    "ideation_node",
    route_after_ideation,
    {"transition_node": "transition_node", END: END}
)
workflow.add_edge("transition_node", END)  # client renders editor, waits for user


workflow.add_conditional_edges(
    "coding_entry",   
    intent_router,
    {
        "chat_node": "chat_node",
        "hint_node": "hint_node",
        "review_node": "review_node",
    }
)

workflow.add_edge("chat_node", END)
workflow.add_edge("hint_node", END)
workflow.add_edge("review_node", END)


mentor_graph = workflow.compile()