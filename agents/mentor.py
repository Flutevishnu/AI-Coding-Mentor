from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from core.config import settings


class MentorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    problem_description: str
    current_phase: Literal["IDEATION", "CODING"]
    code_submission: Optional[str]
    past_related_problems: Optional[list[str]]
    technical_blindspots: Optional[list[str]]
    behavioral_traits: Optional[list[str]]
    review: Optional[dict]
    session_summary: Optional[dict]


class ConceptMastery(BaseModel):
    concept: str = Field(description="E.g., 'Two-pointer technique', 'Space complexity analysis'")
    status: Literal["mastered", "struggling", "needs_review"]
    context: str = Field(description="One sentence explaining WHY they are struggling or mastered it.")


class SessionSummary(BaseModel):
    """Structured payload for the gateway to forward to the Node.js backend."""

    problem_name: str = Field(description="The formal name of the problem solved in this session.")
    core_pattern_used: str = Field(
        description="The primary algorithmic pattern (e.g., 'Two Pointers', 'Sliding Window', 'DFS')."
    )
    technical_blindspots: list[str] = Field(
        description="Specific coding habits to watch out for. E.g., 'Frequently writes out-of-bounds array loops'."
    )
    behavioral_traits: list[str] = Field(
        description="How the user interacts. E.g., 'Jumps straight to coding without planning', 'Gets frustrated easily'."
    )
    concept_updates: list[ConceptMastery] = Field(
        description="Updates on specific DSA concepts encountered in this session."
    )


class IntentClassification(BaseModel):
    intent: Literal["chat", "stuck", "submit"]


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

    llm = get_llm(
        model="google/gemma-4-31b-it",
        api_base="https://api.novita.ai/openai",
        temperature=0.4,
    ).bind_tools([ApprovePlan])
    past_problems = state.get("past_related_problems") or []
    blindspots = state.get("technical_blindspots") or []
    traits = state.get("behavioral_traits") or []

    base_rules = (
            "RULES — follow them strictly:\n"
            "1. The code editor is LOCKED. The candidate must earn it by thinking first.\n"
            "2. NEVER write code. NEVER reveal the algorithm name unprompted.\n"
            "3. Ask EXACTLY ONE guiding question per turn.\n"
            "4. ENFORCE THE OPTIMIZATION LADDER. You must guide the user through this exact progression conceptually:\n"
            "   - Stage 1 (Brute Force): Force them to state the most naive, obvious solution first. Ask for its time/space complexity.\n"
            "   - Stage 2 (Better): Ask them to identify the bottleneck in their brute force approach.\n"
            "   - Stage 3 (Optimal): Guide them via hints to the most efficient algorithmic approach.\n"
            "5. THE GATE: Do NOT call `ApprovePlan` during the Brute Force or Better stages. "
            "   You may ONLY call `ApprovePlan` when they have clearly stated the OPTIMAL approach AND its correct time/space complexity.\n"
            "6. If they jump straight to the optimal solution immediately, ask them to briefly explain the brute force approach first just to verify their fundamentals, then proceed."
    )

    behavior_context = ""
    if blindspots or traits:
        blindspots_str = ", ".join(blindspots) if blindspots else "None noted yet."
        traits_str = ", ".join(traits) if traits else "None noted yet."
        behavior_context = (
            f"\n\nSTUDENT PROFILE:\n"
            f"- Technical Blindspots to watch out for: {blindspots_str}\n"
            f"- Behavioral Traits: {traits_str}\n"
            f"Tailor your tone and hints to fit this profile."
        )

    if past_problems:
        past_problems_str = ", ".join(past_problems)
        custom_instructions = (
            f"USER CONTEXT:\n"
            f"The user has previously solved these related problems using the exact same pattern: {past_problems_str}\n\n"
            "CRITICAL DIRECTIVE: For your very first message, draw a parallel to one of their past related problems. "
            "Ask if they remember the core pattern they used there, but STILL require them to build the logic from Brute Force to Optimal."
        )
    else:
        custom_instructions = (
            "USER CONTEXT:\n"
            "This is the first time the user is encountering this specific algorithmic pattern.\n\n"
            "CRITICAL DIRECTIVE: Assume zero prior knowledge of the optimal pattern. "
            "Focus heavily on Stage 1 (Brute Force). Let them struggle with the naive approach before nudging them toward the bottleneck."
        )

    system = SystemMessage(
        content=(
            f"You are an expert DSA interviewer mentoring a candidate on this problem:\n"
            f"<problem>{state['problem_description']}</problem>\n\n"
            f"{custom_instructions}\n\n"
            f"{base_rules}"
            f"{behavior_context}"
        )
    )

    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}


def route_after_ideation(state: MentorState) -> Literal["transition_node", "__end__"]:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    for tool_call in tool_calls:
        if tool_call["name"] == "ApprovePlan":
            return "transition_node"
    return END


def transition_node(state: MentorState):
    """Unlock coding phase after ApprovePlan is called."""
    last_msg = state["messages"][-1]
    tool_call = last_msg.tool_calls[0]

    tool_msg = ToolMessage(
        content="Plan approved. Transitioning to coding phase.",
        tool_call_id=tool_call["id"],
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
        "messages": [tool_msg, ai_msg],
    }


def intent_router(state: MentorState) -> Literal["chat_node", "hint_node", "review_node"]:
    """Route coding-phase turns. Short-circuit submit when code is already in state."""
    if state.get("code_submission"):
        return "review_node"

    llm = get_llm(
        model="google/gemma-4-31b-it",
        api_base="https://api.novita.ai/openai",
        temperature=0.0,
    ).with_structured_output(IntentClassification)

    system = SystemMessage(
        content=(
            "Classify the intent of the user's last message in a DSA coding session.\n\n"
            "Intents:\n"
            "- chat: thinking out loud, asking clarifying questions, general conversation\n"
            "- stuck: explicitly asking for help, saying they don't know, requesting a hint\n"
            "- submit: saying their code is done / ready for review\n\n"
            f"Problem context: {state['problem_description']}"
        )
    )

    last_message = state["messages"][-1]
    result = llm.invoke([system, last_message], config={"callbacks": []})

    intent_to_node = {
        "chat": "chat_node",
        "stuck": "hint_node",
        "submit": "review_node",
    }
    return intent_to_node[result.intent]


def coding_entry(state: MentorState):
    """Anchor node for the coding phase routing edge."""
    return {}


def chat_node(state: MentorState):
    """Normal coding conversation."""
    llm = get_llm(
        model="google/gemma-4-31b-it",
        api_base="https://api.novita.ai/openai",
        temperature=0.4,
    )

    system = SystemMessage(
        content=(
            f"You are an expert DSA mentor. The candidate is coding this problem:\n"
            f"<problem>{state['problem_description']}</problem>\n\n"
            "RULES:\n"
            "1. Do NOT write code for them. Do NOT give away the solution.\n"
            "2. Encourage their thinking. Ask ONE clarifying question if needed.\n"
            "3. If you spot a bug, point to the area — don't fix it.\n"
            "4. Keep responses short. One thought per turn."
        )
    )

    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}


def hint_node(state: MentorState):
    """Give one hint when the user is stuck."""
    llm = get_llm(
        model="google/gemma-4-31b-it",
        api_base="https://api.novita.ai/openai",
        temperature=0.5,
    )

    last_message_content = state["messages"][-1].content

    system = SystemMessage(
        content=(
            f"You are a DSA mentor. The candidate is solving:\n"
            f"<problem>{state['problem_description']}</problem>\n\n"
            f"The candidate appears stuck. Their last message: \"{last_message_content}\"\n\n"
            "Give ONE hint. Rules:\n"
            "- Do NOT write code.\n"
            "- Do NOT give the full answer.\n"
            "- Give the next smallest concrete thing to think about.\n"
            "- Be warm and encouraging.\n"
            "- Maximum 2-3 sentences."
        )
    )

    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}


def review_node(state: MentorState):
    """Review submitted code and clear code_submission after feedback."""
    llm = get_llm(
        model="google/gemma-4-31b-it",
        api_base="https://api.novita.ai/openai",
        temperature=0.2,
    )

    code = state.get("code_submission")

    if not code:
        response = AIMessage(
            content=(
                "It looks like you're ready for a review! "
                "Please use the Submit button in the editor so I can see your code."
            )
        )
        return {"messages": [response]}

    system = SystemMessage(
        content=(
            f"You are a senior engineer reviewing a DSA solution.\n"
            f"Problem: {state['problem_description']}\n\n"
            f"Code submitted:\n```\n{code}\n```\n\n"
            "Provide feedback in this exact structure:\n\n"
            "**Correctness** — does it solve the problem? Any bugs or missed edge cases?\n"
            "**Complexity** — actual time and space complexity with reasoning.\n"
            "**Style** — readability, naming, structure.\n"
            "**One thing to improve** — the single highest-priority change.\n\n"
            "Be direct and specific. No filler. No praise for the sake of praise."
        )
    )

    response = llm.invoke([system] + state["messages"])

    return {
        "review": response,
        "code_submission": None,
    }


def route_after_review(state: MentorState) -> Literal["extractor_node", "__end__"]:
    last = state["messages"][-1]
    content = last.content if isinstance(last.content, str) else ""
    if "Submit button" in content:
        return END
    return "extractor_node"


def extractor_node(state: MentorState):
    """Extract session summary for the gateway to forward to the product backend."""
    llm = get_llm(
        model="google/gemma-4-31b-it",
        api_base="https://api.novita.ai/openai",
        temperature=0.1,
    ).with_structured_output(SessionSummary)

    system = SystemMessage(
        content=(
            "You are an analytical engine. Review the following Socratic coding session "
            "and extract the student's technical blindspots, behavioral traits, and concept mastery. "
            "Also, identify the formal problem name and the core algorithmic pattern used. "
            "Do not output markdown, only the requested JSON schema."
        )
    )

    summary = llm.invoke([system] + state["messages"])
    return {}


def route_on_phase(state: MentorState) -> Literal["ideation_node", "coding_entry"]:
    return "coding_entry" if state["current_phase"] == "CODING" else "ideation_node"


workflow = StateGraph(MentorState)

workflow.add_node("ideation_node", ideation_node)
workflow.add_node("transition_node", transition_node)
workflow.add_node("coding_entry", coding_entry)
workflow.add_node("chat_node", chat_node)
workflow.add_node("hint_node", hint_node)
workflow.add_node("review_node", review_node)
workflow.add_node("extractor_node", extractor_node)

workflow.add_conditional_edges(START, route_on_phase, ["ideation_node", "coding_entry"])

workflow.add_conditional_edges(
    "ideation_node",
    route_after_ideation,
    {"transition_node": "transition_node", END: END},
)
workflow.add_edge("transition_node", END)

workflow.add_conditional_edges(
    "coding_entry",
    intent_router,
    {
        "chat_node": "chat_node",
        "hint_node": "hint_node",
        "review_node": "review_node",
    },
)

workflow.add_edge("chat_node", END)
workflow.add_edge("hint_node", END)
workflow.add_conditional_edges(
    "review_node",
    route_after_review,
    {"extractor_node": "extractor_node", END: END},
)
workflow.add_edge("extractor_node", END)

mentor_graph = workflow.compile()
