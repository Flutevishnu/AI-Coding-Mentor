from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from core.config import settings
from typing import Annotated, Literal, Optional, TypedDict, Any
from langsmith import Client

client = Client()

class MentorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    problem_description: str
    current_phase: Literal["IDEATION", "CODING"]
    agreed_plan: Optional[str]
    code_submission: Optional[str]
    past_related_problems: Optional[list[str]]
    technical_blindspots: Optional[list[str]]
    behavioral_traits: Optional[list[str]]
    review: Optional[dict]
    session_summary: Optional[dict]


class CodeReview(BaseModel):
    correctness: str = Field(description="Does it solve the problem? Any bugs or missed edge cases?")
    complexity: str = Field(description="Actual time and space complexity with reasoning.")
    style: str = Field(description="Readability, naming, structure.")
    improvement: str = Field(description="The single highest-priority change.")

class ConceptMastery(BaseModel):
    concept: str = Field(description="E.g., 'Two-pointer technique', 'Space complexity analysis'")
    status: Literal["mastered", "struggling", "needs_review"]
    context: str = Field(description="One sentence explaining WHY they are struggling or mastered it.")


class SessionSummary(BaseModel):
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



class VisualizationStep(BaseModel):
    description: str = Field(description="What's happening in this step. Keep it conversational and Socratic.")
    highlights: dict[str, Any] = Field(default_factory=dict, description="Which indices/nodes to highlight. E.g., {'i': 0, 'j': 3}. CRITICAL: MUST be a valid JSON object, NEVER a string.")
    variables: dict[str, Any] = Field(default_factory=dict, description="Current variable values. E.g., {'sum': 8}. CRITICAL: MUST be a nested JSON object. If the underlying array/matrix/linked_list mutates in this step (e.g. marking an island as '#'), you MUST include the entire updated state here! E.g. {'grid': [['#','1','0']...], 'count': 1}")
    found: bool = Field(default=False, description="CRITICAL: MUST be a valid boolean (true or false). True if this step is the 'eureka' moment")


class GenerateVisualization(BaseModel):
    viz_type: Literal["array", "linked_list", "binary_tree", "hashmap", "matrix"] = Field(
        ...,
        description="CRITICAL: You MUST provide this field. Choose the visualization type."
    )
    title: str = Field(..., description="CRITICAL: You MUST provide this field. E.g., 'Two Pointers on [1,3,5,7]'")
    data: dict = Field(
        ...,
        description=(
            "CRITICAL: The initial data structure state. MUST MATCH the viz_type exactly:\n"
            "- array: {'array': [1,2,3], 'target': 5}\n"
            "- linked_list: {'head': [1,2,3,4]}\n"
            "- binary_tree: {'root': [1,2,3,null,5,null,4]} (LeetCode level-order array format)\n"
            "- matrix: {'grid': [[1,2],[3,4]]}\n"
            "- hashmap: {'map': {}}  <-- CRITICAL: MUST be a nested JSON object with key 'map', NEVER a string like '{}' or array."
        )
    )
    steps: list[VisualizationStep]


class ApprovePlan(BaseModel):
    rationale: str = Field(
        description="One sentence: what the user got right that earns the unlock."
    )


def get_llm(model, temperature: float = 0.4):
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite", 
        temperature=temperature,
        google_api_key=settings.GOOGLE_API_KEY,
        max_retries=5,
        timeout=60,
    )


def ideation_node(state: MentorState):
    llm = get_llm(
        model="gemini-2.5-pro",
        temperature=0.4,
    ).bind_tools([ApprovePlan])
    past_problems = state.get("past_related_problems") or []
    blindspots = state.get("technical_blindspots") or []
    traits = state.get("behavioral_traits") or []

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

    
    ideation_prompt = client.pull_prompt("basecase-mentor-ideation")
    formatted_prompt = ideation_prompt.invoke({
        "problem_description": state['problem_description'],
        "custom_instructions": custom_instructions,
        "behavior_context": behavior_context,
        "messages": state["messages"]
    })

    response = llm.invoke(formatted_prompt)
    return {"messages": [response]}


def route_after_ideation(state: MentorState) -> Literal["transition_node", "__end__"]:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    for tool_call in tool_calls:
        if tool_call["name"] == "ApprovePlan":
            return "transition_node"
    return END


def transition_node(state: MentorState):
    last_msg = state["messages"][-1]
    tool_call = last_msg.tool_calls[0]
    
    rationale = tool_call["args"].get("rationale", "User successfully planned the optimal approach.")

    tool_msg = ToolMessage(
        content="Plan approved. Transitioning to coding phase.",
        tool_call_id=tool_call["id"],
    )
    
    congratulatory_msg = AIMessage(
        content="Perfect! Your approach is solid and your complexity analysis is correct. I have unlocked the code editor for you—let's see your implementation. I'm here if you get stuck!"
    )

    return {
        "current_phase": "CODING",
        "agreed_plan": rationale,
        "messages": [tool_msg, congratulatory_msg],
    }


def coding_node(state: MentorState):
    llm = get_llm(
        model="gemini-2.5-pro",
        temperature=0.3,
    ).bind_tools([GenerateVisualization])

    agreed_plan = state.get("agreed_plan", "No plan recorded.")

    coding_prompt = client.pull_prompt("basecase-mentor-coding")
    formatted_prompt = coding_prompt.invoke({
        "problem_description": state['problem_description'],
        "agreed_plan": agreed_plan,
        "messages": state["messages"]
    })

    response = llm.invoke(formatted_prompt)
    return {"messages": [response]}



def review_node(state: MentorState):
    llm = get_llm(
        model="gemini-3.1-flash-lite",
        temperature=0.2,
    ).with_structured_output(CodeReview)

    code = state.get("code_submission")

    if not code:
        response = AIMessage(
            content=(
                "It looks like you're ready for a review! "
                "Please use the Submit button in the editor so I can see your code."
            )
        )
        return {"messages": [response]}

    prompt_template = client.pull_prompt("basecase-mentor-review")
    formatted_prompt = prompt_template.invoke({
        "problem_description": state['problem_description'],
        "code": code,
        "messages": state["messages"]
    })

    summary = llm.invoke(formatted_prompt)

    return {
        "review": summary.dict(),
        "code_submission": None,
    }



def route_after_review(state: MentorState) -> Literal["extractor_node", "__end__"]:
    last = state["messages"][-1]
    content = last.content if isinstance(last.content, str) else ""
    if "Submit button" in content:
        return END
    return "extractor_node"


def extractor_node(state: MentorState):
    llm = get_llm(
        model="gemini-3.1-flash-lite",
        temperature=0.1,
    ).with_structured_output(SessionSummary)

    prompt_template = client.pull_prompt("basecase-mentor-extractor")
    transcript = "\\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
    
    formatted_prompt = prompt_template.invoke({
        "transcript": transcript
    })

    summary = llm.invoke(formatted_prompt)
    return {}


def route_on_phase(state: MentorState) -> Literal["ideation_node", "coding_node", "review_node"]:
    if state["current_phase"] == "CODING":
        if state.get("code_submission"):
            return "review_node"
        return "coding_node"
    return "ideation_node"



workflow = StateGraph(MentorState)

workflow.add_node("ideation_node", ideation_node)
workflow.add_node("transition_node", transition_node)
workflow.add_node("coding_node", coding_node)
workflow.add_node("review_node", review_node)
workflow.add_node("extractor_node", extractor_node)

workflow.add_conditional_edges(START, route_on_phase, ["ideation_node", "coding_node", "review_node"])
workflow.add_conditional_edges(
    "ideation_node",
    route_after_ideation,
    {"transition_node": "transition_node", END: END},
)
workflow.add_edge("transition_node", END)

workflow.add_edge("coding_node", END)
workflow.add_conditional_edges(
    "review_node",
    route_after_review,
    {"extractor_node": "extractor_node", END: END},
)
workflow.add_edge("extractor_node", END)

mentor_graph = workflow.compile()
