# BaseCase AI: LangGraph Backend

This repository contains the intelligent backend agent for the BaseCase Socratic Tutoring platform. It is built using Python, LangGraph, and Google Gemini.

## Architecture Overview

The backend acts as a stateful graph (Finite State Machine) that manages the lifecycle of a user's tutoring session.

- **`ideation_node`**: Guides the user from a brute-force approach to an optimal solution using the Socratic method.
- **`transition_node`**: A gatekeeper that only unlocks the coding phase when the user's plan and complexity analysis are explicitly approved by the AI.
- **`coding_node`**: Assists the user while they write code, strictly refusing to write the answer for them, but providing targeted hints.
- **`review_node`**: Uses native JSON Structured Output to provide a strict, predictable payload containing correctness, complexity, style, and improvement feedback.
- **`extractor_node`**: A background process that parses the entire chat transcript upon submission and extracts a JSON schema of the student's technical blindspots and behavioral traits.

## Observability & Prompt Engineering

We decouple all logic from our prompts to ensure rapid iteration and zero-downtime updates.
All major system instructions are hosted on the **LangChain Prompt Hub**. The python code dynamically pulls:
- `basecase-mentor-ideation`
- `basecase-mentor-coding`
- `basecase-mentor-review`
- `basecase-mentor-extractor`

## Quick Start

```bash
# Install dependencies using uv
uv sync

# Start the LangGraph development server
uv run langgraph dev
```

For full documentation and frontend code, see the `basecase` repository.
