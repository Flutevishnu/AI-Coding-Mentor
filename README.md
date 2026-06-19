# BaseCase AI: LangGraph Backend

This repository contains the intelligent backend agent for the BaseCase Socratic Tutoring platform. It is built using Python, LangGraph, and Google Gemini.

## Architecture Overview

The backend acts as a stateful graph (Finite State Machine) that manages the lifecycle of a user's tutoring session.

- **`ideation_node`**: Guides the user from a brute-force approach to an optimal solution using the Socratic method.
- **`transition_node`**: A gatekeeper that only unlocks the coding phase when the user's plan and complexity analysis are explicitly approved by the AI.
- **`coding_node`**: Assists the user while they write code, strictly refusing to write the answer for them, but providing targeted hints.
- **`review_node`**: Uses native JSON Structured Output to provide a strict, predictable payload containing correctness, complexity, style, and improvement feedback.
- **`extractor_node`**: A background process that parses the entire chat transcript upon submission and extracts a JSON schema of the student's technical blindspots and behavioral traits.

## Prompt Engineering

Logic is decoupled from prompts to allow rapid iteration. All system instructions are hosted on the **LangChain Prompt Hub**:
- `basecase-mentor-ideation`
- `basecase-mentor-coding`
- `basecase-mentor-review`
- `basecase-mentor-extractor`

## Configuration

Configuration is loaded from a `.env` file in the project root:

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `GOOGLE_API_KEY` | Yes | — | Google Gemini API key |
| `LANGSMITH_API_KEY` | Yes | — | Pulls prompts from the LangChain Prompt Hub |
| `BIG_MODEL` | Yes | — | Model for ideation and coding nodes |
| `SMALL_MODEL` | Yes | — | Model for review and extractor nodes |
| `ENV` | No | `dev` | `dev` or `prod` |
| `LOG_DIR` | No | `logs` | Directory for log files |
| `LOG_LEVEL` | No | `INFO` | Standard Python log level |
| `POSTGRES_URI_CUSTOM` | No | — | Custom Postgres for LangGraph checkpointing |

Example `.env`:

```env
GOOGLE_API_KEY=your-google-api-key
LANGSMITH_API_KEY=your-langsmith-api-key
BIG_MODEL=gemini-3.5-flash
SMALL_MODEL=gemini-3.1-flash-lite
ENV=dev
```

## Quick Start

```bash
# Install dependencies using uv
uv sync

# Start the LangGraph development server
uv run langgraph dev
```

For full documentation and frontend code, see the `basecase` repository.
