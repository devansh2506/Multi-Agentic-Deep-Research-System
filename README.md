# Multi-Agentic Deep Research System

A sophisticated, asynchronous multi-agent AI system built with LangGraph, LangChain, and Tavily Search to autonomously scope, research, and synthesize comprehensive reports on any topic.

## 🚀 Overview

This project uses a pipeline of specialized AI agents working together to conduct deep research. Instead of relying on a single LLM prompt, the workload is distributed across four distinct agents that handle user scoping, task delegation, parallel web research, and final report synthesis.

## 🤖 The Agent Architecture

The system operates in three distinct phases, coordinated sequentially by a main entry script (`chat.py`).

### 1. Scoping Agent (`scoping/`)
**Purpose**: To act as the conversational frontend and define the exact parameters of the research.
* **How it works**: It engages in an interactive chat with the user. If a request is vague, it asks clarifying questions using conversation memory (checkpointer). Once it has enough context (or after 5 clarification attempts), it translates the chat history into a highly specific **Research Brief**.
* **Handoff**: The Research Brief is passed downstream as the master instruction manual for the rest of the pipeline.

### 2. Supervisor Agent (`research_supervisor/`)
**Purpose**: To orchestrate the research process and distribute workloads.
* **How it works**: The Supervisor reads the Research Brief and breaks it down into multiple sub-topics. It then asynchronously spawns parallel **Researcher Agents** to investigate each sub-topic simultaneously, drastically reducing research time.

### 3. Researcher Agent (`research/`)
**Purpose**: To conduct deep, iterative web research.
* **How it works**: Each Researcher is given a specific sub-topic by the Supervisor. It operates in a tool-calling loop using the **Tavily Search API**. After every search, it uses an internal `think_tool` to reflect on the findings, identify missing information, and formulate its next query. Once satisfied (or hitting a hard query limit), it cleans and returns its findings to the Supervisor.

### 4. Reporter Agent (`reporter/`)
**Purpose**: To synthesize a polished final deliverable.
* **How it works**: Once all parallel Researchers return their findings to the Supervisor, the aggregated raw notes are passed to the Reporter Agent. It structures the data, removes redundancies, and generates a comprehensive, cleanly formatted Markdown report complete with inline citations and a sources list.

## 🛠️ Setup & Execution

### Prerequisites
- Python 3.9+
- A `.env` file in the root directory containing your API keys:
  ```env
  GROQ_API_KEY=your_groq_key
  GEMINI_API_KEY=your_gemini_key
  TAVILY_API_KEY=your_tavily_key
  
  # Optional: For LangSmith tracing
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=your_langchain_key
  LANGCHAIN_PROJECT="Research Agent"
  ```

### Running the System
To run the full end-to-end pipeline:
```bash
python chat.py
```

### Standalone Testing
You can also test the Scoping or Research agents independently by running their standalone scripts:
```bash
python scoping/chat.py
python research/chat.py
```

## 📊 Observability
The entire system is instrumented for **LangSmith**. Because each node and sub-graph is explicitly named (e.g., `Scoping_Agent`, `Supervisor_Agent`, `Parallel_Worker_xyz`), you can monitor the exact execution traces, tool calls, and latency of parallel agents directly in your LangSmith dashboard.
