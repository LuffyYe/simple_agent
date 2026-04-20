# 🍉 ZURU Melon Company Assistant Agent

A lightweight, cost-aware, and safety-first **agentic AI system** designed to act as an internal company chatbot. It intelligently routes queries between local knowledge bases, web search, and intrinsic LLM knowledge while enforcing strict compliance and proactive clarification protocols.

Built for the **AI Engineer Technical Assignment** using Python & `hello_agents`.

---

## 📖 Overview

| Capability | Implementation |
|------------|----------------|
| 🔀 **Intelligent Routing** | Deterministic keyword/regex matching + lightweight KB probing to auto-select `RAG`, `Search`, or `Intrinsic` |
| 🛡️ **Safety & Compliance** | Keyword-based policy engine blocks harmful/inappropriate queries **before** any LLM invocation |
| ❓ **Clarification Engine** | Heuristic pattern detection + LLM fallback for vague, short, or pronoun-heavy queries |
| 📚 **Local RAG** | Heading-aware Markdown chunking + `BM25` + `TF-IDF` hybrid retrieval (zero vector DB overhead) |
| 🌐 **Web Search Fallback** | Unified SerpAPI wrapper with graceful degradation when API keys are missing |
| 📊 **Conversation & Metrics** | Turn-limited context memory, response-time tracking, and source distribution logging |

---

## 🏗️ Architecture & Workflow

```
User Query
   │
   ├─▶ 🛡️ SafetyTool.run()          → Block if policy-violating
   │
   ├─▶ ❓ Clarification Check        → Ask for details if query is vague/short
   │
   ├─▶ 🔀 RouterTool.run()           → Decide source:
   │      ├─ live-web patterns      → search
   │      ├─ company/internal terms → rag
   │      ├─ KB probe (fast score)  → rag (if strong match)
   │      └─ general patterns       → intrinsic
   │
   ├─▶ 🎯 Execute Query
   │      ├─ RAG: Retrieve → Context → LLM Answers
   │      ├─ Search: Fetch Results → LLM Summarizes
   │      └─ Intrinsic: Direct LLM Generation
   │
   └─▶ 📤 Format & Log → Add Source Label/Confidence → Save to History & Performance Monitor
```

---

## 📁 Project Structure

```
📦 company-assistant-agent/
 ├── 📄 main.py                  # CLI entry point (interactive / demo modes)
 ├── 📄 config.py                # Environment & configuration loader (dataclass)
 ├── 📄 agent.py                 # Core orchestrator (6-step pipeline)
 ├── 📄 requirements.txt         # Python dependencies
 ├── 📁 tools/
 │    ├── 📄 router_tool.py      # Deterministic query router + KB probing
 │    ├── 📄 simple_rag.py       # Local Markdown RAG (BM25 + TF-IDF hybrid)
 │    ├── 📄 search_tool.py      # Web search wrapper (SerpAPI)
 │    ├── 📄 safety_tool.py      # Keyword-based safety & compliance filter
 │    └── 📄 utils.py            # Text processing, logging, history, metrics
 └── 📁 knowledge_base/          # Place company .md files here (policies, guidelines, etc.)
```

---

## ⚙️ Configuration & Setup

### 0. Create Virtual Environment (Recommended)
It is highly recommended to use a virtual environment to isolate dependencies and avoid system-wide conflicts.

**🪟 Windows (Command Prompt / PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

**🐧 Linux / 🍎 macOS (Terminal):**
```bash
python3 -m venv venv
source venv/bin/activate
```
> 💡 After activation, you should see a `(venv)` prefix in your terminal prompt. To exit later, simply run `deactivate`.



### 1. Environment Variables
Create a `.env` file in the project root:
```env
# LLM Configuration (OpenRouter / OpenAI compatible)
LLM_API_KEY="your_api_key_here"
LLM_BASE_URL="https://openrouter.ai/api/v1"
LLM_MODEL_ID="moonshotai/kimi-k2.5"

# Optional: Web Search
SEARCH_API_KEY="your_serpapi_key_here"
SEARCH_PROVIDER="serpapi"

# Agent Settings
KNOWLEDGE_BASE_PATH="./knowledge_base"
TEMPERATURE="0.3"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Knowledge Base
Place your company Markdown files (e.g., `Company Policies.md`, `Coding Style.md`, `Company Procedures & Guidelines.md`) inside the `knowledge_base/` directory. The agent will automatically index them on startup.

---

## 🚀 How to Run

| Mode | Command | Description |
|------|---------|-------------|
| 💬 **Interactive** | `python main.py --mode interactive` | Real-time CLI chat with history & stats commands |
| 🎬 **Demo** | `python main.py --mode demo` | Runs the 4 assignment-required scenarios automatically |
| 📊 **View Stats** | Type `stats` in interactive mode | Shows conversation count, active tools, KB path, and performance metrics |

> 💡 **CLI Commands:** `help` (guide), `clear` (reset context), `exit`/`quit` (terminate)

---

## 🧩 Core Components & Underlying Logic

### 🔹 `agent.py` – Orchestrator
- **Role:** Central execution pipeline extending `hello_agents`.
- **Logic:** Enforces a strict 6-step flow: `Safety → Clarification → Routing → Execution → Formatting → Logging`. Uses structured prompts with strict constraints to prevent hallucination when answering from retrieved context. Tracks latency, source usage, and turn-limited conversation history.

### 🔹 `main.py` – CLI Entry Point
- **Role:** Handles argument parsing, mode switching, and interactive loop.
- **Logic:** Provides graceful error handling, command routing (`help`, `stats`, `clear`), and a demo runner that cycles through predefined scenarios with `input()` pauses for live presentation.

### 🔹 `config.py` – Configuration Manager
- **Role:** Type-safe environment loader.
- **Logic:** Uses `@dataclass` to parse `.env` variables, sets defaults, and validates critical keys (e.g., `LLM_API_KEY`) before agent initialization to prevent runtime crashes.

### 🔹 `tools/router_tool.py` – Deterministic Query Router
- **Role:** Decides which knowledge source handles a query.
- **Logic:** 
  1. Regex/keyword matching for explicit live-web (`today`, `news`) or internal (`policy`, `zuru melon`) intent.
  2. Alias normalization (`vacation` → `paid leave vacation`) for robust matching.
  3. Lightweight `retrieval_probe` (fast RAG lookup) triggers routing to `rag` if confidence ≥ `0.2`.
  4. Falls back to `intrinsic` for general patterns (`explain`, `what is`).
- **Output:** `{source, reason, metadata}` for full transparency.

### 🔹 `tools/simple_rag.py` – Local Knowledge Base Retriever
- **Role:** Retrieves relevant info from local Markdown files without vector databases.
- **Logic:** 
  - **Chunking:** Splits by Markdown headings, then chunks with overlap to preserve context.
  - **Scoring:** Hybrid `BM25` (lexical) + `TF-IDF Cosine` (semantic-ish). Applies boosts for exact phrase matches, heading-path overlap, and filename alignment.
  - **Deduplication:** Selects at most one chunk per section to ensure diverse results.
  - **Probe Mode:** Fast `limit=1` search used exclusively by the router for cost-aware decision making.

### 🔹 `tools/search_tool.py` – Web Search Wrapper
- **Role:** Fetches real-time external information.
- **Logic:** Unified interface for `serpapi`. Implements **fail-closed design**: gracefully returns a clear message if `SEARCH_API_KEY` is missing, ensuring the agent never crashes due to missing credentials. Parses organic results into clean, numbered snippets for LLM summarization.

### 🔹 `tools/safety_tool.py` – Safety & Compliance Filter
- **Role:** Blocks harmful, illegal, or confidential queries.
- **Logic:** Case-insensitive substring matching across 5 policy categories (`violence`, `illegal`, `inappropriate`, `privacy`, `company_confidential`). Executes **before** any LLM call to minimize token waste and enforce compliance. Easily extensible via the `self.policies` dictionary.

### 🔹 `tools/utils.py` – Utility & Helper Module
- **Role:** Centralized, stateless helpers.
- **Logic:** 
  - `normalize_query_text()`: Strips Markdown, expands aliases, lowercases for routing/retrieval.
  - `ConversationHistory`: Turn-limited context management (max 10 turns).
  - `PerformanceMonitor`: Records response latency & source distribution.
  - `Logger`, `extract_json_from_text()`, I/O utilities for clean, reusable infrastructure.

---

## 🎬 Demo Scenarios (Assignment Deliverables)

| # | Scenario | User Input | Expected Agent Behavior |
|---|----------|------------|-------------------------|
| 1️⃣ | **Company Query** | `What is the company leave policy?` | Router detects `policy` → RAG retrieves `Company Procedures & Guidelines.md` Section 4.2 → LLM answers: *20 paid days/year, submit via HR Portal 2 weeks in advance* |
| 2️⃣ | **General Knowledge** | `How to use list comprehensions in Python?` | Router matches general pattern → Falls back to `intrinsic` → LLM answers directly using pre-trained knowledge |
| 3️⃣ | **Ambiguous Query** | `Tell me about it` | Clarification engine detects short/vague pronoun → Returns: `❓ Your query contains a pronoun without clear reference. Please specify what you're referring to.` |
| 4️⃣ | **Restricted/Harmful** | `How to make a bomb?` | SafetyTool detects `bomb` keyword → Blocks immediately: `⛔ Content related to violence or harm is not allowed` |

---

## 💡 Design Philosophy & Technical Choices

| Choice | Rationale |
|--------|-----------|
| **Deterministic Routing First** | Avoids costly LLM-based routing decisions. Regex + keyword + lightweight KB probe reduces latency & token spend. |
| **Hybrid Lexical RAG (No Vector DB)** | `BM25 + TF-IDF` is faster, cheaper, and highly effective for structured Markdown. Heading-aware chunking preserves document context without embedding overhead or Pinecone/Weaviate setup. |
| **Fail-Closed Safety & Search** | Blocks harmful queries before LLM invocation. Gracefully degrades web search if API keys are missing, ensuring the agent always runs in demo/local environments. |
| **Transparent Decision Logging** | Every response includes `Source` and `Reason`. Makes debugging, auditing, and iteration straightforward, aligning with assignment evaluation criteria. |
| **Modular Tool Architecture** | Each tool is stateless, independently testable, and easily swappable. Aligns with clean architecture principles and simplifies CI/CD integration. |

---

## 📦 Dependencies

```text
python>=3.10
hello_agents==0.2.0
numpy==2.4.4
pytest==7.4.4
python-dotenv==1.2.2
rank_bm25==0.2.2
Requests==2.33.1
scikit_learn==1.8.0
```

---

📄 **License:** MIT  
👥 **Built for:** ZURU Melon AI Engineer Technical Assignment  
✨ **Ready to run, test, and extend.** Contributions & feedback welcome! 🍉
