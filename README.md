# local-obsidian-RAG

A fully local RAG assistant for your Obsidian vault. A small local model
(Gemma via Ollama) handles inference and tool use; a dedicated embedding model
(nomic-embed-text) handles vector search; live note CRUD is handled by
[mcp-obsidian](https://github.com/MarkusPfundstein/mcp-obsidian) over stdio MCP
transport; a PyQt6 + qasync chat window provides the interface.

No cloud calls anywhere in the stack.

---

## Features

- **Hybrid retrieval** — BM25 + vector search merged via Reciprocal Rank
  Fusion for accurate recall across both semantic and exact-term queries
- **Header-aware chunking** — each chunk carries its Markdown section header
  as a context prefix, preserving structure for retrieval
- **Incremental indexing** — SHA-256 hashing skips unchanged files; a watchdog
  mode keeps the index live as you write
- **Live note access** — create, read, append, and delete notes at query time
  via the Obsidian Local REST API
- **Manual mode override** — toolbar buttons force RAG, MCP, or Auto routing
  per query; the status bar shows which path ran and whether it was automatic
  or manual
- **Streaming UI** — token-by-token streaming from Ollama back to the Qt
  window without a QThread, using `asyncio.to_thread` + `call_soon_threadsafe`

---

## Architecture

```
PyQt6 ChatWindow  (qasync — asyncio merged with Qt event loop)
        │
        ▼
   Orchestrator
   ├── QueryRouter          rule-based: RAG | MCP | BOTH
   ├── Retriever.search()   ChromaDB cosine + BM25 → RRF top-k
   ├── MCPClient            mcp-obsidian subprocess (stdio)
   └── OllamaClient
       ├── chat()           blocking, used in tool loop (max 3 iters)
       └── chat_stream()    asyncio.to_thread → Qt main thread
```

---

## Requirements

| Dependency | Purpose |
|---|---|
| [Ollama](https://ollama.com) | Local model inference and embeddings |
| `gemma3:4b` or similar | Generation model (set via `OLLAMA_MODEL`) |
| `nomic-embed-text` | Embedding model (set via `OLLAMA_EMBED_MODEL`) |
| [uvx](https://docs.astral.sh/uv/) | Runs `mcp-obsidian` without a global install |
| Obsidian + [Local REST API plugin](https://github.com/coddingtonbear/obsidian-local-rest-api) | Live vault access over HTTP/HTTPS |
| Python 3.11+ | Runtime |

---

## Installation

```bash
git clone <repo-url>
cd local-obsidian-RAG
pip install -e .
```

Pull the required Ollama models:

```bash
ollama pull nomic-embed-text
ollama pull gemma3:4b        # or whichever model you set in OLLAMA_MODEL
```

Install `uvx` if not already present:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Configuration

All machine-specific values live in `.env`. Copy the example and fill in your
values — you never need to edit `settings.yaml` between machines.

```bash
cp .env.example .env
```

**.env**

```dotenv
# Obsidian Local REST API plugin
OBSIDIAN_API_KEY=<key from plugin settings>
OBSIDIAN_HOST=127.0.0.1
OBSIDIAN_PORT=27123          # 27124 if HTTPS is enabled in the plugin

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
OLLAMA_EMBED_MODEL=nomic-embed-text

# Vault
VAULT_PATH=/path/to/your/obsidian/vault
```

`settings.yaml` contains non-machine-specific defaults (chunk sizes, top-k,
UI title, etc.) and is committed to the repository.

---

## Usage

**Verify Ollama connection**

```bash
python main.py --check
```

**Index the vault** (run once, then after bulk edits)

```bash
python main.py --index
```

**Index and watch for live changes**

```bash
python main.py --index --watch
```

**Launch the chat UI**

```bash
python main.py
```

> Obsidian must be open and the Local REST API plugin enabled for MCP
> (live note access) to work. RAG queries work without Obsidian running.

---

## Query routing

The toolbar exposes three mode buttons. **Auto** is the default; **RAG** and
**MCP** override the router for the current query regardless of phrasing.
The status bar shows which path ran and whether it was chosen automatically
or manually after every response.

```
Mode:  [ RAG ]  [ Auto ]  [ MCP ]    |    [ Re-index vault ]
```

### Mode reference

| Mode | When to use | What happens |
|---|---|---|
| **RAG** | "What do I know about X?" / "Summarise my notes on Y" / vault audits and inventories | Embeds the query, runs hybrid BM25 + vector search over the ChromaDB index, injects the top-k chunks as context. No live Obsidian access. Requires the vault to be indexed first. |
| **Auto** | General use — let the system decide | The `QueryRouter` classifies intent from the query text: write keywords → MCP, read/list keywords → MCP + RAG context, everything else → RAG only. Best for conversational queries where intent is unambiguous. |
| **MCP** | "List all files in root" / "Open note X" / "Append … to my daily note" / any live vault operation | Skips vector search entirely. The model receives the full MCP tool list and calls `obsidian_*` tools directly against the running Obsidian instance. Obsidian must be open. Use this whenever you want accurate, real-time vault content rather than indexed snapshots. |

### Choosing the right mode

- Use **RAG** for broad knowledge questions, topic summaries, and vault audits.
  The model synthesises an answer from indexed chunks — fast and works without
  Obsidian running, but limited to the top-k retrieved chunks.
- Use **MCP** when you need the actual current state of a note or the vault
  file tree, or when performing any write operation. The model operates on
  live data through the Obsidian REST API.
- Leave on **Auto** for day-to-day use. Switch to a fixed mode when Auto
  misroutes (the status bar will show the path taken so you can detect this).

---

## Project structure

```
local-obsidian-RAG/
├── app/
│   ├── ollama_client.py      OllamaClient — chat, stream, embed
│   ├── mcp_client.py         MCPClient — mcp-obsidian subprocess lifecycle
│   ├── orchestrator.py       QueryRouter, tool loop, streaming
│   └── rag/
│       ├── indexer.py        header-aware chunking, SHA-256 incremental, watchdog
│       └── retriever.py      BM25 + vector + RRF hybrid search
│   └── ui/
│       └── chat_window.py    PyQt6 ChatWindow + qasync
├── config/
│   └── settings.yaml         non-secret defaults, safe to commit
├── tests/
│   └── test_ollama_client.py
├── main.py                   CLI entry point
└── pyproject.toml
```

---

## Running tests

```bash
pytest tests/
```

Tests are fully offline — no Ollama or Obsidian connection required.

---

## Model selection

All models run locally via Ollama. Set `OLLAMA_MODEL` and `OLLAMA_EMBED_MODEL`
in `.env` to switch without touching any code.

### Generation models

| Model | Size | Tool-call reliability | Response quality | Speed | Notes |
|---|---|---|---|---|---|
| `gemma3:4b` | ~3 GB | Fair — often falls back to FLAG 1 text extraction | Good for simple Q&A and single-step writes | Fast (~5–10 s) | Minimum viable; occasional hallucinations on complex tool chains |
| `gemma4:e4b` | ~9.6 GB | Good — structured `tool_calls` more consistent | Noticeably better reasoning and citation | Moderate (~15–25 s) | Recommended balance of quality and RAM |
| `gemma3:12b` | ~8 GB | Good | Strong | Moderate | Good alternative if you prefer the Gemma 3 series |
| `llama3.1:8b` | ~5 GB | Good — Meta fine-tuned for tool use | Strong general reasoning | Moderate | Solid alternative; broader training data than Gemma |
| `qwen2.5:7b` | ~5 GB | Very good | Strong, especially on structured output | Moderate | Reliable tool-call format; good for complex multi-step MCP chains |
| `mistral:7b` | ~4 GB | Fair | Good | Fast | Weaker on tool use than the options above |

> **FLAG 1 note** — small models (≤4B) frequently embed tool-call JSON inside
> the text `content` field rather than the structured `tool_calls` field. The
> orchestrator includes two fallback extraction strategies for this, but
> reliability improves significantly at 7B+.

### Embedding models

| Model | Size | Retrieval quality | Speed | Notes |
|---|---|---|---|---|
| `nomic-embed-text` | ~274 MB | Good — 768-dim, strong on general prose | Very fast | Default; best size/quality tradeoff for most vaults |
| `mxbai-embed-large` | ~670 MB | Better — 1024-dim | Fast | Noticeable improvement on technical or domain-specific notes |
| `bge-m3` | ~1.2 GB | Best — multilingual, 1024-dim | Slower | Worth it for multilingual vaults or very large collections |
| `all-minilm` | ~45 MB | Basic | Fastest | Only for very constrained hardware; quality drops noticeably |

> Changing the embedding model requires a full re-index (`python main.py --index`)
> because existing ChromaDB vectors were produced by the previous model and are
> not compatible.

---

## Known limitations

- Conversation history is in-memory only and lost on restart
- MCP tool loop is capped at 3 iterations (`MAX_TOOL_ITERS` in `orchestrator.py`)
- Small models (4B) occasionally embed tool-call JSON in the text content field
  rather than the structured `tool_calls` field; the orchestrator includes two
  fallback extraction strategies (FLAG 1) to recover from this
