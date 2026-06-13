# LangChain Overview

LangChain is a framework for developing applications powered by large language models (LLMs).

## Core Concepts

LangChain provides a standard interface for interacting with LLMs. It abstracts away
provider-specific details so you can switch between OpenAI, Anthropic, HuggingFace, and
others without rewriting your application logic.

### Chains

Chains are the fundamental building blocks of LangChain. A chain combines one or more
components (LLMs, prompts, parsers) into a reusable pipeline.

- **LLMChain**: The simplest chain; sends a prompt to an LLM and returns the response.
- **SequentialChain**: Runs chains in sequence, feeding output of one as input to the next.
- **RouterChain**: Dynamically routes inputs to the appropriate chain based on content.

### Prompts

Prompt templates allow you to dynamically generate prompts by injecting variables into
predefined templates. This makes prompts reusable and version-controllable.

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms."
)
```

## Memory

LangChain supports multiple memory types to give LLMs context across interactions:

- **ConversationBufferMemory**: Stores all previous messages verbatim.
- **ConversationSummaryMemory**: Summarizes old messages to save token space.
- **VectorStoreMemory**: Retrieves semantically relevant past messages.

### Memory Usage Pattern

Memory is attached to chains and automatically injected into each prompt. This enables
stateful conversations without manual context management.

## Agents

Agents use LLMs as reasoning engines to decide which tools to call and in what order.

### ReAct Agent

The ReAct (Reasoning + Acting) agent interleaves thinking steps with tool calls:
1. **Thought**: Reason about the next action
2. **Action**: Choose a tool and provide input
3. **Observation**: Receive the tool output
4. Repeat until the task is complete

### Tool Integration

Any Python function can be wrapped as a LangChain tool using the `@tool` decorator:

```python
from langchain.tools import tool

@tool
def search_database(query: str) -> str:
    """Search the internal knowledge base for relevant information."""
    return db.query(query)
```

## Document Loaders

LangChain ships with loaders for dozens of file formats:

- `TextLoader` — plain `.txt` files
- `UnstructuredMarkdownLoader` — Markdown with structure preservation
- `PyPDFLoader` — PDF documents
- `CSVLoader` — tabular data
- `WebBaseLoader` — web pages via HTTP

## Retrieval-Augmented Generation (RAG)

RAG grounds LLM responses in factual external knowledge. The pipeline:

1. Load and split documents into chunks
2. Embed chunks with a dense encoder
3. Store embeddings in a vector database
4. At query time, retrieve top-k relevant chunks
5. Inject retrieved chunks into the LLM prompt

This reduces hallucination and keeps responses up-to-date without retraining.
