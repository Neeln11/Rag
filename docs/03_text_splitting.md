# Text Splitting Strategies for RAG

Chunking strategy is one of the most impactful decisions in a RAG pipeline. Poor chunking
leads to context fragmentation, retrieval misses, and incoherent LLM responses.

## Why Chunking Matters

LLMs and embedding models have fixed context windows. A 100-page PDF must be broken into
manageable chunks that:
1. Fit within the embedding model's max token limit (typically 512 tokens)
2. Preserve semantic coherence (don't split mid-sentence or mid-concept)
3. Provide enough context for the LLM to generate a useful response

## RecursiveCharacterTextSplitter

The `RecursiveCharacterTextSplitter` is the recommended general-purpose splitter in LangChain.
It tries each separator in order, falling back to the next if the current one produces chunks
that are too large.

### How It Works

Given `separators=["\n## ", "\n### ", "\n\n", "\n", " "]`:

1. First, split on `\n## ` (Markdown H2 headers) — preserves section boundaries
2. If sections are still too large, split on `\n### ` (H3 headers)
3. If still too large, split on double newlines (paragraph boundaries)
4. Fall back to single newlines, then spaces as a last resort

This hierarchy ensures semantically meaningful splits before resorting to arbitrary cuts.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,       # Target chunk size in characters
    chunk_overlap=100,    # Overlap to prevent context loss at boundaries
    separators=["\n## ", "\n### ", "\n\n", "\n", " "]
)
```

### Chunk Size Selection

- **Too small** (< 200 chars): Chunks lose context; retrieval is noisy
- **Too large** (> 1000 chars): Reduces precision; retrieved chunks may be irrelevant
- **Optimal range**: 400–800 characters for technical documentation

### Chunk Overlap

Overlap of 10–20% of chunk size ensures that concepts spanning chunk boundaries are
captured by at least one chunk. For `chunk_size=600`, an overlap of 100 characters (17%)
is a good balance.

## Markdown-Aware Splitting

Markdown has rich structure that splitters can exploit:

```markdown
# Title        ← H1: top-level document split
## Section     ← H2: major section split
### Subsection ← H3: sub-section split

Paragraph text...  ← Paragraph split

- List item    ← List items stay together when possible
```

By including `\n## ` and `\n### ` as high-priority separators, the splitter respects
the document's intended hierarchy rather than making arbitrary cuts.

## Alternative Splitters

### TokenTextSplitter

Splits on token boundaries (more accurate for LLM context windows):

```python
from langchain.text_splitter import TokenTextSplitter
splitter = TokenTextSplitter(chunk_size=150, chunk_overlap=20)
```

Requires `tiktoken` or a HuggingFace tokenizer. More accurate but slower.

### SemanticChunker

Uses embedding similarity to find natural breakpoints:

```python
from langchain_experimental.text_splitter import SemanticChunker
splitter = SemanticChunker(embeddings=embedding_model)
```

Most semantically accurate but significantly slower — suitable for offline preprocessing
of high-value documents where quality trumps speed.

### MarkdownHeaderTextSplitter

Splits exclusively on headers, preserving full section context:

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers = [("#", "H1"), ("##", "H2"), ("###", "H3")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
```

Best when documents are well-structured Markdown and you want metadata per section.

## Metadata Enrichment During Splitting

Adding metadata to each chunk enables powerful filtering at retrieval time:

```python
docs = splitter.create_documents(
    texts=[raw_text],
    metadatas=[{
        "source": "manual.md",
        "chunk_index": i,
        "total_chunks": total,
        "doc_type": "markdown",
        "ingest_timestamp": datetime.utcnow().isoformat()
    }]
)
```

This metadata travels with the chunk through embedding and storage, allowing downstream
retrieval to filter by source, date, or document type.
