# RAG (Retrieval-Augmented Generation) Comprehensive Assignment Documentation

Repository: [https://github.com/cvijayvinoth/RAG](https://github.com/cvijayvinoth/RAG)

---

## Table of Contents
1. [01. Introduction to RAG](#01-introduction-to-rag)
2. [02. Types of RAG](#02-types-of-rag)
3. [03. Self RAG](#03-self-rag)
4. [04. Corrective RAG](#04-corrective-rag)
5. [05. Fusion RAG](#05-fusion-rag)
6. [06. BM25 + ReRank (Hybrid Search)](#06-bm25--rerank-hybrid-search)
7. [07. RAG Demo (`rag_app.py`)](#07-rag-demo-rag_apppy)
8. [08. RAG Demo with LangSmith](#08-rag-demo-with-langsmith)

---

## 01. Introduction to RAG

### Overview
**Retrieval-Augmented Generation (RAG)** is an architectural pattern that enhances Large Language Models (LLMs) by connecting them to external knowledge sources. Instead of relying solely on parametric memory learned during pre-training, RAG dynamically fetches relevant context from private or real-time datasets to ground the model's answer.

### Key Benefits
* **Reduces Hallucinations**: Grounding answers in retrieved context ensures factual accuracy.
* **Access to Private Data**: Enables models to answer questions using private enterprise documents.
* **Cost Efficiency**: Keeps knowledge up-to-date without needing costly fine-tuning or full pre-training runs.

### Standard RAG Workflow

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ User Query  ├────►│ Document Search ├────►│ Relevant Context │
└─────────────┘     └─────────────────┘     └────────┬─────────┘
                                                     │
                                                     ▼
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   Response  │◄────┤    LLM Model    │◄────┤  Prompt + Context│
└─────────────┘     └─────────────────┘     └──────────────────┘
```

1. **Ingestion**: Documents are chunked, embedded into vectors, and stored in a vector store.
2. **Retrieval**: User queries are transformed into vectors to find top-$K$ matching chunks.
3. **Augmentation**: Retrieved chunks are injected into a structured prompt template alongside the user query.
4. **Generation**: The LLM reads the context and generates an accurate, grounded answer.

---

## 02. Types of RAG

As RAG applications evolved beyond basic document querying, three primary architectural patterns emerged: **Naive RAG**, **Advanced RAG**, and **Modular RAG**.

### Architecture Comparison

| Feature | Naive RAG | Advanced RAG | Modular RAG |
| :--- | :--- | :--- | :--- |
| **Pipeline Flow** | Linear (Retrieve $\rightarrow$ Read) | Pre & Post-Retrieval optimization | Dynamic, agentic, non-linear routes |
| **Indexing** | Basic text chunking | Hierarchical indexing, metadata filtering | Multi-index & multi-modal vector stores |
| **Query Processing**| Raw query matching | Query rewrite, decomposition | Dynamic intent routing |
| **Reranking** | None | Cross-Encoder / Reciprocal Rank Fusion | Adaptive contextual re-ranking |

### 1. Naive RAG
The standard 3-step pipeline: split text $\rightarrow$ embed $\rightarrow$ retrieve top-$K$ chunks $\rightarrow$ pass to LLM. Vulnerable to low retrieval precision and context loss.

### 2. Advanced RAG
Addresses Naive RAG weaknesses through targeted optimizations:
* **Pre-Retrieval**: Query expansion, query rewriting, and sentence window chunking.
* **Post-Retrieval**: Re-ranking retrieved chunks using cross-encoders and context compression to filter irrelevant noise.

### 3. Modular RAG
Integrates discrete components that can be dynamically routed. Enables features like fallback web search, SQL database querying, memory routing, and iterative reflection loops.

---

## 03. Self RAG

### Overview
**Self-RAG** is an advanced framework that enables an LLM to dynamically evaluate its own retrieval and generation steps using special **reflection tokens**. Instead of blindly retrieving documents for every query, Self-RAG decides *if* retrieval is needed, *whether* the retrieved passages are relevant, and *if* the generated answer contains factual hallucinations.

### Reflection Tokens

Self-RAG introduces four standard reflection tags during generation:
1. **`[Retrieve]`**: Determines if external knowledge search is necessary. Values: `{IsREL, NoREL, Continue}`.
2. **`[IsREL]`**: Evaluates whether a retrieved passage is relevant to the query. Values: `{Relevant, Irrelevant}`.
3. **`[IsSUP]`**: Checks if the generated answer text is supported by the retrieved context. Values: `{FullySupported, PartiallySupported, NoSupport}`.
4. **`[IsUSE]`**: Rates the overall utility of the generated response. Values: `{5, 4, 3, 2, 1}`.

---

## 04. Corrective RAG

### Overview
**Corrective RAG (CRAG)** is a self-correcting RAG pattern designed to handle poor retrieval performance. If a vector database returns low-quality or irrelevant chunks, CRAG triggers a fallback strategy (such as an external web search API) to fetch high-quality context instead of feeding bad data into the LLM.

### Retrieval Evaluator Outcomes

When documents are retrieved from the vector store, an evaluator model assigns a confidence score to classify the documents into three execution branches:
1. **Correct**: The retrieved context directly answers the user prompt. Low-scoring sections are stripped, and the clean text is sent straight to generation.
2. **Incorrect**: The vector store failed to retrieve relevant documents. The chunks are discarded, and an external search tool (e.g., Tavily, Serper) is invoked to get context from the web.
3. **Ambiguous**: Retrieval quality is uncertain. CRAG combines the top vector store chunks with web search results to ensure full context coverage.

---

## 05. Fusion RAG

### Overview
**Fusion RAG** addresses a key limitation of vector search: user queries are often ambiguous, under-specified, or missing domain terminology. RAG Fusion overcomes this by taking a single user input, generating multiple variations using an LLM, searching the vector database across all variations, and re-ranking the combined results using **Reciprocal Rank Fusion (RRF)**.

### RAG Fusion Pipeline Steps
1. **Query Generation**: An LLM generates 3–5 different variations of the original user prompt.
2. **Vector Retrieval**: Each query variation executes its own vector similarity search in parallel.
3. **Reciprocal Rank Fusion (RRF)**: A scoring algorithm merges and re-ranks all retrieved document lists to push the highest-quality chunks to the top.
4. **Final Context Assembly**: The top-ranked unique chunks are combined into a final context block for answer generation.

### Reciprocal Rank Fusion (RRF) Formula

$$\text{RRF\_Score}(d \in D) = \sum_{q \in Q} \frac{1}{k + r_q(d)}$$

---

## 06. BM25 + ReRank (Hybrid Search)

### Overview
Combining **Dense Retrieval** (semantic embeddings) with **Sparse Retrieval** (keyword matching via BM25) creates a robust search pipeline. Sparse search catches exact matches like serial numbers, codes, or obscure names, while dense search captures conceptual intent. Adding a **Cross-Encoder ReRanker** on top ensures only the most relevant passages reach the LLM.

### Implementation Example
```python
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings

# 1. Setup Dense Retriever
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 2. Setup Sparse Retriever
bm25_retriever = BM25Retriever.from_texts(
    texts=["LangChain documentation", "Python code guide"], k=5
)

# 3. Create Ensemble (Hybrid) Search
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5, 0.5]
)
```

---

## 07. RAG Demo (`rag_app.py`)

Below is the implementation code for `rag_app.py` utilizing the LangChain Expression Language (LCEL) framework.

```python
# rag_app.py
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def run_rag_pipeline(data_file_path: str, user_query: str):
    # 1. Data Ingestion
    loader = TextLoader(data_file_path)
    documents = loader.load()

    # 2. Document Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Vector Indexing
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma.from_documents(chunks, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 4. Prompt Engineering
    prompt_template = """Answer the question based only on the provided context below.

Context:
{context}

Question:
{question}
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 5. LCEL Chain Assembly
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Pipeline Execution
    return rag_chain.invoke(user_query)

if __name__ == "__main__":
    result = run_rag_pipeline("sample_data.txt", "What are the core features of LangChain?")
    print("RAG Response:\n", result)
```

---

## 08. RAG Demo with LangSmith

### Environment Setup for Tracing
```bash
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
export LANGCHAIN_API_KEY="your_langsmith_api_key_here"
export LANGCHAIN_PROJECT="rag-assignment-project"
```

### Trace Execution Hierarchy
When running `rag_app.py` with LangSmith enabled, the execution sequence is automatically recorded:
```text
RunnableSequence (rag_chain)
├── RunnableParallel
│   ├── VectorStoreRetriever (Retrieves Top-3 Chunks)
│   └── RunnablePassthrough (Passes User Query)
├── ChatPromptTemplate (Formats Context & Query)
├── ChatOpenAI (Generates LLM Completion)
└── StrOutputParser (Outputs Clean String Response)
```
