# AI Advanced High School — Student Handbook RAG Assistant

## Title and Summary

This project is a Retrieval-Augmented Generation (RAG) tool that allows students at AI Advanced High School to ask natural-language questions about their student handbook and receive accurate, grounded answers. Instead of scrolling through a long policy document, students type a question and the system finds the most relevant handbook sections, then uses Claude to generate a clear answer from that context. It is based on the RAG pattern introduced in Module 4: rather than fine-tuning a model, we supply retrieved text as context at inference time.

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────┐       cosine similarity        ┌──────────────────────┐
│  Retriever  │ ◄───── encoded query ─────────► │  Chunk Embeddings    │
│ (sentence-  │                                  │  (all-MiniLM-L6-v2)  │
│ transformers│ ──── top-K chunks ────────────►  └──────────────────────┘
└─────────────┘
    │
    ▼  (retrieved context + question)
┌─────────────┐
│  Generator  │
│  (Claude    │  ──── grounded answer ──────► User
│  Haiku API) │
└─────────────┘
```

1. **Chunking** — `handbook.txt` is split into overlapping 150-word chunks so no information is cut off at a boundary.
2. **Embedding** — Each chunk is encoded into a dense vector using `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no extra API key needed).
3. **Retrieval** — The user's query is also embedded; cosine similarity ranks all chunks and the top 3 are selected.
4. **Generation** — The 3 retrieved chunks are passed to Claude Haiku as context. Claude answers only from that context, so responses stay grounded in the actual handbook.

---

## Setup Instructions

### 1. Clone / navigate to the project
```bash
cd applied-ai-system-project
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> The first install downloads the `all-MiniLM-L6-v2` embedding model (~80 MB). This only happens once.

### 4. Add your Anthropic API key
```bash
cp .env.example .env
# Then open .env and replace "your-api-key-here" with your real key
```

### 5. Run the assistant
```bash
python rag_tool.py
```

---

## Sample Interactions

**Example 1 — Attendance policy**
```
Your question: What happens if I have more than 10 unexcused absences?

Answer:
If a student accumulates 10 or more unexcused absences in a semester,
they may be required to complete a recovery plan in order to receive
credit for the affected courses.
```

**Example 2 — Graduation requirements**
```
Your question: How many community service hours do I need to graduate?

Answer:
Students must complete a minimum of 40 hours of community service
before graduation. At least 10 of those hours must be related to
technology education or digital equity initiatives.
```

**Example 3 — AI tool usage**
```
Your question: Am I allowed to use ChatGPT on assignments?

Answer:
AI tools like ChatGPT may be used as learning aids unless a teacher
explicitly prohibits their use on a specific assignment. When you use
AI tools, you must disclose which tools you used and how. Submitting
AI-generated content as your own original work without disclosure is a
violation of the Academic Integrity Policy.
```

---

## Design Decisions

| Decision | Trade-off |
|---|---|
| **Local embeddings** (`sentence-transformers`) instead of an embeddings API | No extra API cost or latency for retrieval; slight delay on first run while the model downloads. |
| **Word-based chunking** with 30-word overlap | Simple and language-agnostic; a semantic chunker (split on section headers) would be more precise but requires more code. |
| **Top-3 retrieval** | Balances context quality vs. prompt length. More chunks means more context but also more noise. |
| **Claude Haiku** for generation | Fast and cheap for short Q&A answers; Sonnet would be better for complex multi-step reasoning. |
| **Prompt caching on system message** | The system instruction never changes between queries, so caching it saves tokens and reduces latency on repeated calls. |
| **In-memory vector store** | Sufficient for a single document. A production system would use a persistent store like ChromaDB or Pinecone. |

---

## Testing Summary

- **What worked well:** Retrieval was accurate for specific policy questions (attendance numbers, GPA thresholds, club meeting times). Claude stayed grounded and did not hallucinate details not present in the retrieved chunks.
- **What didn't work as well:** Very broad questions ("What are all the rules?") returned less useful answers because the top-3 chunks only captured a slice of the relevant information.
- **What I learned:** The quality of RAG answers depends heavily on chunk size and retrieval count. Smaller chunks improve retrieval precision but can lose sentence context; larger chunks capture more context but reduce retrieval precision.

---

## Reflection

Building this project made clear that the hardest part of RAG is not the generation step — the LLM handles that well — but the retrieval step. Choosing the right chunk size, overlap, and number of retrieved chunks required experimentation, and there is a real trade-off between recall (getting all the relevant information) and precision (not flooding the LLM with noise). It also showed how powerful grounding an LLM in a specific document can be: the assistant refuses to guess when the answer is not in the handbook, which is exactly the behavior you want in a school policy tool.

---

## Ethics and Responsible AI Reflection

The biggest limitation of this system is that the retriever only knows what is written in the handbook. For example, if a policy is outdated or poorly worded, the tool will surface that flawed information with no way to flag it, and TF-IDF's vocabulary bias means students who phrase questions differently from the handbook text may get weak results, putting less policy-savvy students at a disadvantage. The tool could also be misused: a student could query for exact consequence thresholds and stay just under them, or use it to coach others on avoiding detection for policy violations; prevention strategies include adding a disclaimer on every result to verify answers with a counselor and logging queries so administrators can spot patterns of misuse. The biggest testing surprise was how sensitive retrieval was to small wording differences "what are the lunch hours" failed to surface the right chunk in the top 3 results even though the word "lunch" is in the document, because TF-IDF weighted other terms more heavily, which was a reminder that the answer being in the document does not guarantee the system will find it. As for AI collaboration during this project, one genuinely helpful suggestion was switching to TF-IDF with cosine similarity as the retrieval method, which eliminated the need for any external API or model download and made the tool runnable with only numpy; one flawed suggestion was the original design built around the sentence transformers library and the Anthropic Claude API, which assumed a newer Python environment that did not match the actual Python 3.8 32-bit setup — the install broke immediately and the entire approach had to be redesigned, a good reminder that AI-generated code makes assumptions about your environment that you always need to verify before committing to them.