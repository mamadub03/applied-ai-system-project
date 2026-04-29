"""
RAG Tool — AI Advanced High School Student Handbook Assistant

Retrieval uses TF-IDF vectors + cosine similarity.
No API key. No internet connection. Only numpy required.

How it works:
  1. The handbook is split into overlapping word chunks.
  2. Each chunk is converted into a TF-IDF vector (a list of numbers
     where each number represents how important a word is in that chunk
     relative to the whole document).
  3. The user's query is converted into the same kind of vector.
  4. Cosine similarity measures the angle between the query vector and
     every chunk vector — a score of 1.0 means a perfect match.
  5. The top-K highest-scoring chunks are printed as results.
"""

import os
import numpy as np
from collections import Counter

HANDBOOK_PATH = "handbook.txt"
CHUNK_SIZE    = 150   # words per chunk
CHUNK_OVERLAP = 30    # words shared between adjacent chunks
TOP_K         = 3     # results to display


# ── Tokenization ───────────────────────────────────────────────────────────────

def tokenize(text):
    """Lowercase and strip punctuation, return list of words."""
    cleaned = ""
    for ch in text.lower():
        if ch.isalnum() or ch == " ":
            cleaned += ch
        else:
            cleaned += " "
    return [w for w in cleaned.split() if w]


# ── Chunking ───────────────────────────────────────────────────────────────────

def load_and_chunk(path, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Read the handbook and return overlapping word-based chunks."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ── TF-IDF Vectorizer ──────────────────────────────────────────────────────────

class TFIDFVectorizer:
    """
    Builds a vocabulary from a list of documents and converts any text
    into a TF-IDF vector that can be compared with cosine similarity.

    TF  (term frequency)  = how often a word appears in THIS chunk
    IDF (inverse doc freq) = log(total chunks / chunks containing the word)
                             rare words that only appear in a few chunks
                             get a higher IDF score, so they carry more weight.
    """

    def __init__(self):
        self.vocab = {}          # word -> column index
        self.idf = np.array([])

    def fit(self, documents):
        """Build vocabulary and compute IDF from a list of documents."""
        tokenized = [tokenize(doc) for doc in documents]
        n_docs = len(tokenized)

        # collect every unique word
        all_words = {word for tokens in tokenized for word in tokens}
        self.vocab = {word: idx for idx, word in enumerate(sorted(all_words))}

        # document frequency: how many chunks contain each word
        df = np.zeros(len(self.vocab))
        for tokens in tokenized:
            for word in set(tokens):
                if word in self.vocab:
                    df[self.vocab[word]] += 1

        # IDF with +1 smoothing to avoid division by zero
        self.idf = np.log((n_docs + 1) / (df + 1)) + 1.0

    def transform(self, text):
        """Convert a piece of text into a TF-IDF vector."""
        tokens = tokenize(text)
        if not tokens:
            return np.zeros(len(self.vocab))

        counts = Counter(tokens)
        tf = np.zeros(len(self.vocab))
        for word, count in counts.items():
            if word in self.vocab:
                tf[self.vocab[word]] = count / len(tokens)

        return tf * self.idf


# ── Cosine Similarity ──────────────────────────────────────────────────────────

def cosine_similarity(a, b):
    """
    Measures how similar two vectors point in the same direction.
    1.0 = identical direction (very relevant)
    0.0 = perpendicular (unrelated)
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ── Single-document Retriever (used by CLI and tests) ─────────────────────────

class HandbookRetriever:
    """Indexes the handbook and retrieves the most relevant chunks for a query."""

    def __init__(self):
        self.vectorizer = TFIDFVectorizer()
        self.chunks = []
        self.chunk_vectors = np.array([])

    def index(self, path):
        print("Reading '{}'...".format(path))
        self.chunks = load_and_chunk(path)

        print("Building TF-IDF index over {} chunks...".format(len(self.chunks)))
        self.vectorizer.fit(self.chunks)
        self.chunk_vectors = np.array(
            [self.vectorizer.transform(chunk) for chunk in self.chunks]
        )
        print("Done. Vocabulary size: {} words.\n".format(len(self.vectorizer.vocab)))

    def search(self, query, top_k=TOP_K):
        """Return [(similarity_score, chunk_text), ...] highest score first."""
        query_vec = self.vectorizer.transform(query)
        scores = [cosine_similarity(query_vec, cv) for cv in self.chunk_vectors]
        ranked = sorted(zip(scores, self.chunks), key=lambda x: x[0], reverse=True)
        return ranked[:top_k]


# ── Multi-document Retriever (used by Streamlit app) ──────────────────────────

class MultiDocRetriever:
    """
    Indexes multiple documents together and tracks which source each chunk
    came from. Rebuilds the TF-IDF index whenever documents are added or
    removed so IDF scores always reflect the full loaded corpus.
    """

    def __init__(self):
        self.vectorizer = TFIDFVectorizer()
        # parallel lists: each position i holds data for chunk i
        self._chunk_texts = []   # raw text
        self._chunk_sources = [] # filename it came from
        self.chunk_vectors = np.array([])
        self.sources_loaded = [] # filenames currently indexed

    def load_document(self, path, source_name=None):
        """Add one document to the corpus. Call rebuild() after all docs are loaded."""
        name = source_name or os.path.basename(path)
        chunks = load_and_chunk(path)
        for chunk in chunks:
            self._chunk_texts.append(chunk)
            self._chunk_sources.append(name)
        if name not in self.sources_loaded:
            self.sources_loaded.append(name)

    def load_text(self, text, source_name):
        """Add raw text (e.g. from an uploaded file) to the corpus."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP
        for chunk in chunks:
            self._chunk_texts.append(chunk)
            self._chunk_sources.append(source_name)
        if source_name not in self.sources_loaded:
            self.sources_loaded.append(source_name)

    def rebuild(self):
        """Fit TF-IDF on all loaded chunks and build the vector matrix."""
        self.vectorizer.fit(self._chunk_texts)
        self.chunk_vectors = np.array(
            [self.vectorizer.transform(t) for t in self._chunk_texts]
        )

    def clear(self):
        self._chunk_texts = []
        self._chunk_sources = []
        self.chunk_vectors = np.array([])
        self.sources_loaded = []

    @property
    def total_chunks(self):
        return len(self._chunk_texts)

    @property
    def vocab_size(self):
        return len(self.vectorizer.vocab)

    def search(self, query, top_k=TOP_K):
        """Return [(score, chunk_text, source_name), ...] highest score first."""
        if self.total_chunks == 0:
            return []
        query_vec = self.vectorizer.transform(query)
        scores = [cosine_similarity(query_vec, cv) for cv in self.chunk_vectors]
        triples = list(zip(scores, self._chunk_texts, self._chunk_sources))
        triples.sort(key=lambda x: x[0], reverse=True)
        return triples[:top_k]


# ── CLI ────────────────────────────────────────────────────────────────────────

def wrap(text, width=76, indent="  "):
    """Simple word-wrap for terminal display."""
    words = text.split()
    lines, line = [], []
    for word in words:
        line.append(word)
        if len(" ".join(line)) > width:
            lines.append(indent + " ".join(line[:-1]))
            line = [word]
    if line:
        lines.append(indent + " ".join(line))
    return "\n".join(lines)


def main():
    retriever = HandbookRetriever()
    retriever.index(HANDBOOK_PATH)

    print("=" * 60)
    print("  AI Advanced High School — Student Handbook Search")
    print("=" * 60)
    print("Ask a question or type keywords to find the most relevant")
    print("handbook passages. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        results = retriever.search(query)

        if results[0][0] == 0.0:
            print("\nNo matching passages found. Try different keywords.\n")
            continue

        print("\nTop {} result(s) for: \"{}\"\n".format(len(results), query))
        for rank, (score, chunk) in enumerate(results, start=1):
            print("  Result {}  (similarity score: {:.3f})".format(rank, score))
            print("  " + "-" * 56)
            print(wrap(chunk))
            print()
        print("=" * 60)


if __name__ == "__main__":
    main()
