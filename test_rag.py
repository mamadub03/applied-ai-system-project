"""
Tests for the RAG retrieval tool.

Each test sends a query and checks that at least one of the expected
keywords appears somewhere in the top-K results. Run with:

    python test_rag.py
"""

from rag_tool import HandbookRetriever, HANDBOOK_PATH

# ── Test Cases ─────────────────────────────────────────────────────────────────
# Each entry: (query, [keywords that must appear in the combined results])

TEST_CASES = [
    (
        "how many unexcused absences before I get in trouble",
        ["unexcused", "absences", "tardy"],
    ),
    (
        "what is the GPA needed to join extracurricular clubs",
        ["2.5", "extracurricular", "probation"],
    ),
    (
        "how many community service hours do I need to graduate",
        ["40", "community", "service"],
    ),
    (
        "can I use ChatGPT or AI tools on my homework",
        ["disclose", "ai", "chatgpt"],
    ),
    (
        "what happens if I cheat or plagiarize",
        ["plagiarism", "cheating", "integrity", "zero"],
    ),
    (
        "what time does school start on Wednesdays",
        ["wednesday", "9:30", "late"],
    ),
    (
        "how do AP courses affect my GPA",
        ["ap", "weighted", "0.5", "honors"],
    ),
    (
        "what are the lunch hours",
        ["lunch", "11", "12"],
    ),
]


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_tests(retriever):
    passed = 0
    failed = 0

    print("\nRunning {} tests...\n".format(len(TEST_CASES)))
    print("{:<55} {}".format("Query", "Result"))
    print("-" * 70)

    for query, expected_keywords in TEST_CASES:
        results = retriever.search(query, top_k=5)

        # combine all retrieved chunk text into one string for keyword checking
        combined = " ".join(chunk for _, chunk in results).lower()

        missing = [kw for kw in expected_keywords if kw.lower() not in combined]

        if not missing:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL  (missing: {})".format(", ".join(missing))
            failed += 1

        # truncate long queries for display
        display_query = query if len(query) <= 52 else query[:49] + "..."
        print("{:<55} {}".format(display_query, status))

    print("-" * 70)
    print("Results: {}/{} passed\n".format(passed, len(TEST_CASES)))

    if failed == 0:
        print("All tests passed.")
    else:
        print("{} test(s) failed. The retriever may not be surfacing the right".format(failed))
        print("chunks for those queries — try adjusting CHUNK_SIZE or TOP_K in rag_tool.py.")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  RAG Tool — Retrieval Tests")
    print("=" * 70)

    retriever = HandbookRetriever()
    retriever.index(HANDBOOK_PATH)

    run_tests(retriever)
