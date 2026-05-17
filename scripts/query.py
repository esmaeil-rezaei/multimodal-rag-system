
# import argparse
# from src.query.pipeline import QueryPipeline
# from src.utils.logger import get_logger

# logger = get_logger("scripts.query")


# def run_single_query(pipeline: QueryPipeline, question: str) -> None:
#     """Run a single query and print the result."""

#     print(f"\nQuery: {question}")
#     print("─" * 60)

#     result = pipeline.query(raw_query=question)

#     print(f"\nAnswer:\n{result.answer}")

#     if result.citations:
#         print(f"\nCitations ({len(result.citations)}):")
#         for cite in result.citations:
#             src = cite.get("source_name") or cite.get("source_file") or "unknown"
#             ts = (cite.get("ingestion_ts") or "")[:10]
#             print(f"  • [{cite['chunk_id'][:8]}…] {src} ({ts})")

#     if result.faithfulness_score is not None:
#         print(f"\nFaithfulness: {result.faithfulness_score:.2f}")

#     if result.has_conflict:
#         print(f"\nConflict: {result.conflict_resolution}")

#     print()


# def interactive_repl(pipeline: QueryPipeline) -> None:
#     """Interactive query mode (REPL)."""

#     history = []

#     print("\nRAG Interactive Mode (type 'exit' to quit)\n")

#     while True:
#         try:
#             question = input("You: ").strip()
#         except (EOFError, KeyboardInterrupt):
#             print("\nGoodbye!")
#             break

#         if not question:
#             continue
#         if question.lower() in {"exit", "quit", "q"}:
#             print("Goodbye!")
#             break

#         result = pipeline.query(
#             raw_query=question,
#             conversation_history=history,
#         )

#         print(f"\nAssistant: {result.answer}\n")

#         history.append({"role": "user", "content": question})
#         history.append({"role": "assistant", "content": result.answer})


# def main() -> None:
#     parser = argparse.ArgumentParser(description="RAG query CLI")

#     parser.add_argument(
#         "question",
#         nargs="?",
#         help="Run a single query (omit for interactive mode)",
#     )
#     parser.add_argument(
#         "--interactive",
#         action="store_true",
#         help="Start interactive mode",
#     )

#     args = parser.parse_args()

#     pipeline = QueryPipeline()

#     if args.interactive or not args.question:
#         interactive_repl(pipeline)
#     else:
#         run_single_query(pipeline, args.question)


# if __name__ == "__main__":
#     main()




import argparse
import asyncio
from src.agents.orchestrator import RAGOrchestrator
from src.utils.logger import get_logger

logger = get_logger("scripts.query")


def run_single_query(orchestrator: RAGOrchestrator, question: str) -> None:
    """Run a single query and print the result."""

    print(f"\nQuery: {question}")
    print("─" * 60)

    result = asyncio.run(orchestrator.run(raw_query=question))

    print(f"\nAnswer:\n{result.answer}")

    if result.citations:
        print(f"\nCitations ({len(result.citations)}):")
        for cite in result.citations:
            src = cite.get("source_name") or cite.get("source_file") or "unknown"
            ts = (cite.get("ingestion_ts") or "")[:10]
            print(f"  • [{cite['chunk_id'][:8]}…] {src} ({ts})")

    if result.faithfulness_score is not None:
        print(f"\nFaithfulness: {result.faithfulness_score:.2f}")

    if result.has_conflict:
        print(f"\nConflict: {result.conflict_resolution}")

    print()


def interactive_repl(orchestrator: RAGOrchestrator) -> None:
    """Interactive query mode (REPL)."""

    history = []

    print("\nRAG Interactive Mode (type 'exit' to quit)\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        result = asyncio.run(
            orchestrator.run(
                raw_query=question,
                conversation_history=history,
            )
        )

        print(f"\nAssistant: {result.answer}\n")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result.answer})


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG query CLI")

    parser.add_argument(
        "question",
        nargs="?",
        help="Run a single query (omit for interactive mode)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive mode",
    )

    args = parser.parse_args()

    orchestrator = RAGOrchestrator()

    if args.interactive or not args.question:
        interactive_repl(orchestrator)
    else:
        run_single_query(orchestrator, args.question)


if __name__ == "__main__":
    main()