
import argparse
from src.query.pipeline import QueryPipeline
from src.utils.logger import get_logger

logger = get_logger("scripts.query")


def run_single_query(pipeline: QueryPipeline, question: str) -> None:
    """Run a single query and print the result."""

    print(f"\nQuery: {question}")
    print("─" * 60)

    result = pipeline.query(raw_query=question)

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


def interactive_repl(pipeline: QueryPipeline) -> None:
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

        result = pipeline.query(
            raw_query=question,
            conversation_history=history,
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

    pipeline = QueryPipeline()

    if args.interactive or not args.question:
        interactive_repl(pipeline)
    else:
        run_single_query(pipeline, args.question)


if __name__ == "__main__":
    main()







# from __future__ import annotations

# import argparse
# import asyncio
# import sys
# from pathlib import Path
# from typing import Optional

# from src.core.container import init_container
# from src.utils.logger import BOLD as _BOLD, DIM as _DIM, GREEN as _GREEN, AMBER as _AMBER, RED as _RED, CYAN as _CYAN, RESET as _RESET

# _HR = f"{_DIM}" + "─" * 60 + f"{_RESET}"

# def _print_result(result, question: str) -> None:
#     """Render a GenerationResult to the terminal."""
#     print()
#     print(f"{_BOLD}Query{_RESET}   {question}")
#     print(_HR)
#     print()
#     print(result.answer)
#     print()

#     if result.citations:
#         print(f"{_DIM}Sources ({len(result.citations)}){_RESET}")
#         for cite in result.citations:
#             src  = cite.get("source_name") or cite.get("source_file") or "unknown"
#             ts   = (cite.get("ingestion_ts") or "")[:10]
#             cid  = (cite.get("chunk_id") or "")[:8]
#             date = f" {_DIM}·{_RESET} {ts}" if ts else ""
#             print(f"  {_CYAN}[{cid}…]{_RESET} {src}{date}")
#         print()

#     if result.faithfulness_score is not None:
#         score = result.faithfulness_score
#         colour = _GREEN if score >= 0.7 else (_AMBER if score >= 0.4 else _RED)
#         bar_filled = int(score * 10)
#         bar = "█" * bar_filled + "░" * (10 - bar_filled)
#         print(f"{_DIM}Faithfulness{_RESET}  {colour}{bar}{_RESET}  {score:.0%}")

#     if result.has_conflict and result.conflict_resolution:
#         print(f"\n{_AMBER}⚡ Conflict{_RESET}  {result.conflict_resolution}")

#     print()

# async def _run_query(
#     question: str,
#     history: list,
#     namespace: Optional[str],
# ) -> object:
#     """Initialise the container and run a single query."""
#     container = init_container()
#     try:
#         return await container.orchestrator.run(
#             raw_query=question,
#             conversation_history=history,
#             namespace=namespace,
#         )
#     finally:
#         container.shutdown()

# def run_single_query(question: str, namespace: Optional[str]) -> None:
#     result = asyncio.run(_run_query(question, [], namespace))
#     _print_result(result, question)

# def interactive_repl(namespace: Optional[str]) -> None:
#     """Interactive REPL — maintains conversation history across turns."""
#     ns_label = f" {_DIM}[{namespace}]{_RESET}" if namespace else ""
#     print(f"\n{_BOLD}RAG Interactive Mode{_RESET}{ns_label}  "
#           f"{_DIM}(type 'exit' or press Ctrl-C to quit){_RESET}\n")

#     container = init_container()
#     history: list = []

#     try:
#         while True:
#             try:
#                 question = input(f"{_BOLD}You:{_RESET} ").strip()
#             except (EOFError, KeyboardInterrupt):
#                 print(f"\n{_DIM}Session ended.{_RESET}")
#                 break

#             if not question:
#                 continue
#             if question.lower() in {"exit", "quit", "q", ":q"}:
#                 print(f"{_DIM}Goodbye.{_RESET}")
#                 break

#             try:
#                 result = asyncio.run(
#                     container.orchestrator.run(
#                         raw_query=question,
#                         conversation_history=history,
#                         namespace=namespace,
#                     )
#                 )
#             except Exception as exc:
#                 print(f"{_RED}Error:{_RESET} {exc}\n", file=sys.stderr)
#                 continue

#             _print_result(result, question)
#             history.append({"role": "user",      "content": question})
#             history.append({"role": "assistant", "content": result.answer})
#     finally:
#         container.shutdown()

# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Query the RAG knowledge assistant from the command line.",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog=(
#             "Examples:\n"
#             "  python -m scripts.query 'What are the main findings?'\n"
#             "  python -m scripts.query 'Summarise biomarkers' --namespace Radiology\n"
#             "  python -m scripts.query --interactive --namespace MDLC\n"
#         ),
#     )
#     parser.add_argument(
#         "question",
#         nargs="?",
#         help="Question to answer (omit to enter interactive mode)",
#     )
#     parser.add_argument(
#         "--interactive", "-i",
#         action="store_true",
#         help="Start an interactive conversation REPL",
#     )
#     parser.add_argument(
#         "--namespace", "-n",
#         metavar="NS",
#         default=None,
#         help="Restrict retrieval to a specific namespace (e.g. Radiology)",
#     )

#     args = parser.parse_args()

#     if args.interactive or not args.question:
#         interactive_repl(namespace=args.namespace)
#     else:
#         run_single_query(args.question, namespace=args.namespace)

# if __name__ == "__main__":
#     main()
