

import argparse
from src.utils.logger import get_logger, set_correlation_id
from src.ingestion.pipeline import IngestionPipeline

logger = get_logger("scripts.ingest")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest data into the system."
    )

    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Tenant namespace for ACL isolation (default: 'default')",
    )

    args = parser.parse_args()

    set_correlation_id()
    logger.info(f"Starting ingestion for namespace='{args.namespace}'")

    pipeline = IngestionPipeline(args.namespace)


if __name__ == "__main__":
    main()