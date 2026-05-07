import argparse
from src.utils.logger import get_logger, set_correlation_id
from src.ingestion.pipeline import IngestionPipeline

logger = get_logger("scripts.ingest")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ingestion pipeline"
    )

    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Tenant namespace",
    )

    args = parser.parse_args()

    set_correlation_id()
    logger.info(f"Starting ingestion (namespace={args.namespace})")

    pipeline = IngestionPipeline()
    stats = pipeline.run(namespace=args.namespace)


if __name__ == "__main__":
    main()