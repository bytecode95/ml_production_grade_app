"""Launch the ZenML training pipeline."""

import argparse
import logging

from pipelines.zenml.training_pipeline import training_pipeline
from price_prediction.observability.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the validated Ames Housing CSV",
    )
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["baseline", "ridge", "random_forest", "extra_trees"],
    )
    parser.add_argument("--version", default="candidate-001")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging()
    logging.getLogger(__name__).info(
        "Launching pipeline for model=%s version=%s",
        args.model,
        args.version,
    )

    run = training_pipeline(
        data_path=args.data,
        model_name=args.model,
        model_version=args.version,
        random_state=args.seed,
    )

    # ZenML's pipeline invocation starts the run. Inspect the ZenML UI
    # or CLI for detailed run and step metadata.
    logging.getLogger(__name__).info(
        "Pipeline invocation completed: %s",
        run,
    )


if __name__ == "__main__":
    main()