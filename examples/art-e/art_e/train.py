import art
from art.local import LocalBackend
import asyncio
from dotenv import load_dotenv
from typing import List, cast
from rollout import rollout
from art_e.data.query_iterators import load_synthetic_queries
from art_e.data.types_enron import SyntheticQuery
from art_e.data.local_email_db import generate_database
from art.utils import iterate_dataset
from art_e.project_types import ProjectPolicyConfig
from art_e.evaluate.benchmark import benchmark_model
from art_e.judge_group import judge_group
from art_e.rollout import ProjectTrajectory
import os

load_dotenv()

# First, I defined a trainable model. The `ProjectPolicyConfig` contains the
# specific parameters I varied between runs for this project. They're
# interpreted in the rollout defined in `rollout.py` and used to control
# generation.


async def train(model: art.TrainableModel[ProjectPolicyConfig]):
    generate_database()

    if model.config.training_config is None:
        raise ValueError("Training config is not set")
    with LocalBackend() as backend:
        print(f"Pulling from S3 bucket: `{os.environ['BACKUP_BUCKET']}`")
        await backend._experimental_pull_from_s3(
            model,
            s3_bucket=os.environ["BACKUP_BUCKET"],
            verbose=True,
        )
        await model.register(backend)

        print("Loading training data...")
        train_scenarios: List[SyntheticQuery] = load_synthetic_queries(
            split="train", limit=model.config.training_config.training_dataset_size
        )
        print("Loading validation data...")
        val_scenarios: List[SyntheticQuery] = load_synthetic_queries(
            split="test", limit=model.config.training_config.val_set_size
        )

        print(f"Training data size: {len(train_scenarios)}")
        print(f"Validation data size: {len(val_scenarios)}")

        train_iterator = iterate_dataset(
            train_scenarios,
            groups_per_step=model.config.training_config.groups_per_step,
            num_epochs=model.config.training_config.num_epochs,
            initial_step=await model.get_step(),
        )

        for batch, epoch, global_step, epoch_step in train_iterator:
            if global_step % model.config.training_config.eval_steps == 0:
                print(f"\n--- Evaluating at Iteration {global_step} ---")
                await benchmark_model(model)
                await model.delete_checkpoints()
                await backend._experimental_push_to_s3(
                    model,
                    s3_bucket=os.environ["BACKUP_BUCKET"],
                )

            groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        (
                            rollout(model, scenario)
                            for _ in range(
                                model.config.training_config.trajectories_per_group
                            )
                        )
                    )
                    for scenario in batch
                )
            )

            # Optionally rescore each trajectory group with the LLM-judge before training.
            if model.config.training_config.rescore_with_judge_group:
                # Run the rescoring concurrently for better throughput.
                await asyncio.gather(
                    *(
                        judge_group(cast(list[ProjectTrajectory], g.trajectories))
                        for g in groups
                    )
                )

            await model.train(
                groups,
                config=art.TrainConfig(
                    learning_rate=model.config.training_config.learning_rate
                ),
            )

        await benchmark_model(model)
        await backend._experimental_push_to_s3(
            model,
            s3_bucket=os.environ["BACKUP_BUCKET"],
        )
        print("Training finished.")


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("model_json", help="JSON string serialization of the Model")
    args = parser.parse_args()

    print("Model JSON: ", args.model_json)

    model_dict = json.loads(args.model_json)
    model_dict["config"] = ProjectPolicyConfig(**model_dict["config"])
    model: art.TrainableModel[ProjectPolicyConfig] = art.TrainableModel(
        **model_dict,
    )
    asyncio.run(train(model))
