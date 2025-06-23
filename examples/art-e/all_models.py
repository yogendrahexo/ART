import art
from art_e.project_types import ProjectPolicyConfig, TrainingConfig

models = {
    "002": art.TrainableModel(
        name="email-agent-002",
        project="email_agent",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=ProjectPolicyConfig(
            max_turns=10,
            log_to_openpipe=True,
            training_config=TrainingConfig(
                trajectories_per_group=6,
                groups_per_step=8,
                learning_rate=1.2e-5,
                eval_steps=30,
                val_set_size=100,
                training_dataset_size=4000,
                num_epochs=1,
            ),
        ),
    )
}


models["004"] = models["002"].model_copy(deep=True)
models["004"].name = "email-agent-004"
models["004"].config.max_turns = 30

models["005"] = models["002"].model_copy(deep=True)
models["005"].name = "email-agent-005"

models["006"] = models["005"].model_copy(deep=True)
models["006"].name = "email-agent-006"

models["007"] = models["005"].model_copy(deep=True)
models["007"].name = "email-agent-007"

models["008"] = models["005"].model_copy(deep=True)
models["008"].name = "email-agent-008"
assert models["008"].config.training_config is not None
models["008"].config.training_config.trajectories_per_group = 4
models["008"].config.training_config.groups_per_step = 12
models["008"].config.training_config.num_epochs = 3

models["011"] = models["008"].model_copy(deep=True)
models["011"].name = "email-agent-011"
assert models["011"].config.training_config is not None
models["011"].config.training_config.num_epochs = 4

models["012"] = models["008"].model_copy(deep=True)
models["012"].name = "email-agent-012"

models["013"] = models["002"].model_copy(deep=True)
models["013"].name = "email-agent-013"
assert models["013"].config.training_config is not None
models["013"].config.training_config.num_epochs = 4
models["013"].config.training_config.trajectories_per_group = 4
models["013"].config.training_config.groups_per_step = 24

models["014"] = models["008"].model_copy(deep=True)
models["014"].name = "email-agent-014"
models["014"].config.stupid_simple_reward_fn = True

models["201"] = models["008"].model_copy(deep=True)
models["201"].name = "email-agent-201"

# Model 202: like 008 but with judge-group rescoring during training
models["202"] = models["008"].model_copy(deep=True)
models["202"].name = "email-agent-202"
# Ensure training config exists and enable the new flag
assert models["202"].config.training_config is not None
models["202"].config.training_config.use_judge_group_variant = "v1"

models["203"] = models["201"].model_copy(deep=True)
models["203"].name = "email-agent-203"

# Model 204: like 202 but with judge-group rescoring variant v2
models["204"] = models["202"].model_copy(deep=True)
models["204"].name = "email-agent-204"
# Ensure training config exists and enable the v2 flag
assert models["204"].config.training_config is not None
models["204"].config.training_config.use_judge_group_variant = "v2"

# Model 205: like 204 but using Gemini 2.5 Flash as the judge-group model
models["205"] = models["204"].model_copy(deep=True)
models["205"].name = "email-agent-205"
# Ensure training config exists and set the judge group model
assert models["205"].config.training_config is not None
models["205"].config.training_config.judge_group_model_name = "gemini/gemini-2.5-flash"

# Model 206: like 204 but using Qwen3 32B as the judge-group model
models["206"] = models["204"].model_copy(deep=True)
models["206"].name = "email-agent-206"
# Ensure training config exists and set the judge group model
assert models["206"].config.training_config is not None
models[
    "206"
].config.training_config.judge_group_model_name = "openrouter/qwen/qwen3-32b"

# Model 207: like 205 but only uses 12 training examples total
models["207"] = models["205"].model_copy(deep=True)
models["207"].name = "email-agent-207"
assert models["207"].config.training_config is not None
models["207"].config.training_config.training_dataset_size = 12
models["207"].config.training_config.num_epochs = 500
