![Screenshot 2025-03-21 at 10 11 14 PM](https://github.com/user-attachments/assets/5510fc07-82d0-491c-9c47-284b4ad74a51)
# The OpenPipe Agent Reinforcement Trainer (ART)

An open-source reinforcement training library for LLMs and agentic workflows

## Getting Started

Clone the repository:

```bash
git clone https://github.com/OpenPipe/agent-reinforcement-training.git
cd agent-reinforcement-training
```

Install the dependencies:

```bash
uv sync
```

Then follow the SkyPilot or Local Training instructions below.

> **Warning:** There is currently a bug with tool use functionality. The issue appears to be that vLLM does not return all the token log probabilities for tool use. Further investigation is needed to determine the exact cause. For now, teaching use case-specific tool use with non-tool use models is the recommended workaround.

### SkyPilot

Copy the `.env.example` file to `.env` and set the environment variables:

```bash
cp .env.example .env
```

Ensure you have a valid SkyPilot cloud available:

```bash
uv run sky check
```

Launch a cluster:

```bash
./launch-cluster.sh # you can pass any sky launch arguments here
```

SSH into the `art` cluster with VSCode or from the command line:

```bash
ssh art
```

### Local Training

Make sure you are on a machine with at least one H100 or A100-80GB GPU.

Reinstall torchtune due to a CLI naming conflict:

```bash
uv remove torchtune
uv add torchtune
```

### "Temporal Clue" example

Now you can run the "Temporal Clue" example in `/examples/temporal-clue.ipynb`.

It has been tested with the `NousResearch/Hermes-2-Theta-Llama-3-8B` model on a 1xH100 instance.

You can monitor training progress with Weights & Biases at https://wandb.ai/your-wandb-username/agent-reinforcement-training.

You should see immediate improvement in `val/reward` after one iteration.

If you run into any issues, the training output is set to maximum verbosity. Copying the outputs such as the vLLM or torchtune logs, or copying/screenshotting the plotted packed tensors, may help me debug the issue.
