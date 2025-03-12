# OpenPipe Agent Reinforcement Training (ART)

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

Make sure you are on a machine with at least 1xH100 or 1xA100-80GB GPU.

Reinstall torchtune due to a CLI naming conflict:

```bash
uv remove torchtune
uv add torchtune
```

### "Temporal Clue" example

Now you can run the "Temporal Clue" example in `/examples/temporal-clue.ipynb`.

It has been tested with the Hermes 2 Theta Llama 3 8B model on a 1xH100 instance. You should see immediate improvement in `val/reward` after one iteration.
