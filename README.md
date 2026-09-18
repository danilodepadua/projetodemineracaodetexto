# Text Mining

Environment for text analysis and machine learning experiments using the data available in `data/`.

## Start with devcontainer

You can open it in a devcontainer by clicking here: [![Open in Dev Container](https://img.shields.io/badge/Open%20in-Dev%20Container-blue?logo=visual-studio-code)](https://code.visualstudio.com/docs/devcontainers/containers)

or,

1. Install Docker and the VS Code **Dev Containers** extension.
2. Open this repository in VS Code.
3. Run `Dev Containers: Reopen in Container` from the Command Palette.

When the container is created, the project automatically creates the `.venv` environment and installs the dependencies from `requirements.txt`.

## Use outside the container

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Activate the environment when you want to run commands directly:

```bash
source .venv/bin/activate
```

From then on you can run the code as you desire.

## Update dependencies

Edit `requirements.txt` when adding or updating a dependency, then run:

```bash
.venv/bin/python -m pip install --upgrade -r requirements.txt
```

Then, in VS Code, use `Dev Containers: Rebuild Container` to recreate the container environment from scratch.

To record the locally installed versions:

```bash
.venv/bin/python -m pip freeze > requirements-lock.txt
```

The lock file is optional; keep `requirements.txt` as the direct dependency list for the project.

## Guides

- [Training](docs/trainning.md)
- [Evaluation](docs/evaluate.md)
- [Hyperparameter tuning](docs/tune.md)