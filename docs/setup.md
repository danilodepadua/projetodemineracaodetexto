# Setup

Run project commands from the repository root. The supported Python contract is Python 3.12.x. The application does not require an activated shell, VS Code terminal activation, or a `.env` file.

## Host `.venv` (recommended)

### Linux/macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip check
```

Use the explicit interpreter path for project commands (`.venv/bin/python` or `.venv\Scripts\python.exe`). Activation is optional and is not part of the runtime contract.

Expected result: `pip check` reports no broken requirements. The project `.venv/` is ignored by Git.

## Raw system Python

This is the least isolated option. Install the same dependency contract into the intentionally selected Python 3.12 environment:

```bash
python3.12 -m pip install --upgrade pip
python3.12 -m pip install -r requirements.txt
python3.12 -m pip check
```

Run commands with that same interpreter, for example `python3.12 -m src.preprocessing ...`. Do not use a system Python from another major/minor version.

## Dev Container

The Dev Container uses the Python 3.12 Bookworm image and `vscode` workspace user. On first create, rebuild, or reopen, `.devcontainer/post-create.sh`:

1. resolves the repository root;
2. creates `.venv` if it does not exist, or reuses it;
3. verifies that its interpreter is Python 3.12;
4. upgrades pip and installs `requirements.txt`;
5. runs `pip check`.

VS Code points at `${containerWorkspaceFolder}/.venv/bin/python`, but the Python pipeline does not depend on editor activation. Rebuild the container if bootstrap fails; no manual pip installation should be necessary after a successful lifecycle run.

The Python pipeline does not require Node, Codex, or bubblewrap. They are not installed by the project container lifecycle.

## Dev Containers CLI

The same lifecycle can be used without the VS Code UI:

```bash
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . .venv/bin/python -m pip check
devcontainer exec --workspace-folder . .venv/bin/python -m pytest
```

Use the equivalent `devcontainer` CLI installation documented by its provider. The repository's initialization is the `postCreateCommand`, not an editor task.

## Data placement

Competition data is private and ignored by Git. Place these files under `data/raw/`:

```text
data/raw/
├── train.csv
├── valid.csv
├── test.csv
└── sample_submission.csv
```

`train.csv` and `valid.csv` require `id`, `essay`, `prompt`, and the four target columns. `test.csv` requires `id`, `essay`, and `prompt`. `sample_submission.csv` is required by input validation. Missing files or invalid schemas fail preprocessing with a data-validation error.

## Verification

From any supported environment, run:

```bash
python -m pip check
python -m src.preprocessing --help
python -m src.models.train --help
python -m src.models.evaluate --help
python -m src.models.tune --help
python -m pytest
```

Replace `python` with the explicit host or container interpreter when the shell is not activated.
