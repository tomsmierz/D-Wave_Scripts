# D-Wave_Scripts
Scripts for D-Wave

# Setting up this repository
For managing package and Python versions tool named [poetry](https://github.com/python-poetry/poetry) is used.

In order to create Python virtual environment, install poetry and all other packages please follow:

```bash
python -m venv .venv
source .venv/bin/activate
pip install poetry
poetry install
```

Please make sure that your currently active Python version matches one specified in `pyproject.toml` file.
File `pyproject.toml` is used to specify the package versions in use, as well as the Python's versions.
