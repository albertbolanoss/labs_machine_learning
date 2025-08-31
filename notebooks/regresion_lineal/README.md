# Machine Learning

## Requirements

### Install Python 3.x

```sh
# Use pyenv to manage different versions of Python in a specific folder
python3 -m pip install --user pipx
pipx install pyenv
pyenv install 3.13.0
pyenv local 3.13.0

# Install a virtual environment to manage the project dependencies
python -m venv .venv

# Next, enable it in each terminal session.
source .venv/bin/activate           # for linux
.venv/Scripts/activate              # for windows
```

## References

1. [Kaggle free datasets provider](https://www.kaggle.com/datasets)