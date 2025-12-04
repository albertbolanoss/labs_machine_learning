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

### Commands

```sh
jupyter nbconvert --to python notebooks/[notebook].ipynb          # Export code from notebook to python script: 
pip freeze > requirements.txt					  # Define dependencies
py notebooks/hyderabad_house_price/app/model_endpoint.py     	  # Run Model endpoint
```

### Data version control

```sh
# Install and init the data control version
pip install "dvc[s3]"
dvc init


pip install dvc-s3
dvc add datasets
dvc remote add -d storage s3://labsdatasets/housing-prices
```

#### Create a tag version

```sh
dvc add datasets
dvc push
git add .
git commit -m "Version #"
git push
git tag -a v1 -m "Version 1"

```

```sh
dvc pull
```

dvc add datasets/

## References

1. [Kaggle free datasets provider](https://www.kaggle.com/datasets)
