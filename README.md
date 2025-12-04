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

#### Set up the data control version repository
```sh
# Install and init the data control version
pip install dvc                 # Install data version control
pip install dvc-s3              # Install data version control for aws s3


dvc init                                                    # Init the data version control track
dvc remote add -d storage s3://labsdatasets/housing-prices  # Set up the S3 bucket (require authentication with aws cli)
```

#### Add a data version and push it.

```sh
# Add and push a new data version
dvc add datasets
dvc push

# Commit and push in Github
git add .
git commit -m "Version #"
git push
```

#### Download the data from repository
```sh
dvc pull
```

#### Tag in Github
```sh
git checkout main
git tag -a v1 -m "Version #"
git push origin --tags
```


## References

1. [Kaggle free datasets provider](https://www.kaggle.com/datasets)
