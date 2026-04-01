#!/bin/bash

github_repository=$1
repository_branch=$2

if [[ -n "$github_repository" ]]; then
  git clone $github_repository
  cd sagemaker-python-sdk
  if [[ -n "$repository_branch" ]]; then
    git checkout $repository_branch
  fi
  pip install .
else
  pip install "sagemaker>=2.0,<3.0"
fi
