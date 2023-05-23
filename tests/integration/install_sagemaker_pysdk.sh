#!/bin/bash

github_repository=$1
repository_branch=$2

if [[ -z "$github_repository" ]]; then
  git clone $github_repository
  cd sagemaker-python-sdk
  if [[ -z "$repository_branch" ]]; then
    git checkout $repository_branch
  fi
  pip install .
else
  pip install -U sagemaker
fi
