#!/usr/bin/env bash

EXPECTED_ARGS=1
if [ $# -ne $EXPECTED_ARGS ]; then
    echo "Invalid number of arguments. Expected: $EXPECTED_ARGS"
    exit 1
fi

VALID_VERSIONS=("major" "minor" "patch")
if [[ ! " ${VALID_VERSIONS[@]} " =~ " $1 " ]]; then
    echo "Invalid version type. Expected one of: ${VALID_VERSIONS[*]}"
    exit 1
fi

git checkout main && git pull
poetry version $1
TAG_NAME=v$(poetry version -s)
git add pyproject.toml
git commit -m "Bump version to $TAG_NAME"
git push origin main

git tag $TAG_NAME && git push origin tag $TAG_NAME
