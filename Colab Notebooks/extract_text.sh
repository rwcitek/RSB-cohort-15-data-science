#!/bin/bash

for notebook in *.ipynb; do
    jq -r '.cells[] | select(.cell_type=="markdown") | .source[]' "$notebook" > "${notebook%.ipynb}.txt"
done