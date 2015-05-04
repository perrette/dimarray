#!/bin/bash

# get notebook nbformat
python -c "import json; nb=json.load(open('$1')); print nb['nbformat']"
