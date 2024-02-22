#!/bin/bash

source ~/.bashrc
micromamba activate diss
micromamba env export --from-history > environment.yml