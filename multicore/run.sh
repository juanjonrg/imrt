#!/bin/bash

set -o errexit
set -o xtrace

slurm_command=""
plan=5

$slurm_command ./gradient_mkl $plan 
