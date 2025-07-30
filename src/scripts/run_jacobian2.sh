#!/bin/bash

# to run
# nohup bash run_jacobian2.sh > jacobian_outs2.out 2> jacobian_errors2.err &

# Set paths
HOME_LOC=/home/a03-sgoel/FLaT 
SCRIPT_LOC=$HOME_LOC/src/sampling
LOG_LOC=$HOME_LOC/src/scripts


CONDA_ENV=shrey_flat
#PYTHON_EXECUTABLE=$(source ~/miniconda3/bin/activate $CONDA_ENV && which python)
PYTHON_EXECUTABLE=$(conda run -n $CONDA_ENV which python)  


mkdir -p $LOG_LOC

# Activate Conda environment
echo "Activating conda environment: $CONDA_ENV"
source ~/miniconda3/bin/activate $CONDA_ENV

# Run the script using multiple GPUs and redirect output/error logs
CUDA_VISIBLE_DEVICES=1 nohup $PYTHON_EXECUTABLE $SCRIPT_LOC/jacobian2.py > "$LOG_LOC/jacobian_outs2.out" 2> "$LOG_LOC/jacobian_errors2.err" &

echo "Script started at $(date)"
conda deactivate
