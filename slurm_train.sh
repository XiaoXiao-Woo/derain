#!/usr/bin/env bash

set -x

PARTITION=defq
JOB_NAME=derain
#CONFIG=$3
#WORK_DIR=$4
GPUS=${GPUS:-16}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}
#NNODE=${NNODE:-'node[004]'}
#PY_ARGS=${@:5}

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:0 \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --nodelist=node[004-005] \
    ${SRUN_ARGS} \
    python -u run_derain.py --launcher="slurm" #${PY_ARGS}

