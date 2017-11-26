#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --job-name=tdr_dag
#SBATCH --output=tdr_dag.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load modules
#module restore

# Run the job
EXP_NAME = "rnn.scheduled_sampling"
srun python -m train -data data/stackexchange/stackexchange.train_valid.pt -vocab data/stackexchange/stackexchange.vocab.pt -exp_path "exp/rnn.scheduled_sampling/%s.uni-directional.%s" -save_path "model/rnn.scheduled_sampling/%s.uni-directional.%s" -exp "stackexchange" -batch_size 512 -run_valid_every 1000 -scheduled_sampling -scheduled_sampling_batches 30000
