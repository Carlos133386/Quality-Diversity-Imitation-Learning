#!/usr/bin/env bash

ENV_NAME="halfcheetah"
GRID_SIZE=50  # number of cells per archive dimension
SEED=1111


RUN_NAME="paper_ppo_"$ENV_NAME"_seed_"$SEED
echo $RUN_NAME
python -m RL.train_ppo --env_name=$ENV_NAME \
                                --rollout_length=128 \
                                --use_wandb=True \
                                --wandb_group='PPO' \
                                --num_dims=2 \
                                --seed=$SEED \
                                --anneal_lr=False \
                                --num_minibatches=8 \
                                --update_epochs=4 \
                                --normalize_obs=True \
                                --normalize_returns=True \
                                --wandb_run_name=$RUN_NAME \
                                --env_batch_size=3000 \
                                --learning_rate=0.001 \
                                --vf_coef=2 \
                                --entropy_coef=0.0 \
                                --target_kl=0.008 \
                                --max_grad_norm=1 \
                                --total_iterations=2000 \
                                --wandb_project PPGA_${ENV_NAME}
                                 
