#!/usr/bin/env bash

ENV_NAME="halfcheetah"
GRID_SIZE=50  # number of cells per archive dimension
SEED=1111


RUN_NAME="paper_ppga_"$ENV_NAME"_seed_"$SEED
echo $RUN_NAME
GROUP_NAME=IL_ppga_"$ENV_NAME"_expert

cp_dir=./experiments_experts/$GROUP_NAME/${SEED}/checkpoints
cp_iter=00001090
scheduler_cp=${cp_dir}/cp_${cp_iter}/scheduler_${cp_iter}.pkl
archive_cp=${cp_dir}/cp_${cp_iter}/archive_df_${cp_iter}.pkl

python -m algorithm.train_ppga --env_name=$ENV_NAME \
                                --rollout_length=128 \
                                --use_wandb=False \
                                --wandb_group=paper \
                                --num_dims=2 \
                                --seed=$SEED \
                                --anneal_lr=False \
                                --num_minibatches=8 \
                                --update_epochs=4 \
                                --normalize_obs=True \
                                --normalize_returns=True \
                                --wandb_run_name=$RUN_NAME\
                                --popsize=300 \
                                --env_batch_size=3000 \
                                --learning_rate=0.001 \
                                --vf_coef=2 \
                                --entropy_coef=0.0 \
                                --target_kl=0.008 \
                                --max_grad_norm=1 \
                                --total_iterations=2000 \
                                --dqd_algorithm=cma_maega \
                                --sigma0=1.0 \
                                --restart_rule=no_improvement \
                                --calc_gradient_iters=10 \
                                --move_mean_iters=10 \
                                --archive_lr=0.5 \
                                --threshold_min=-200 \
                                --grid_size=$GRID_SIZE \
                                --expdir=./experiments_experts/${GROUP_NAME} \
                                --wandb_project PPGA_${ENV_NAME} \
                                --load_scheduler_from_cp=${scheduler_cp} \
                                --load_archive_from_cp=${archive_cp} \
                                # no elite in archive when threshold_min=200
