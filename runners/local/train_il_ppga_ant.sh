#!/usr/bin/env bash

ENV_NAME="ant"
GRID_SIZE=10  # number of cells per archive dimension
SEED=1111
# SEED=2222

# bonus_type='weighted_fitness_cond_measure_entropy'
# bonus_type='fitness_cond_measure_entropy'
# bonus_type='measure_entropy'
bonus_type='measure_error'

# intrinsic_module='m_cond_reg_icm'
# intrinsic_module='m_reg_icm'
# intrinsic_module='m_cond_icm'
# intrinsic_module='icm'
# intrinsic_module='gail'
# intrinsic_module='m_cond_gail'
# intrinsic_module='m_reg_gail'
intrinsic_module='m_cond_reg_gail'
# intrinsic_module='vail'
# intrinsic_module='giril'

GROUP_NAME="IL_ppga_"$ENV_NAME"_${intrinsic_module}"
RUN_NAME=$GROUP_NAME"_seed_"$SEED
num_demo=8

echo $RUN_NAME
data_str=good_and_diverse_elite_with_measures_top500
python -m algorithm.train_il_ppga --env_name=$ENV_NAME \
                                     --intrinsic_module=${intrinsic_module} \
                                     --demo_dir=trajs_${data_str}/${num_demo}episodes/ \
                                     --reward_save_dir=reward_${num_demo}_${data_str}/ \
                                     --bonus_type=${bonus_type} \
                                     --num_demo=${num_demo} \
                                     --rollout_length=128 \
                                     --use_wandb=False \
                                     --seed=$SEED \
                                     --wandb_group=$GROUP_NAME \
                                     --num_dims=4 \
                                     --num_minibatches=8 \
                                     --update_epochs=4 \
                                     --normalize_obs=True \
                                     --normalize_returns=True \
                                     --wandb_run_name=$RUN_NAME \
                                     --popsize=300 \
                                     --env_batch_size=3000 \
                                     --learning_rate=0.001 \
                                     --vf_coef=2 \
                                     --max_grad_norm=1 \
                                     --torch_deterministic=False \
                                     --total_iterations=2000 \
                                     --dqd_algorithm=cma_maega \
                                     --calc_gradient_iters=10 \
                                     --move_mean_iters=10 \
                                     --archive_lr=0.1 \
                                     --restart_rule=no_improvement \
                                     --sigma0=3.0 \
                                     --threshold_min=-500 \
                                     --grid_size=$GRID_SIZE \
                                     --expdir=./experiments_${num_demo}_${data_str}/$GROUP_NAME \
                                     --wandb_project IL_PPGA_${ENV_NAME}
