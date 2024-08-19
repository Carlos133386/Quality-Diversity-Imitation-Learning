#!/usr/bin/env bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ENV_NAME="humanoid"
GRID_SIZE=50  # number of cells per archive dimension
SEED=1111
# SEED=2222

bonus_type='weighted_fitness_cond_measure_entropy'
# bonus_type='fitness_cond_measure_entropy'
# bonus_type='measure_entropy'
# bonus_type='measure_error'

# intrinsic_module='m_cond_reg_icm'
# intrinsic_module='m_reg_icm'
# intrinsic_module='m_cond_icm'
# intrinsic_module='icm'
# intrinsic_module='gail'
# intrinsic_module='m_cond_gail'
intrinsic_module='m_reg_gail'
# intrinsic_module='m_cond_reg_gail'
# intrinsic_module='vail'

GROUP_NAME="IL_ppga_"$ENV_NAME"_${intrinsic_module}"
RUN_NAME=$GROUP_NAME"_seed_"$SEED
num_demo=4
echo $RUN_NAME
python -m algorithm.train_il_ppga --env_name=$ENV_NAME \
                                --intrinsic_module=${intrinsic_module} \
                                --demo_dir='trajs_random_elite_with_measures/10episodes/' \
                                --reward_save_dir=reward_${num_demo}_random_elite/ \
                                --bonus_type=${bonus_type} \
                                --num_demo ${num_demo} \
                                --rollout_length=128 \
                                --use_wandb=False \
                                --wandb_group=$GROUP_NAME \
                                --num_dims=2 \
                                --seed=$SEED \
                                --anneal_lr=False \
                                --num_minibatches=8 \
                                --update_epochs=4 \
                                --normalize_obs=True \
                                --normalize_returns=True \
                                --clip_obs_rew=True \
                                --adaptive_stddev=False \
                                --clip_obs_rew=True \
                                --wandb_run_name=$RUN_NAME\
                                --popsize=300 \
                                --env_batch_size=3000 \
                                --learning_rate=0.0003 \
                                --vf_coef=2 \
                                --entropy_coef=0.0 \
                                --target_kl=0.008 \
                                --max_grad_norm=1 \
                                --total_iterations=2000 \
                                --dqd_algorithm=cma_maega \
                                --sigma0=0.5 \
                                --restart_rule=no_improvement \
                                --calc_gradient_iters=10 \
                                --move_mean_iters=10 \
                                --archive_lr=0.1 \
                                --threshold_min=200 \
                                --grid_size=$GRID_SIZE \
                                --expdir=./experiments_${num_demo}_random_elite_with_measures/$GROUP_NAME \
                                --wandb_project IL_PPGA_${ENV_NAME}
                                 
