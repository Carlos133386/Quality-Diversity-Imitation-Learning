#!/usr/bin/env bash


export XLA_PYTHON_CLIENT_PREALLOCATE=false

ENV_NAME="halfcheetah"
GRID_SIZE=50  # number of cells per archive dimension
SEED=1111
SEED=2222
SEED=3333


# bonus_type='weighted_fitness_cond_measure_entropy'
# bonus_type='fitness_cond_measure_entropy'
bonus_type='measure_entropy'
bonus_type='single_step_bonus'
bonus_type='None'
# bonus_type='measure_error'

# intrinsic_module='m_cond_reg_icm'
# intrinsic_module='m_reg_icm'
# intrinsic_module='m_cond_icm'
# intrinsic_module='icm'

# intrinsic_module='zero'
# intrinsic_module='gail'

# intrinsic_module='m_acgail'
# intrinsic_module='m_cond_acgail'

auxiliary_loss_fn='MSE'
# auxiliary_loss_fn='NLL'

intrinsic_module='m_cond_gail'
#intrinsic_module='m_reg_gail'
#intrinsic_module='m_cond_reg_gail'
#intrinsic_module='abgail'
#intrinsic_module='m_cond_vail'
#intrinsic_module='vail'
#intrinsic_module='pwil'


GROUP_NAME=IL_ppga_"$ENV_NAME"_${intrinsic_module} #_RegLoss_${auxiliary_loss_fn}_Bonus_${bonus_type}
RUN_NAME=$GROUP_NAME"_seed_"$SEED
num_elite=4
num_demo_per_elite=50
num_demo=`expr ${num_elite} \* ${num_demo_per_elite}`
num_demo=4
demo_str=${num_elite}x${num_demo_per_elite}


gail_batchsize=200
echo $RUN_NAME
data_str=good_and_diverse_elite_with_measures_top500

# cp_dir=./experiments_${num_demo}_${data_str}/$GROUP_NAME/${SEED}/checkpoints
# cp_iter=00000740
# scheduler_cp=${cp_dir}/cp_${cp_iter}/scheduler_${cp_iter}.pkl
# archive_cp=${cp_dir}/cp_${cp_iter}/archive_df_${cp_iter}.pkl
archive_bonus=True
if [ "$archive_bonus" = "True" ]; then
    GROUP_NAME="${GROUP_NAME}_archive_bonus"
fi

if [ "$archive_bonus" = "False" ] && [ "$bonus_type" = "None" ]; then
    GROUP_NAME="${GROUP_NAME}"
fi

wo_a=False
if [ "$wo_a" = "True" ]; then
    GROUP_NAME="${GROUP_NAME}_wo_a"
fi

bonus_smooth=False
if [ "$bonus_smooth" = "False" ] && [ "$archive_bonus" = "True" ]; then
    GROUP_NAME="${GROUP_NAME}_wo_smooth"
fi
python -m algorithm.train_il_ppga --env_name=$ENV_NAME \
                                --wo_a=${wo_a} \
                                --bonus_smooth=${bonus_smooth} \
                                --archive_bonus=${archive_bonus} \
                                --intrinsic_module=${intrinsic_module} \
                                --demo_dir=trajs_${data_str}/${num_demo}episodes/ \
                                --reward_save_dir=reward_converged_${num_demo}_${data_str}/ \
                                --auxiliary_loss_fn=${auxiliary_loss_fn} \
                                --bonus_type=${bonus_type} \
                                --num_demo ${num_demo} \
                                --gail_batchsize=${gail_batchsize} \
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
                                --threshold_min=-500 \
                                --grid_size=$GRID_SIZE \
                                --expdir=./experiments_${num_demo}_${data_str}/$GROUP_NAME \
                                --wandb_project IL_PPGA_${ENV_NAME} #\
