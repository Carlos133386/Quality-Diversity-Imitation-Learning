
# intrinsic_module='icm'

# env_name='ant'
env_name='walker2d'
# env_name='humanoid'
# env_name='halfcheetah'
# env_name='hopper'

num_demo=4

# bonus_type='weighted_fitness_cond_measure_entropy'
# bonus_type='fitness_cond_measure_entropy'
# bonus_type='measure_entropy'
bonus_type='measure_error'


auxiliary_loss_fn='MSE'
# auxiliary_loss_fn='NLL'

# intrinsic_module='icm'
# intrinsic_module='m_cond_icm'
# intrinsic_module='m_reg_icm'
# intrinsic_module='m_cond_reg_icm'

# intrinsic_module='giril'
# intrinsic_module='m_cond_giril'
# intrinsic_module='m_reg_giril'
intrinsic_module='m_cond_reg_giril'

data_str=good_and_diverse_elite_with_measures_top500

python -m algorithm.learn_url_reward --env_name ${env_name} \
		--intrinsic_module ${intrinsic_module} --num_minibatches 32 \
		--demo_dir trajs_${data_str}/${num_demo}episodes/ --num_demo ${num_demo} \
		--reward_save_dir reward_${num_demo}_${data_str}/ \
		--intrinsic_save_interval 100 --intrinsic_epoch 1000 \
		--auxiliary_loss_fn ${auxiliary_loss_fn} --bonus_type ${bonus_type}


