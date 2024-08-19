
# intrinsic_module='icm'

env_name='ant'
# env_name='walker2d'
# env_name='humanoid'

num_demo=4
# intrinsic_module='icm'
# intrinsic_module='m_cond_icm'
# intrinsic_module='m_reg_icm'
intrinsic_module='m_cond_reg_icm'

	python -m algorithm.learn_url_reward --env_name ${env_name} \
		--intrinsic_module ${intrinsic_module} --num_minibatches 32 \
    --demo_dir 'trajs_random_elite_with_measures/10episodes/' --num_demo ${num_demo} \
    --reward_save_dir reward_${num_demo}_random_elite/ \
		--intrinsic_save_interval 100 --intrinsic_epoch 1000 


