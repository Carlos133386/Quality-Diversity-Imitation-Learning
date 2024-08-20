
# intrinsic_module='icm'

# env_name='ant'
# env_name='walker2d'
env_name='humanoid'

# num_demo=8
num_demo=4

# intrinsic_module='icm'
# intrinsic_module='m_cond_icm'
# intrinsic_module='m_reg_icm'
# intrinsic_module='m_cond_reg_icm'

intrinsic_module='giril'

data_str=good_and_diverse_elite_with_measures_top500

python -m algorithm.learn_url_reward --env_name ${env_name} \
		--intrinsic_module ${intrinsic_module} --num_minibatches 32 \
		--demo_dir trajs_${data_str}/${num_demo}episodes/ --num_demo ${num_demo} \
		--reward_save_dir reward_${num_demo}_${data_str}/ \
			--intrinsic_save_interval 100 --intrinsic_epoch 1000 


