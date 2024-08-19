# Quality Diversity Imitation Learning based on Proximal Policy Gradient Arborescence 


## Implemented IL algorithms
- Implement imitation learning methods based on PPGA.

    - GAIL

    - measure conditioned GAIL

    - measure regularized GAIL
        - measure_error bonus
        - measure_entropy bonus

    - VAIL

    - ICM

    - measure conditioned ICM

    - measure regularized ICM
        - measure_error bonus
        - measure_entropy bonus 

- Check ```RL/intrinsic_ppo.py```, ```algorithm/learn_url_reward.py```, and ```algorithm/train_il_ppga.py```  for implementations.

## Requirements
Follow README.md to intall the environment. 
```bash
conda env create -f environment.yml
conda activate ppga  
```
Then install this project's custom version of pyribs.
```bash
cd pyribs && pip install -e. && cd ..
```

## Running Experiments
We provide run scripts to train IL algorithms. 

### Trainning PPGA as expert demonstrators.
```bash
# from PPGA root. Ex. to run humanoid
./runners/local/train_ppga_humanoid.sh 
./runners/local/train_ppga_ant.sh 
./runners/local/train_ppga_walker2d.sh 
```

### Generate demonstrations using the Pretrained Archieves


#### Pretrained Archives 

The pretrained archives is stored in the ```experiments``` folder.

#### Generate demonstrations

Use ```gen_traj.py``` to generate demonstrations from best elite or random elites with state-dependent measures.
The generated demonstrations are stored in the ```demo_dir```(10 episodes and 100 episodes).


### Run IL experiments
#### For GAILs and VAILs, run the following commands to train the IL models.
```bash
# from PPGA root. Ex. to run humanoid
./runners/local/train_il_ppga_humanoid.sh 
./runners/local/train_il_ppga_ant.sh 
./runners/local/train_il_ppga_walker2d.sh 
```

#### For other methods,

Run ```./runners/local/train_url_reward.sh``` first to pretrain the reward models;
then run 
```bash
# from PPGA root. Ex. to run humanoid
./runners/local/train_il_ppga_humanoid.sh 
./runners/local/train_il_ppga_ant.sh 
./runners/local/train_il_ppga_walker2d.sh 
```
to train IL models.

The pretrained reward models are stored in the ```reward_save_dir``` folder. 
The experiments results are stored in the ```expdir``` folder.

### Visualization
Run ```plot.py```
Tables and Figures are stored in the ```expdir``` folder.


## Done [13 Aug 2024]

- Generate demonstrations with state-dependent measures.

- Build measure-conditioned and measure-regularized imitation learning models for QDIL. (m_cond_gail, m_reg_gail, m_cond_icm, m_reg_icm)
