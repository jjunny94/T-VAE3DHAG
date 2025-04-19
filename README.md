## T-VAE3DHAG
Pytorch Implementation of '[Transformer VAE Based 3D Human Action Generation (2024)](https://eares.org/siteadmin/upload/1868EAP1224127.pdf)'


## Setup
Environment Setup & Data Preparation Following [here](https://github.com/EricGuo5513/action-to-motion)

## Usage
```shell script
# Training
python train_motion_vae.py --name <Experiment_name> --dataset_type humanact12 --batch_size 128 --motion_length 60 --coarse_grained --lambda_kld 0.001 --eval_every 2000 --plot_every 50 --print_every 20 --save_every 2000 --save_latest 50 --time_counter --gpu_id 0 --iters 50000

# Generating
python evaluate_motion_vae.py --name <Experiment_name> --dataset_type humanact12  --motion_length 60 --coarse_grained --gpu_id 0 --replic_times 5 --name_ext R0
```

## Acknowledgements
Code is built based on [Action2Motion](https://github.com/EricGuo5513/action-to-motion)
