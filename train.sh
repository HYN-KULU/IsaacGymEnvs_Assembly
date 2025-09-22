source ~/automate/miniconda3/bin/activate
conda activate rlgpu

# Go into isaacgymenvs
cd isaacgymenvs
python train.py task=AutoMateTaskAssemble task.env.overwrite_subassemblies=True task.env.desired_subassemblies=['asset_00346'] headless=True
