conda activate rlgpu
cd isaacgymenvs
# python train.py task=AutoMateTaskAssemble task.env.overwrite_subassemblies=True task.env.desired_subassemblies=['asset_00346'] headless=True
python train.py task=AutoMateTaskDisassemble task.env.overwrite_subassemblies=True task.env.desired_subassemblies=['asset_00346'] headless=True
cd ..