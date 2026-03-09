source ~/automate/miniconda3/bin/activate
conda activate rlgpu

# Go into isaacgymenvs
export PYTHONPATH=$(pwd):$PYTHONPATH
cd isaacgymenvs
python iga.py \
    task=AutoMateTaskDisassemble \
    task.env.overwrite_subassemblies=True \
    task.env.desired_subassemblies=['asset_00030'] \
    headless=True \
    seed=29
