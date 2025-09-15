import json
def load_json(path):
    with open(path,"r") as f:
        return json.load(f)
def show_keys(d):
    """Print the keys if it's a dict, or length if it's a list."""
    if isinstance(d, dict):
        print("Top-level keys:", list(d.keys()))
    elif isinstance(d, list):
        print("This is a list with length:", len(d))
        if len(d) > 0:
            print("First element type:", type(d[0]))
    else:
        print("Type:", type(d))

data=load_json("/home/yinong/git/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/asset_00346_disassembly_traj.json")
# Data Info: 
# dict_keys(['fingertip_centered_pos', 'fingertip_centered_quat', 'arm_dof_pos', 'plug_grasp_pos', 'plug_grasp_quat', 'init_plug_pos', 'init_plug_quat', 'plug_pos', 'plug_quat'])
import pdb;pdb.set_trace()