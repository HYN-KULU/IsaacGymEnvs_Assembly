from preprocess_data import load_processed_dataset
import h5py 
import numpy as np
data1=load_processed_dataset("/home/ubuntu/automate/IsaacGymEnvs_Assembly/processed_dataset_depth_relative_scale_1010.h5")
data2=load_processed_dataset("/home/ubuntu/automate/IsaacGymEnvs_Assembly/processed_dataset_depth_relative_1014_DAgger.h5")
actions=np.concatenate([data1["actions"], data2["actions"]])
proprioception=np.concatenate([data1["proprioception"], data2["proprioception"]])
out_filename = "processed_dataset_depth_relative_1015_scale_dagger.h5"
with h5py.File(out_filename, "w") as f:
    # f.create_dataset("depth", data=depth_array, chunks=(1, 480, 640), compression="gzip", compression_opts=4)
    f.create_dataset("proprioception", data=proprioception, compression="gzip", compression_opts=4)
    f.create_dataset("actions", data=actions, compression="gzip", compression_opts=4)