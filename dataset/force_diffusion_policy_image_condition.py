import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
# from visualize.visualize_depth_img import visualize_depth
import cv2
def rotate_and_translate_depth(
    depth,
    angle_deg,
    tx,
    ty,
    invalid_fill=0.0,
):
    """
    Rotate a depth image by angle_deg, then translate by (tx, ty).

    Args:
        depth: (H, W) numpy array
        angle_deg: rotation angle in degrees
        tx, ty: translation in pixels
        invalid_fill: fill value for empty area

    Returns:
        augmented depth image with same shape
    """
    H, W = depth.shape[:2]
    center = (W / 2.0, H / 2.0)

    # rotation matrix
    M_rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    rotated = cv2.warpAffine(
        depth,
        M_rot,
        (W, H),
        flags=cv2.INTER_NEAREST,   # important for depth
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=invalid_fill,
    )

    # translation matrix
    M_trans = np.array([
        [1, 0, tx],
        [0, 1, ty],
    ], dtype=np.float32)

    translated = cv2.warpAffine(
        rotated,
        M_trans,
        (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=invalid_fill,
    )

    return translated


def augment_two_depths(
    depth1,
    depth2,
    rotation_range=(0.0, 255.0),
    translation_range_x=(-20, 20),
    translation_range_y=(-20, 20),
    invalid_fill=0.0,
):
    """
    Augment two depth images:
    - same random rotation for both
    - independent random local translation for each

    Args:
        depth1, depth2: (H, W) numpy arrays
        rotation_range: sampled uniformly, in degrees
        translation_range_x: sampled uniformly, in pixels
        translation_range_y: sampled uniformly, in pixels
        invalid_fill: fill value for blank area

    Returns:
        depth1_aug, depth2_aug, angle_deg, (tx1, ty1), (tx2, ty2)
    """
    angle_deg = np.random.uniform(rotation_range[0], rotation_range[1])

    tx1 = np.random.randint(translation_range_x[0], translation_range_x[1] + 1)
    ty1 = np.random.randint(translation_range_y[0], translation_range_y[1] + 1)

    tx2 = np.random.randint(translation_range_x[0], translation_range_x[1] + 1)
    ty2 = np.random.randint(translation_range_y[0], translation_range_y[1] + 1)

    depth1_aug = rotate_and_translate_depth(
        depth1, angle_deg, tx1, ty1, invalid_fill=invalid_fill
    )
    depth2_aug = rotate_and_translate_depth(
        depth2, angle_deg, tx2, ty2, invalid_fill=invalid_fill
    )

    return depth1_aug, depth2_aug, angle_deg, (tx1, ty1), (tx2, ty2)

class DepthActionDataset(Dataset):
    def __init__(self, hdf5_file, transform=None, normalize=True, scale_to_unit=True, hdf5_file_action_normalize="/home/ubuntu/automate/IsaacGymEnvs_Assembly/data_utils/preprocess/processed_dataset_forward_0429_force_feedback.h5"):
        """
        Args:
            hdf5_file (str): Path to processed_dataset.h5
            transform: Optional transform for depth images
            normalize: Whether to normalize delta positions
            scale_to_unit: If True, scale to [-1, 1], otherwise [0, 1]
        """
        self.hdf5_file_action_normalize = hdf5_file_action_normalize
        self.hdf5_file_action = hdf5_file
        self.transform = transform
        self.normalize = normalize
        self.scale_to_unit = scale_to_unit

        # open in read-only mode, but don't load into RAM
        self.h5_action_normalize = h5py.File(self.hdf5_file_action_normalize, "r")
        self.actions_normalize = self.h5_action_normalize["actions"]

        self.length = self.actions_normalize.shape[0]
        self.index = np.load("/home/ubuntu/automate/IsaacGymEnvs_Assembly/data/diffusion_policy_index_0501.npy", mmap_mode = "r").copy()
        all_forces = self.h5_action_normalize['force'][()]
        if self.normalize:
            # compute min/max for delta_pos only (first 3 dims)

            all_actions = self.actions_normalize[()]  # (N, K, 9)
            delta_pos = all_actions[..., 0:3]  # (N, K, 3)
            delta_rot = all_actions[..., 3:]
            self.pos_min = delta_pos.min(axis=(0, 1))
            self.pos_max = delta_pos.max(axis=(0, 1))
            self.rot_min=delta_rot.min(axis=(0,1))
            self.rot_max=delta_rot.max(axis=(0,1))
            self.force_min = all_forces.min(axis=(0))
            self.force_max = all_forces.max(axis=(0))
        else:
            self.pos_min = None
            self.pos_max = None
            self.rot_min = None
            self.rot_max = None
            self.force_min = None
            self.force_max = None
        self.h5_action = h5py.File(self.hdf5_file_action, "r")
        self.actions = self.h5_action["actions"]
        self.proprio = self.h5_action["proprioception"]
        self.forces = self.h5_action["force"]
        print("Finish Loading Dataset")
        

    def __len__(self):
        return len(self.index) 

    def __getitem__(self, idx):
        task_id, hdf_id, depth_id = self.index[idx]
        data = np.load(f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/data/diffusion_policy_force_image_condition/asset_{task_id:05d}/hdf_{hdf_id}/depth_{depth_id}.npz",allow_pickle=False)
        depth = data["depth"].copy()
        depth = depth.astype(np.float32)
        depth = np.expand_dims(depth, axis=0)
        depth = torch.from_numpy(depth)
        depth = torch.clamp(depth, min=-0.5, max=0.0)
        depth = (depth - (-0.1890)) / 0.0795
        # proprio (9,)
        proprio = self.proprio[idx].astype(np.float32)
        proprio = torch.from_numpy(proprio)[...,2:]

        # actions (K, 9)
        actions = self.actions[idx].astype(np.float32)
        forces = self.forces[idx].astype(np.float32)
        if self.normalize:
            delta_pos = actions[..., 0:3]
            delta_rot6d = actions[..., 3:]
            forces = (forces - self.force_min) / (self.force_max - self.force_min + 1e-8)
            # min-max normalization
            denom = (self.pos_max - self.pos_min) + 1e-8
            norm_delta = (delta_pos - self.pos_min) / denom
            denom_rot=(self.rot_max-self.rot_min)+1e-8
            norm_delta_rot=(delta_rot6d-self.rot_min)/denom_rot
            if self.scale_to_unit:
                # map to [-1, 1]
                norm_delta = norm_delta * 2.0 - 1.0
                norm_delta_rot=norm_delta_rot*2.0-1.0
                forces = forces * 2.0 - 1.0
            actions = np.concatenate([norm_delta, norm_delta_rot], axis=-1)

        actions = torch.from_numpy(actions)
        socket_depth = data["socket_depth"].copy()
        init_plug_photo_depth = data["init_plug_photo_depth"].copy()
        forces = torch.from_numpy(forces)
        init_plug_photo_depth_aug, socket_depth_aug, angle_deg, trans1, trans2 = augment_two_depths(
            init_plug_photo_depth,
            socket_depth,
            rotation_range=(-180, 180),
            translation_range_x=(-20, 20),
            translation_range_y=(-20, 20),
            invalid_fill=0.0,
        )
        socket_depth = torch.from_numpy(socket_depth_aug)
        init_plug_photo_depth = torch.from_numpy(init_plug_photo_depth_aug)
        # Image Zero Experiment
        socket_depth[...]=0.0
        init_plug_photo_depth[...]=0.0
        return {
            "depth": depth,
            "proprioception": proprio,
            "actions": actions,
            "socket_depth": socket_depth,
            "init_plug_photo_depth": init_plug_photo_depth,
            "force": forces
        }

def main():
    hdf5_file = "/home/ubuntu/automate/processed_dataset_forward_0429_force_feedback.h5"

    print("Creating dataset...")
    dataset = DepthActionDataset(
        hdf5_file=hdf5_file,
        normalize=True,
        scale_to_unit=True
    )

    print("Dataset length:", len(dataset))

    print("\nNormalization stats:")
    print("pos_min:", dataset.pos_min)
    print("pos_max:", dataset.pos_max)
    print("rot_min:", dataset.rot_min)
    print("rot_max:", dataset.rot_max)
    print("force_min:", dataset.force_min)
    print("force_max:", dataset.force_max)

    print("\nTesting one sample...")
    sample = dataset[13100]

    print("Depth shape:", sample["depth"].shape)
    print("Proprioception shape:", sample["proprioception"].shape)
    print("Actions shape:", sample["actions"].shape)
    print("Socket depth shape:", sample["socket_depth"].shape)
    print("Init plug photo depth shape:", sample["init_plug_photo_depth"].shape)
    print("Force shape:", sample["force"].shape)
if __name__ == "__main__":
    main()