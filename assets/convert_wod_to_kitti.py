
import os
import warnings

# Suppress annoying warnings from PyArrow and TensorFlow
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

import dask.dataframe as dd
from waymo_open_dataset import v2
import glob
import fire
import numpy as np
from tqdm import tqdm

from waymo_open_dataset.v2.perception.utils.lidar_utils import convert_range_image_to_point_cloud


class KittiConverter:
    def __init__(self,
                 root: str = '/mnt/disk/waymo_open_dataset_V2',
                 waymo_kitti_dir: str = '/mnt/disk/waymo_open_dataset_V2/waymo_kitti_firstReturn',
                 lidar_name: int = 1,
                 lidar_return: int = 0):
        """
        :param root: The root directory of waymo open dataset V2.
        :param waymo_kitti_dir: Where to write the KITTI-style annotations.
        :param lidar_name: Name of the lidar sensor.
        :param lidar_return: The first or the second return of LiDAR. Must be 0 or 1.
        :param split: Dataset split to use.
        """
        assert lidar_return in [0,1]

        self.base_root = root
        self.splits = ['training', 'validation', 'testing']
        self.waymo_kitti_dir = os.path.expanduser(waymo_kitti_dir)
        self.lidar_name = lidar_name
        self.lidar_return = lidar_return

        # Create waymo_kitti_dir.
        if not os.path.isdir(self.waymo_kitti_dir):
            os.makedirs(self.waymo_kitti_dir)

    def waymo_construct_kitti_PCR_data(self) -> None:
        """
        Converts waymo Lidar sequences and poses into KITTI form
        """

        for split in self.splits:
            self.split = split
            self.root = os.path.join(self.base_root, self.split)
            # Get assignment of scenes to splits.
            subset_names = os.listdir(os.path.join(self.root, 'vehicle_pose'))

            # Create output folder.
            base_folder = os.path.join(self.waymo_kitti_dir, self.split, 'sequences')
            for folder in [base_folder]:
                if not os.path.isdir(folder):
                    os.makedirs(folder)

            # for log in tqdm(iter(subset_names)):
            for log in iter(subset_names):
                log_name = log[:-8]

                # skip the processed logs and the metadata directory
                if os.path.exists(os.path.join(base_folder, log_name, 'velodyne', 'poses.npy')) or log == '_metadata':
                    print(f"Skipping {log=}")
                    continue

                print(f"Processing {log=}", flush=True)

                token_idx = 0  # Start tokens from 0.
                trans = []

                log_folder = os.path.join(base_folder, log_name, 'velodyne')
                if not os.path.isdir(log_folder):
                    os.makedirs(log_folder)

                vehicle_pose_df = self.read(os.path.join(self.root, 'vehicle_pose', log))
                lidar_df = self.read(os.path.join(self.root, 'lidar', log))
                lidar_df = lidar_df.where(lidar_df['key.laser_name']==self.lidar_name).dropna(how='any')
                lidar_w_pose_df = v2.merge(lidar_df, vehicle_pose_df)
                lidar_w_pose_iterator = iter(lidar_w_pose_df.iterrows())

                lidar_calib_df = self.read(os.path.join(self.root, 'lidar_calibration', log))
                lidar_calib_df = lidar_calib_df.where(lidar_calib_df['key.laser_name']==self.lidar_name).dropna(how='any')
                calib_component = v2.LiDARCalibrationComponent.from_dict(lidar_calib_df.compute().iloc[0])

                # for _ in tqdm(range(len(vehicle_pose_df))):
                for _ in range(len(vehicle_pose_df)):
                    # print(f"Processing {log}, {token_idx}")
                    _, row = next(lidar_w_pose_iterator)

                    # Get lidar data.
                    # note that wodV2 provides built-in extrinsic correction with rot and trans in 'convert_range_image_to_point_cloud'.
                    # This means that lidar frame origins are located at the imu, instead of the lidar center.
                    lidar_com = v2.LiDARComponent.from_dict(row)
                    range_component = lidar_com.range_image_returns[self.lidar_return]
                    tf_pointcloud = convert_range_image_to_point_cloud(range_component, calib_component)
                    pointcloud = tf_pointcloud.numpy()

                    # Get ego pose. Note that ego pose is the position of imu.
                    pose  = v2.VehiclePoseComponent.from_dict(row).world_from_vehicle.transform
                    pose = np.array(pose).tolist()
                    trans.append(pose)

                    # rename the individual frames with tokens
                    token = '%06d' % token_idx # We use KITTI names for consistency.
                    token_idx += 1

                    # Store lidar.
                    dst_lid_path = os.path.join(log_folder, token + '.bin')
                    with open(dst_lid_path, "w") as lid_file:
                        pointcloud.tofile(lid_file)

                # Save poses of a single log sequence into one file
                trans = np.array(trans)
                pose_path = os.path.join(base_folder, log_folder, 'poses')
                np.save(pose_path, trans)

    def read(self, path: str) -> dd.DataFrame:
        """Creates a Dask DataFrame for the component specified by its tag."""
        paths = glob.glob(os.path.join(path[:-7]+"*"))
        return dd.read_parquet(paths)


if __name__ == '__main__':
    fire.Fire(KittiConverter)
