# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.
# Adapted by Quan Liu.

"""
This script converts nuScenes data to KITTI format and KITTI results to nuScenes.
It is used for compatibility with software that uses KITTI-style annotations.

We do not encourage this, as:
- KITTI has only front-facing cameras, whereas nuScenes has a 360 degree horizontal fov.
- KITTI has no radar data.
- The nuScenes database format is more modular.
- KITTI fields like occluded and truncated cannot be exactly reproduced from nuScenes data.
- KITTI has different categories.

Limitations:
- We don't specify the KITTI imu_to_velo_kitti projection in this code base.
- We map nuScenes categories to nuScenes detection categories, rather than KITTI categories.
- Attributes are not part of KITTI and therefore set to '' in the nuScenes result format.
- Velocities are not part of KITTI and therefore set to 0 in the nuScenes result format.
- This script uses the `train` and `val` splits of nuScenes, whereas standard KITTI has `training` and `testing` splits.

This script includes one main function:
- nuscenes_construct_kitti_PCR_data(): Converts nuScenes LiDAR data and pose annotation to KITTI format.

To launch these scripts run:
- python export_kitti_minimal.py

To work with the original KITTI dataset, use these parameters:
 --nusc_kitti_dir /data/sets/kitti --split training

See https://www.nuscenes.org/object-detection for more information on the nuScenes result format.
"""
import json
import os
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import create_splits_logs


class KittiConverter:
    def __init__(self,
                 nusc_dir: str = '/mnt/disk/NUSCENES',
                 nusc_kitti_dir: str = '/mnt/disk/NUSCENES/nusc_kitti',
                 lidar_name: str = 'LIDAR_TOP',
                 nusc_version: str = 'v1.0-trainval',
                 split: str = 'val'):
        """
        :param nusc_dir: Root of nuScenes dataset.
        :param nusc_kitti_dir: Where to write the KITTI-style annotations.
        :param lidar_name: Name of the lidar sensor.
        :param image_count: Number of images to convert.
        :param nusc_version: nuScenes version to use.
        :param split: Dataset split to use.
        """
        self.nusc_kitti_dir = os.path.expanduser(nusc_kitti_dir)
        self.lidar_name = lidar_name
        self.nusc_version = nusc_version
        self.split = split

        # Create nusc_kitti_dir.
        if not os.path.isdir(self.nusc_kitti_dir):
            os.makedirs(self.nusc_kitti_dir)

        # Select subset of the data to look at.
        self.nusc = NuScenes(version=nusc_version, dataroot=nusc_dir, verbose=True)

    def nuscenes_construct_kitti_PCR_data(self) -> None:
        """
        Converts nuScenes Lidar sequences and poses into KITTI form
        """
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse

        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(self.split, self.nusc)
        # print(split_logs[0])

        # Create output folder.
        base_folder = os.path.join(self.nusc_kitti_dir, self.split, 'sequences')
        # indice_folder = os.path.join(self.nusc_kitti_dir, self.split, 'indice')
        for folder in [base_folder]:
            if not os.path.isdir(folder):
                os.makedirs(folder)

        for log in split_logs:
            # Use only the samples from the current split.
            sample_tokens = self._split_to_samples(log)

            token_idx = 0  # Start tokens from 0.
            trans = []

            log_folder = os.path.join(base_folder, log, 'velodyne')
            if not os.path.isdir(log_folder):
                os.makedirs(log_folder)

            for sample_token in sample_tokens:
                print(f"Processing {log}, {token_idx}")
                # Get sample data.
                sample = self.nusc.get('sample', sample_token)
                lidar_token = sample['data'][self.lidar_name]

                # Retrieve sensor records.
                sd_record_lid = self.nusc.get('sample_data', lidar_token)
                cs_record_lid = self.nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])

                # Get ego pose. Note that ego pose is the position of imu, not that of Lidar, thus it needs correcting.
                pos = self.nusc.get('ego_pose', sd_record_lid['ego_pose_token'])
                ego_to_world = transform_matrix(pos['translation'], Quaternion(pos['rotation']),
                                                inverse=False)
                lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']),
                                              inverse=False)
                lid_to_world = np.dot(ego_to_world, lid_to_ego)
                lid_to_world_kitti = np.dot(lid_to_world, kitti_to_nu_lidar.transformation_matrix)
                trans.append(lid_to_world_kitti)

                # Retrieve the token from the lidar.
                # Note that this may be confusing as the filename of the camera will include the timestamp of the lidar,
                # not the camera.
                filename_lid_full = sd_record_lid['filename']
                token = '%06d' % token_idx # We use KITTI names instead of nuScenes names
                token_idx += 1

                # Convert lidar.
                # Note that we are only using a single sweep, instead of the commonly used n sweeps.
                src_lid_path = os.path.join(self.nusc.dataroot, filename_lid_full)
                dst_lid_path = os.path.join(log_folder, token + '.bin')
                assert not dst_lid_path.endswith('.pcd.bin')
                pcl = LidarPointCloud.from_file(src_lid_path)
                pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)  # In KITTI lidar frame.
                with open(dst_lid_path, "w") as lid_file:
                    pcl.points.T.tofile(lid_file)

            # Save poses of a single log sequence into one file
            trans = np.array(trans)
            pose_path = os.path.join(base_folder, log, 'poses')
            np.save(pose_path, trans)


if __name__ == '__main__':
    for convert_split in ['train', 'val', 'test']:
        converter = KittiConverter(split=convert_split)
        converter.nuscenes_construct_kitti_PCR_data()
