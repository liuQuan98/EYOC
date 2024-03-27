# -*- coding: future_fstrings -*-
#
# Written by Chris Choy <chrischoy@ai.stanford.edu>
# Distributed under MIT License
import logging
import random
import torch
import torch.utils.data
import numpy as np
import glob
import os
from scipy.linalg import expm, norm
import pathlib
import copy

from util.pointcloud import get_matching_indices, make_open3d_point_cloud
import lib.transforms as t
from lib.timer import Timer

import MinkowskiEngine as ME
import open3d as o3d

import dask.dataframe as dd
from pytorch3d.ops.knn import knn_points
import pytorch3d

pose_cache = {}
kitti_icp_cache = {}


def collate_pair_fn(list_data):
    xyz0, xyz1, coords0, coords1, feats0, feats1, matching_inds, trans, frame_distances = list(
        zip(*list_data))
    # xyz_batch0, xyz_batch1 = [], []
    matching_inds_batch, trans_batch, len_batch = [], [], []

    batch_id = 0
    curr_start_inds = np.zeros((1, 2))

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        else:
            raise ValueError(f'Can not convert to torch tensor, {x}')

    for batch_id, _ in enumerate(coords0):
        N0 = coords0[batch_id].shape[0]
        N1 = coords1[batch_id].shape[0]
        if len(matching_inds[batch_id]) != 0:
            # xyz_batch0.append(to_tensor(xyz0[batch_id]))
            # xyz_batch1.append(to_tensor(xyz1[batch_id]))

            trans_batch.append(to_tensor(trans[batch_id]).float())

            matching_inds_batch.append(
                torch.from_numpy(
                    np.array(matching_inds[batch_id]) + curr_start_inds))
            len_batch.append([N0, N1])
        # Move the head
        curr_start_inds[0, 0] += N0
        curr_start_inds[0, 1] += N1

    coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)
    coords_batch1, feats_batch1 = ME.utils.sparse_collate(coords1, feats1)

    # Concatenate all lists
    # xyz_batch0 = torch.cat(xyz_batch0, 0).float()
    # xyz_batch1 = torch.cat(xyz_batch1, 0).float()
    # trans_batch = torch.cat(trans_batch, 0).float()
    matching_inds_batch = torch.cat(matching_inds_batch, 0).int()

    return {
        'pcd0': list(xyz0),
        'pcd1': list(xyz1),
        'sinput0_C': coords_batch0,
        'sinput0_F': feats_batch0.float(),
        'sinput1_C': coords_batch1,
        'sinput1_F': feats_batch1.float(),
        'correspondences': matching_inds_batch,
        'T_gt': trans_batch,
        'len_batch': len_batch,
        'frame_distance': frame_distances
    }


# Rotation matrix along axis with angle theta
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
    T = np.eye(4)
    R = M(
        randg.rand(3) - 0.5,
        rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T


class PairDataset(torch.utils.data.Dataset):
    AUGMENT = None

    def __init__(self,
                 phase,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 config=None):
        self.phase = phase
        self.files = []
        self.data_objects = []
        self.transform = transform
        self.voxel_size = config.voxel_size
        self.matching_search_voxel_size = \
            config.voxel_size * config.positive_pair_search_voxel_size_multiplier

        self.random_scale = random_scale
        self.min_scale = config.min_scale
        self.max_scale = config.max_scale
        self.random_rotation = random_rotation
        self.rotation_range = config.rotation_range
        self.randg = np.random.RandomState()
        if manual_seed:
            self.reset_seed()

    def reset_seed(self, seed=0):
        logging.info(f"Resetting the data loader seed to {seed}")
        self.randg.seed(seed)

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def __len__(self):
        return len(self.files)


class ThreeDMatchTestDataset(PairDataset):
    DATA_FILES = {'test': './config/test_3dmatch.txt'}

    def __init__(self,
                 phase,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 scene_id=None,
                 config=None,
                 return_ply_names=False):

        PairDataset.__init__(self, phase, transform, random_rotation,
                             random_scale, manual_seed, config)
        assert phase == 'test', "Supports only the test set."

        self.root = config.threed_match_dir

        subset_names = open(self.DATA_FILES[phase]).read().split()
        if scene_id is not None:
            subset_names = [subset_names[scene_id]]
        for sname in subset_names:
            traj_file = os.path.join(self.root, sname + '-evaluation/gt.log')
            assert os.path.exists(traj_file)
            # traj = read_trajectory(traj_file)
            raise NotImplementedError
            for ctraj in traj:
                i = ctraj.metadata[0]
                j = ctraj.metadata[1]
                T_gt = ctraj.pose
                self.files.append((sname, i, j, T_gt))

        self.return_ply_names = return_ply_names

    def __getitem__(self, pair_index):
        sname, i, j, T_gt = self.files[pair_index]
        ply_name0 = os.path.join(self.root, sname, f'cloud_bin_{i}.ply')
        ply_name1 = os.path.join(self.root, sname, f'cloud_bin_{j}.ply')

        if self.return_ply_names:
            return sname, ply_name0, ply_name1, T_gt

        pcd0 = o3d.io.read_point_cloud(ply_name0)
        pcd1 = o3d.io.read_point_cloud(ply_name1)
        pcd0 = np.asarray(pcd0.points)
        pcd1 = np.asarray(pcd1.points)
        return sname, pcd0, pcd1, T_gt


class IndoorPairDataset(PairDataset):
    OVERLAP_RATIO = None
    AUGMENT = None

    def __init__(self,
                 phase,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 config=None):
        PairDataset.__init__(self, phase, transform, random_rotation,
                             random_scale, manual_seed, config)
        self.root = root = config.threed_match_dir
        logging.info(f"Loading the subset {phase} from {root}")

        subset_names = open(self.DATA_FILES[phase]).read().split()
        for name in subset_names:
            fname = name + "*%.2f.txt" % self.OVERLAP_RATIO
            fnames_txt = glob.glob(root + "/" + fname)
            assert len(
                fnames_txt
            ) > 0, f"Make sure that the path {root} has data {fname}"
            for fname_txt in fnames_txt:
                with open(fname_txt) as f:
                    content = f.readlines()
                fnames = [x.strip().split() for x in content]
                for fname in fnames:
                    self.files.append([fname[0], fname[1]])

    def __getitem__(self, idx):
        file0 = os.path.join(self.root, self.files[idx][0])
        file1 = os.path.join(self.root, self.files[idx][1])
        data0 = np.load(file0)
        data1 = np.load(file1)
        xyz0 = data0["pcd"]
        xyz1 = data1["pcd"]
        color0 = data0["color"]
        color1 = data1["color"]
        matching_search_voxel_size = self.matching_search_voxel_size

        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + \
                (self.max_scale - self.min_scale) * random.random()
            matching_search_voxel_size *= scale
            xyz0 = scale * xyz0
            xyz1 = scale * xyz1

        if self.random_rotation:
            T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
            T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
            trans = T1 @ np.linalg.inv(T0)

            xyz0 = self.apply_transform(xyz0, T0)
            xyz1 = self.apply_transform(xyz1, T1)
        else:
            trans = np.identity(4)

        # Voxelization
        _, sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size,
                                           return_index=True)
        _, sel1 = ME.utils.sparse_quantize(xyz1 / self.voxel_size,
                                           return_index=True)

        # Make point clouds using voxelized points
        pcd0 = make_open3d_point_cloud(xyz0)
        pcd1 = make_open3d_point_cloud(xyz1)

        # Select features and points using the returned voxelized indices
        pcd0.colors = o3d.utility.Vector3dVector(color0[sel0])
        pcd1.colors = o3d.utility.Vector3dVector(color1[sel1])
        pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
        pcd1.points = o3d.utility.Vector3dVector(np.array(pcd1.points)[sel1])
        # Get matches
        matches = get_matching_indices(pcd0, pcd1, trans,
                                       matching_search_voxel_size)

        # Get features
        npts0 = len(pcd0.colors)
        npts1 = len(pcd1.colors)

        feats_train0, feats_train1 = [], []

        feats_train0.append(np.ones((npts0, 1)))
        feats_train1.append(np.ones((npts1, 1)))

        feats0 = np.hstack(feats_train0)
        feats1 = np.hstack(feats_train1)

        # Get coords
        xyz0 = np.array(pcd0.points)
        xyz1 = np.array(pcd1.points)

        coords0 = np.floor(xyz0 / self.voxel_size)
        coords1 = np.floor(xyz1 / self.voxel_size)

        if self.transform:
            coords0, feats0 = self.transform(coords0, feats0)
            coords1, feats1 = self.transform(coords1, feats1)

        return (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans)


class ThreeDMatchPairDataset(IndoorPairDataset):
    OVERLAP_RATIO = 0.3
    DATA_FILES = {
        'train': './config/train_3dmatch.txt',
        'val': './config/val_3dmatch.txt',
        'test': './config/test_3dmatch.txt'
    }


class KITTIPairDataset(PairDataset):
    AUGMENT = None
    DATA_FILES = {
        'train': './config/train_kitti.txt',
        'val': './config/val_kitti.txt',
        'test': './config/test_kitti.txt'
    }
    TEST_RANDOM_ROTATION = False
    IS_ODOMETRY = True

    def __init__(self,
                 phase,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 config=None):
        # For evaluation, use the odometry dataset training following the 3DFeat eval method
        if self.IS_ODOMETRY:
            self.root = root = config.kitti_root + '/dataset'
            random_rotation = self.TEST_RANDOM_ROTATION
        else:
            self.date = config.kitti_date
            self.root = root = os.path.join(config.kitti_root, self.date)

        self.icp_path = os.path.join(config.kitti_root, 'icp')
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

        PairDataset.__init__(self, phase, transform, random_rotation,
                             random_scale, manual_seed, config)

        logging.info(f"Loading the subset {phase} from {root}")
        # Use the kitti root
        self.max_time_diff = max_time_diff = config.kitti_max_time_diff

        subset_names = open(self.DATA_FILES[phase]).read().split()
        for dirname in subset_names:
            drive_id = int(dirname)
            inames = self.get_all_scan_ids(drive_id)
            for start_time in inames:
                for time_diff in range(2, max_time_diff):
                    pair_time = time_diff + start_time
                    if pair_time in inames:
                        self.files.append((drive_id, start_time, pair_time))

    def get_all_scan_ids(self, drive_id):
        if self.IS_ODOMETRY:
            fnames = glob.glob(self.root +
                               '/sequences/%02d/velodyne/*.bin' % drive_id)
        else:
            fnames = glob.glob(self.root + '/' + self.date +
                               '_drive_%04d_sync/velodyne_points/data/*.bin' %
                               drive_id)
        assert len(
            fnames
        ) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
        inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
        return inames

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02,
                7.280733e-04, -9.998902e-01, 9.998621e-01, 7.523790e-03,
                1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02,
                          -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_video_odometry(self,
                           drive,
                           indices=None,
                           ext='.txt',
                           return_all=False):
        if self.IS_ODOMETRY:
            data_path = self.root + '/poses/%02d.txt' % drive
            if data_path not in pose_cache:
                pose_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return pose_cache[data_path]
            else:
                return pose_cache[data_path][indices]
        else:
            data_path = self.root + '/' + self.date + '_drive_%04d_sync/oxts/data' % drive
            odometry = []
            if indices is None:
                fnames = glob.glob(
                    self.root + '/' + self.date +
                    '_drive_%04d_sync/velodyne_points/data/*.bin' % drive)
                indices = sorted(
                    [int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            for index in indices:
                filename = os.path.join(data_path, '%010d%s' % (index, ext))
                if filename not in pose_cache:
                    pose_cache[filename] = np.genfromtxt(filename)
                odometry.append(pose_cache[filename])

            odometry = np.array(odometry)
            return odometry

    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0
        else:
            lat, lon, alt, roll, pitch, yaw = odometry.T[:6]

            R = 6378137  # Earth's radius in metres

            # convert to metres
            lat, lon = np.deg2rad(lat), np.deg2rad(lon)
            mx = R * lon * np.cos(lat)
            my = R * lat

            times = odometry.T[-1]
            return np.vstack([mx, my, alt, roll, pitch, yaw, times]).T

    def rot3d(self, axis, angle):
        ei = np.ones(3, dtype='bool')
        ei[axis] = 0
        i = np.nonzero(ei)[0]
        m = np.eye(3)
        c, s = np.cos(angle), np.sin(angle)
        m[i[0], i[0]] = c
        m[i[0], i[1]] = -s
        m[i[1], i[0]] = s
        m[i[1], i[1]] = c
        return m

    def pos_transform(self, pos):
        x, y, z, rx, ry, rz, _ = pos[0]
        RT = np.eye(4)
        RT[:3, :3] = np.dot(np.dot(self.rot3d(0, rx), self.rot3d(1, ry)),
                            self.rot3d(2, rz))
        RT[:3, 3] = [x, y, z]
        return RT

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive,
                                                                       t)
        else:
            fname = self.root + \
                '/' + self.date + '_drive_%04d_sync/velodyne_points/data/%010d.bin' % (
                    drive, t)
        return fname

    def __getitem__(self, idx):
        drive = self.files[idx][0]
        t0, t1 = self.files[idx][1], self.files[idx][2]
        all_odometry = self.get_video_odometry(drive, [t0, t1])
        positions = [
            self.odometry_to_positions(odometry) for odometry in all_odometry
        ]
        fname0 = self._get_velodyne_fn(drive, t0)
        fname1 = self._get_velodyne_fn(drive, t1)

        # XYZ and reflectance
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]

        key = '%d_%d_%d' % (drive, t0, t1)
        filename = self.icp_path + '/' + key + '.npy'
        if key not in kitti_icp_cache:
            if not os.path.exists(filename):
                # work on the downsampled xyzs, 0.05m == 5cm
                _, sel0 = ME.utils.sparse_quantize(xyz0 / 0.05,
                                                   return_index=True)
                _, sel1 = ME.utils.sparse_quantize(xyz1 / 0.05,
                                                   return_index=True)

                M = (self.velo2cam @ positions[0].T @ np.linalg.inv(
                    positions[1].T) @ np.linalg.inv(self.velo2cam)).T
                xyz0_t = self.apply_transform(xyz0[sel0], M)
                pcd0 = make_open3d_point_cloud(xyz0_t)
                pcd1 = make_open3d_point_cloud(xyz1[sel1])
                reg = o3d.pipelines.registration.registration_icp(
                    pcd0, pcd1, 0.2, np.eye(4),
                    o3d.pipelines.registration.
                    TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=200))
                pcd0.transform(reg.transformation)
                # pcd0.transform(M2) or self.apply_transform(xyz0, M2)
                M2 = M @ reg.transformation
                # o3d.draw_geometries([pcd0, pcd1])
                # write to a file
                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            kitti_icp_cache[key] = M2
        else:
            M2 = kitti_icp_cache[key]

        if self.random_rotation:
            T0 = sample_random_trans(xyz0, self.randg, np.pi / 4)
            T1 = sample_random_trans(xyz1, self.randg, np.pi / 4)
            trans = T1 @ M2 @ np.linalg.inv(T0)

            xyz0 = self.apply_transform(xyz0, T0)
            xyz1 = self.apply_transform(xyz1, T1)
        else:
            trans = M2

        matching_search_voxel_size = self.matching_search_voxel_size
        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + \
                (self.max_scale - self.min_scale) * random.random()
            matching_search_voxel_size *= scale
            xyz0 = scale * xyz0
            xyz1 = scale * xyz1

        # Voxelization
        xyz0_th = torch.from_numpy(xyz0)
        xyz1_th = torch.from_numpy(xyz1)

        _, sel0 = ME.utils.sparse_quantize(xyz0_th / self.voxel_size,
                                           return_index=True)
        _, sel1 = ME.utils.sparse_quantize(xyz1_th / self.voxel_size,
                                           return_index=True)

        # Make point clouds using voxelized points
        pcd0 = make_open3d_point_cloud(xyz0[sel0])
        pcd1 = make_open3d_point_cloud(xyz1[sel1])

        # Get matches
        matches = get_matching_indices(pcd0, pcd1, trans,
                                       matching_search_voxel_size)
        if len(matches) < 1000:
            # raise ValueError(f"{drive}, {t0}, {t1}")
            if len(matches) == 0:
                print("length = 0!")
            print(
                f"Matching indices small at {drive}, {t0}, {t1},len()={len(matches)}"
            )

        # Get features
        npts0 = len(sel0)
        npts1 = len(sel1)

        feats_train0, feats_train1 = [], []

        unique_xyz0_th = xyz0_th[sel0]
        unique_xyz1_th = xyz1_th[sel1]

        feats_train0.append(torch.ones((npts0, 1)))
        feats_train1.append(torch.ones((npts1, 1)))

        feats0 = torch.cat(feats_train0, 1)
        feats1 = torch.cat(feats_train1, 1)

        coords0 = torch.floor(unique_xyz0_th / self.voxel_size)
        coords1 = torch.floor(unique_xyz1_th / self.voxel_size)

        if self.transform:
            coords0, feats0 = self.transform(coords0, feats0)
            coords1, feats1 = self.transform(coords1, feats1)

        return (unique_xyz0_th.float(), unique_xyz1_th.float(), coords0.int(),
                coords1.int(), feats0.float(), feats1.float(), matches, trans)


class PointDataset(torch.utils.data.Dataset):
    def __init__(self,
                 phase,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 config=None):
        self.phase = phase
        self.files = []
        self.transform = transform
        self.voxel_size = config.voxel_size
        self.random_scale = random_scale
        self.min_scale = config.min_scale
        self.max_scale = config.max_scale
        self.random_rotation = random_rotation
        self.rotation_range = config.rotation_range
        self.random_dist = True
        if 'random_dist' in [k for (k, v) in config.items()]:
            self.random_dist = config.random_dist
        self.randg = np.random.RandomState()
        if manual_seed:
            self.reset_seed()

    def reset_seed(self, seed=0):
        logging.info(f"Resetting the data loader seed to {seed}")
        self.randg.seed(seed)

    def apply_transform(self, pts, trans):
        trans = trans.astype(np.float32)
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def __len__(self):
        return len(self.files)


class KittiDataset(PointDataset):
    def get_all_scan_ids(self, drive_id):
        if self.IS_ODOMETRY:
            fnames = glob.glob(self.root +
                               '/sequences/%02d/velodyne/*.bin' % drive_id)
        else:
            fnames = glob.glob(self.root + '/' + self.date +
                               '_drive_%04d_sync/velodyne_points/data/*.bin' %
                               drive_id)
        assert len(
            fnames
        ) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
        inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
        return inames
    
    def parse_calibration(self, filename):
        calib = {}
        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()
        return calib

    def get_video_odometry(self,
                           dirname,
                           indices=None,
                           ext='.txt',
                           return_all=False):
        if type(dirname) == int or type(dirname) == np.int64:  # kitti
            # data_path = self.root + '/poses/%02d.txt' % dirname
            # if data_path not in pose_cache:
            #     pose_cache[data_path] = np.genfromtxt(data_path)
            # if return_all:
            #     return pose_cache[data_path]
            # else:
            #     return pose_cache[data_path][indices]
            data_path = self.root + '/sequences/%02d' % dirname
            calib_filename = data_path + '/calib.txt'
            pose_filename = data_path + '/poses.txt'
            calibration = self.parse_calibration(calib_filename)

            Tr = calibration["Tr"]
            Tr_inv = np.linalg.inv(Tr)

            poses = []
            pose_file = open(pose_filename)
            for line in pose_file:
                values = [float(v) for v in line.strip().split()]

                pose = np.zeros((4, 4))
                pose[0, 0:4] = values[0:4]
                pose[1, 0:4] = values[4:8]
                pose[2, 0:4] = values[8:12]
                pose[3, 3] = 1.0

                poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
            
            if pose_filename not in kitti_icp_cache:
                kitti_icp_cache[pose_filename] = np.array(poses)
            if return_all:
                return kitti_icp_cache[pose_filename]
            else:
                return kitti_icp_cache[pose_filename][indices]
        else:  # converted data format, i.e., nuscenes or waymo
            data_path = os.path.join(self.root, 'sequences', dirname,
                                     'poses.npy')
            if not os.path.exists(data_path):   # compatibility.
                data_path = os.path.join(self.root, 'sequences', dirname,
                                         'velodyne', 'poses.npy')
            if data_path not in pose_cache:
                pose_cache[data_path] = np.load(data_path)
            if return_all:
                return pose_cache[data_path]
            else:
                return pose_cache[data_path][indices]

    def odometry_to_positions(self, odometry):
        T_w_cam0 = odometry.reshape(3, 4)
        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
        return T_w_cam0


class KittiNFramePairDataset(KittiDataset):
    DATA_FILES = {
        'train': './config/train_kitti.txt',
        'val': './config/val_kitti.txt',
        'test': './config/test_kitti.txt'
    }

    def __init__(self,
                 phase,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 config=None):
        PointDataset.__init__(self, phase, transform, random_rotation,
                              random_scale, manual_seed, config)

        self.root = root = config.kitti_root + '/dataset'
        self.matching_search_voxel_size = \
          config.voxel_size * config.positive_pair_search_voxel_size_multiplier
        self.MIN_DIST = config.pair_min_dist
        self.MAX_DIST = config.pair_max_dist
        self.supervised = config.supervised

        logging.info(f"Loading the subset {phase} from {root}")
        self.phase = phase
        self.config = config

        try:
            self.skip_initialization = config.skip_initialization
        except:
            self.skip_initialization = False

        if phase == 'test' and config.LoKITTI == True:
            # load LoKITTI point cloud pairs, instead of generating them based on frame distance
            self.files = np.load("config/file_LoKITTI_50.npy")
        else:
            self.prepare_kitty_ply(phase)
        print(f"Data size for phase {phase}: {len(self.files)}")
        # self._debug_get_maximum_distance(phase)

    def prepare_kitty_ply(self, phase):
        # load all frames that are several frames apart
        subset_names = open(self.DATA_FILES[phase]).read().split()
        for dirname in subset_names:
            drive_id = int(dirname)
            print(f"Processing drive {drive_id}")
            fnames = glob.glob(self.root +
                               '/sequences/%02d/velodyne/*.bin' % drive_id)
            assert len(
                fnames
            ) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted(
                [int(os.path.split(fname)[-1][:-4]) for fname in fnames])
            curr_time = inames[0]

            np.random.seed(0)
            while curr_time in inames:
                # calculate the distance (by random or not)
                dist_tmp = max(
                    1,
                    int(self.MIN_DIST + np.random.rand() *
                       (self.MAX_DIST - self.MIN_DIST)))

                if (curr_time + dist_tmp) in inames:
                    next_time = curr_time + dist_tmp
                    self.files.append((drive_id, curr_time, next_time))

                    # curr_time = next_time + max(1, int(10-self.MAX_DIST))
                    curr_time += 11
                else:
                    curr_time += 1
        if self.config.percentage != 1.0:
            print(f"Reducing dataset to the first {self.config.percentage*100:.1f} %.")
            self.files = self.files[:int(len(self.files)*self.config.percentage)]
            print(f"dataset length after reduction: {len(self.files)}")

    def _debug_get_maximum_distance(self, phase):
        # load all frames that are several meters apart
        subset_names = open(self.DATA_FILES[phase]).read().split()
        maximum_distance = -1
        last_drive_id = -1
        for (drive_id, curr_time, next_time) in self.files:
            if last_drive_id != int(drive_id):
                print(f"Reflecting distances on drive {drive_id}")
                all_pos = self.get_slam_odometry(drive_id, return_all=True)
                self.Ts = all_pos[:, :3, 3]
                last_drive_id = drive_id

            maximum_distance = np.max([np.linalg.norm(self.Ts[curr_time] - self.Ts[next_time]), maximum_distance])
        print(f"maximum distance for frame range [{self.MIN_DIST}, {self.MAX_DIST}]: {maximum_distance}m")
        raise ValueError

    def parse_calibration(self, filename):
        calib = {}
        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()
        return calib

    def get_slam_odometry(self, drive, indices=None, return_all=False):
        data_path = self.root + '/sequences/%02d' % drive
        calib_filename = data_path + '/calib.txt'
        pose_filename = data_path + '/poses.txt'
        calibration = self.parse_calibration(calib_filename)

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        poses = []
        pose_file = open(pose_filename)
        for line in pose_file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        if pose_filename not in kitti_icp_cache:
            kitti_icp_cache[pose_filename] = np.array(poses)
        if return_all:
            return kitti_icp_cache[pose_filename]
        else:
            return kitti_icp_cache[pose_filename][indices]

    def _get_velodyne_fn(self, drive, t):
        return self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)

    def _get_labels_fn(self, drive, t):
        return self.root + '/sequences/%02d/labels/%06d.label' % (drive, t)

    # simple function for getting the xyz point-cloud w.r.t drive and time
    def _get_xyz(self, drive, time):
        fname = self._get_velodyne_fn(drive, time)
        xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
        return xyzr[:, :3]

    def _get_semantic_label(self, drive, time):
        fname = self._get_labels_fn(drive, time)
        labels = np.fromfile(fname, dtype=np.int16).reshape(-1, 2)
        return labels[:, 0]
    
    def get_matching_indices_pytorch3d(self, source, target, trans, search_voxel_size, K=None):
        source_copy, target_copy = source.to('cuda:0'), target.to('cuda:0')
        
        P1 = pytorch3d.structures.Pointclouds([source_copy])
        P2 = pytorch3d.structures.Pointclouds([target_copy])
        P1_F = P1.points_padded()
        P2_F = P2.points_padded()
        P1_N = P1.num_points_per_cloud()
        P2_N = P2.num_points_per_cloud()
        _, idx_1, _ = knn_points(P1_F, P2_F, P1_N, P2_N, K=1)

        target_to_source_corr = target_copy[idx_1[0][:,0]]
        within_range_mask = (torch.norm(source_copy - target_to_source_corr, dim=1) < search_voxel_size)
        idx_1 = idx_1[0][within_range_mask]

        pos_sel_1 = torch.arange(len(source_copy)).to('cuda:0').long()
        match_inds = (torch.cat([pos_sel_1.unsqueeze(1), idx_1[pos_sel_1]], dim=1)).cpu().detach().tolist()
        return match_inds

    def __getitem__(self, idx):
        # Note that preparation procedures with or without complement frames are very much different,
        # we might as well just throw them in an if-else case, for simplicity of tuning and debugging
        prepare_timer, icp_timer, rot_crop_timer = Timer(), Timer(), Timer()
        prepare_timer.tic()
        drive, t_0, t_1 = self.files[idx]
        positions = self.get_slam_odometry(drive, [t_0, t_1])
        pos_0, pos_1 = positions[0:2]

        # load two center point clouds
        xyz_0 = self._get_xyz(drive, t_0)
        xyz_1 = self._get_xyz(drive, t_1)
        prepare_timer.toc()

        icp_timer.tic()
        # determine icp result between t0 and t1
        # This item is not used during training. Only testing involves calculating RR using this label.
        M2 = np.linalg.inv(pos_1) @ pos_0
        icp_timer.toc()

        # add random rotation if needed, note that the same rotation is applied to both curr and nghb
        rot_crop_timer.tic()
        if self.random_rotation:
            T0 = sample_random_trans(xyz_0, self.randg, np.pi / 4)
            T1 = sample_random_trans(xyz_1, self.randg, np.pi / 4)
            trans = T1 @ M2 @ np.linalg.inv(T0)

            xyz_0 = self.apply_transform(xyz_0, T0)
            xyz_1 = self.apply_transform(xyz_1, T1)
        else:
            trans = M2
        rot_crop_timer.toc()

        # random scaling
        matching_search_voxel_size = self.matching_search_voxel_size
        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + \
                (self.max_scale - self.min_scale) * random.random()
            matching_search_voxel_size *= scale
            xyz_0 = scale * xyz_0
            xyz_1 = scale * xyz_1
            trans[:3, 3] = scale * trans[:3, 3]

        # voxelization
        xyz_0 = torch.from_numpy(xyz_0)
        xyz_1 = torch.from_numpy(xyz_1)

        # Make point clouds using voxelized points
        _, sel_0 = ME.utils.sparse_quantize(xyz_0 / self.voxel_size,
                                            return_index=True)
        _, sel_1 = ME.utils.sparse_quantize(xyz_1 / self.voxel_size,
                                            return_index=True)

        pcd_0 = make_open3d_point_cloud(xyz_0[sel_0])
        pcd_1 = make_open3d_point_cloud(xyz_1[sel_1])

        if self.MAX_DIST <= 1 and self.phase == "train" and not self.skip_initialization:
            # Base mode training, we default an identity transformation.
            matches = get_matching_indices(pcd_0, pcd_1, np.identity(4), matching_search_voxel_size)
            # matches = self.get_matching_indices_pytorch3d(xyz_0[sel_0], xyz_1[sel_1], np.identity(4), matching_search_voxel_size)
        elif self.phase != "train" or self.supervised == True:
            # Extension mode val/test, GT pose is used to assess model performance.
            matches = get_matching_indices(pcd_0, pcd_1, trans, matching_search_voxel_size)
        else:
            # Extenstion mode training, fill in dummy value for compatibility
            matches = torch.zeros((1,2))
        if len(matches) == 0:
            # if two PCs share no overlap, then randomly subtitute another pair
            # print(f"Matching indices small at {drive}, {t_0}, {t_1},len()={len(matches)}")
            return self.__getitem__(np.random.choice(self.__len__(), 1)[0])

        # apply voxelization
        xyz_0_th = xyz_0[sel_0]
        xyz_1_th = xyz_1[sel_1]
        del sel_0
        del sel_1

        coords_0 = torch.floor(xyz_0_th / self.voxel_size)
        coords_1 = torch.floor(xyz_1_th / self.voxel_size)
        feats_0 = torch.ones((len(coords_0), 1))
        feats_1 = torch.ones((len(coords_1), 1))

        if self.transform:
            coords_0, feats_0 = self.transform(coords_0, feats_0)
            coords_1, feats_1 = self.transform(coords_1, feats_1)

        return (xyz_0_th.float(), xyz_1_th.float(), coords_0.int(),
                coords_1.int(), feats_0.float(), feats_1.float(), matches, trans, t_1-t_0)


class NuscenesNFramePairDataset(KittiDataset):
    icp_voxel_size = 0.05  # 0.05 meters, i.e. 5cm
    TEST_RANDOM_ROTATION = False

    def __init__(self,
                 phase,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 config=None):
        PointDataset.__init__(self, phase, transform, random_rotation,
                              random_scale, manual_seed, config)

        self.root = root = os.path.join(config.kitti_root, phase)
        self.matching_search_voxel_size = \
          config.voxel_size * config.positive_pair_search_voxel_size_multiplier
        self.MIN_DIST = config.pair_min_dist
        self.MAX_DIST = config.pair_max_dist
        self.supervised = config.supervised

        self.phase = phase
        self.config = config

        logging.info(f"Loading the subset {phase} from {root}")
        self.files = []
        self.nuscenes_icp_cache = {}
        pose_cache = {}

        # load LoNuscenes point cloud pairs, instead of generating them based on distance
        if phase == 'test' and config.LoNUSCENES == True:
            self.files = np.load("config/file_LoNUSCENES_50.npy",
                                 allow_pickle=True)
        else:
            self.prepare_nuscenes_ply(phase)
        print(f"Data size for phase {phase}: {len(self.files)}")

    def prepare_nuscenes_ply(self, phase):
        # load all frames that are several frames apart
        subset_names = os.listdir(os.path.join(self.root, 'sequences'))
        for dirname in subset_names:
            print(f"Processing log {dirname}")
            fnames = glob.glob(self.root +
                                '/sequences/%s/velodyne/*.bin' % dirname)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted(
                [int(os.path.split(fname)[-1][:-4]) for fname in fnames])
            curr_time = inames[0]

            all_pos = self.get_video_odometry(dirname, return_all=True)
            Ts = all_pos[:, :3, 3]

            np.random.seed(0)
            while curr_time in inames:
                # calculate the distance (by random or not)
                dist_tmp = max(
                    1,
                    int(self.MIN_DIST + np.random.rand() *
                       (self.MAX_DIST - self.MIN_DIST)))

                if (curr_time + dist_tmp) in inames:
                    next_time = curr_time + dist_tmp

                    # NuScenes has some drive sequences that are discontinuous in ego trajectory.
                    # We guess that the missing parts are cut off due to privacy issues.
                    # It is vital to identify these discontinuities (through exceptionally large GT translation vector),
                    # so that frames on two sides of the discontinuity are not considered as registerable frames any more.
                    # E.g., the sequence "n015-2018-08-03-15-21-40+0800" has multiple discontinuities.
                    if np.linalg.norm(Ts[curr_time] - Ts[next_time]) > 100: # An emperical distance. Justification: PCs that far are unsuitble for registration whatsoever.
                        curr_time += 1
                    else:
                        self.files.append((dirname, curr_time, next_time))
                        # curr_time = next_time + 1
                        curr_time += 8
                else:
                    curr_time += 1

        if self.config.percentage != 1.0:
            print(f"Reducing dataset to the first {self.config.percentage*100:.1f} %.")
            self.files = self.files[:int(len(self.files)*self.config.percentage)]
            print(f"dataset length after reduction: {len(self.files)}")
        if phase == 'train':
            pass
            # print(
            #     f"Data size for phase {phase} before pruning: {len(self.files)}"
            # )
            # self.files = self.files[::3]
            # self.files = self.files[:1200]
        # self._debug_get_maximum_distance(phase)

    def _debug_get_maximum_distance(self, phase):
        # load all frames that are several meters apart
        maximum_distance = -1
        last_drive_id = "dummy_value"
        for (drive_id, curr_time, next_time) in self.files:
            if last_drive_id != drive_id:
                print(f"Reflecting distances on drive {drive_id}")
                all_pos = self.get_video_odometry(drive_id, return_all=True)
                self.Ts = all_pos[:, :3, 3]
                last_drive_id = drive_id
                maximum_distance = np.max([np.linalg.norm(self.Ts[curr_time] - self.Ts[next_time]), maximum_distance])
        print(f"maximum distance for frame range [{self.MIN_DIST}, {self.MAX_DIST}]: {maximum_distance}m")
        raise ValueError

    def _get_velodyne_fn(self, dirname, t):
        fname = self.root + '/sequences/%s/velodyne/%06d.bin' % (dirname, t)
        return fname

    # simple function for getting the xyz point-cloud w.r.t drive and time
    def _get_xyz(self, drive, time):
        fname = self._get_velodyne_fn(drive, time)
        xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
        return xyzr[:, :3]

    def __getitem__(self, idx):
        # Note that preparation procedures with or without complement frames are very much different,
        # we might as well just throw them in an if-else case, for simplicity of tuning and debugging
        prepare_timer, icp_timer, rot_crop_timer = Timer(), Timer(), Timer()
        prepare_timer.tic()
        dirname, t_0, t_1 = self.files[idx]

        positions = self.get_video_odometry(dirname, [t_0, t_1])

        # load two center point clouds
        xyz_0 = self._get_xyz(dirname, t_0)
        xyz_1 = self._get_xyz(dirname, t_1)
        prepare_timer.toc()

        icp_timer.tic()
        # determine icp result between t0 and t1
        # note that this label is not used during training.
        M2 = np.linalg.inv(positions[1]) @ positions[0]
        icp_timer.toc()

        # add random rotation if needed, note that the same rotation is applied to both curr and nghb
        rot_crop_timer.tic()
        if self.random_rotation:
            T0 = sample_random_trans(xyz_0, self.randg, np.pi / 4)
            T1 = sample_random_trans(xyz_1, self.randg, np.pi / 4)
            trans = T1 @ M2 @ np.linalg.inv(T0)

            xyz_0 = self.apply_transform(xyz_0, T0)
            xyz_1 = self.apply_transform(xyz_1, T1)
        else:
            trans = M2
        rot_crop_timer.toc()

        # random scaling
        matching_search_voxel_size = self.matching_search_voxel_size
        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + \
                (self.max_scale - self.min_scale) * random.random()
            matching_search_voxel_size *= scale
            xyz_0 = scale * xyz_0
            xyz_1 = scale * xyz_1
            trans[:3, 3] = scale * trans[:3, 3]

        # voxelization
        xyz_0 = torch.from_numpy(xyz_0)
        xyz_1 = torch.from_numpy(xyz_1)

        # Make point clouds using voxelized points
        _, sel_0 = ME.utils.sparse_quantize(xyz_0 / self.voxel_size,
                                            return_index=True)
        _, sel_1 = ME.utils.sparse_quantize(xyz_1 / self.voxel_size,
                                            return_index=True)

        pcd_0 = make_open3d_point_cloud(xyz_0[sel_0])
        pcd_1 = make_open3d_point_cloud(xyz_1[sel_1])

        if self.MAX_DIST <= 1 and self.phase == "train":
            # Base mode training, we default an identity transformation.
            matches = get_matching_indices(pcd_0, pcd_1, np.identity(4), matching_search_voxel_size)
        elif self.phase != "train" or self.supervised == True:
            # Extension mode val/test, or manually forced supervised training during comparison, where GT pose is used.
            matches = get_matching_indices(pcd_0, pcd_1, trans, matching_search_voxel_size)
        else:
            # Extenstion mode training, fill in dummy value for compatibility
            matches = torch.zeros((1,2))

        if len(matches) == 0:
            # if two PCs share no overlap, then randomly subtitute another pair
            # print(f"Matching indices small at {dirname}, {t_0}, {t_1},len()={len(matches)}")
            return self.__getitem__(np.random.choice(self.__len__(), 1)[0])
        
        # apply voxelization
        xyz_0_th = xyz_0[sel_0]
        xyz_1_th = xyz_1[sel_1]
        del sel_0
        del sel_1

        coords_0 = torch.floor(xyz_0_th / self.voxel_size)
        coords_1 = torch.floor(xyz_1_th / self.voxel_size)
        feats_0 = torch.ones((len(coords_0), 1))
        feats_1 = torch.ones((len(coords_1), 1))

        if self.transform:
            coords_0, feats_0 = self.transform(coords_0, feats_0)
            coords_1, feats_1 = self.transform(coords_1, feats_1)

        return (xyz_0_th.float(), xyz_1_th.float(), coords_0.int(),
                coords_1.int(), feats_0.float(), feats_1.float(), matches, trans, t_1-t_0)
    


class WaymoNFramePairDataset(KittiDataset):
    phase_dir_mapping = {"train": "training",
                         "val": "validation",
                         "test": "testing"}
    
    def __init__(self,
                 phase,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 config=None):
        PointDataset.__init__(self, phase, transform, random_rotation,
                              random_scale, manual_seed, config)

        self.root = root = os.path.join(config.kitti_root, self.phase_dir_mapping[phase])
        self.matching_search_voxel_size = \
          config.voxel_size * config.positive_pair_search_voxel_size_multiplier
        self.MIN_DIST = config.pair_min_dist
        self.MAX_DIST = config.pair_max_dist
        self.supervised = config.supervised

        logging.info(f"Loading the subset {phase} from {root}")
        self.phase = phase

        self.config = config

        if phase == 'test' and config.LoWAYMO == True:
            raise NotImplementedError("LoWaymo has yet to be built!")
        else:
            self.prepare_waymo_ply(phase)
        print(f"Data size for phase {phase}: {len(self.files)}")
        # self._debug_get_maximum_distance(phase)

    def prepare_waymo_ply(self, phase):
        # load all frames that are several frames apart
        subset_names = os.listdir(os.path.join(self.root, 'sequences'))
        for i, dirname in enumerate(subset_names):
            if not i%100:
                print(f"Processing the {i+1}'th drive, {dirname=}")
            fnames = glob.glob(self.root +
                               '/sequences/%s/velodyne/*.bin' % dirname)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted(
                [int(os.path.split(fname)[-1][:-4]) for fname in fnames])
            curr_time = inames[0]

            np.random.seed(0)
            while curr_time in inames:
                # calculate the distance (by random or not)
                dist_tmp = max(
                    1,
                    int(self.MIN_DIST + np.random.rand() *
                       (self.MAX_DIST - self.MIN_DIST)))

                if (curr_time + dist_tmp) in inames:
                    next_time = curr_time + dist_tmp
                    self.files.append((dirname, curr_time, next_time))

                    # curr_time = next_time + 1
                    curr_time += 8
                else:
                    curr_time += 1
        if self.config.percentage != 1.0:
            print(f"Reducing dataset to the first {self.config.percentage*100:.1f} %.")
            self.files = self.files[:int(len(self.files)*self.config.percentage)]
            print(f"dataset length after reduction: {len(self.files)}")
        if phase == 'train':
            # pass
            print(
                f"Data size for phase {phase} before pruning: {len(self.files)}"
            )
            self.files = self.files[::4]
            # self.files = self.files[:1200]
        if phase == 'val':
            # pass
            print(
                f"Data size for phase {phase} before pruning: {len(self.files)}"
            )
            self.files = self.files[::4]

    def _debug_get_maximum_distance(self, phase):
        # load all frames that are several meters apart
        maximum_distance = -1
        last_drive_id = "dummy_value"
        for (drive_id, curr_time, next_time) in self.files:
            if last_drive_id != drive_id:
                print(f"Reflecting distances on drive {drive_id}")
                all_pos = self.get_video_odometry(drive_id, return_all=True)
                all_pos = all_pos.reshape(-1,4,4)

                self.Ts = all_pos[:, :3, 3]
                last_drive_id = drive_id
                maximum_distance = np.max([np.linalg.norm(self.Ts[curr_time] - self.Ts[next_time]), maximum_distance])
        print(f"maximum distance for frame range [{self.MIN_DIST}, {self.MAX_DIST}]: {maximum_distance}m")
        raise ValueError

    def _get_velodyne_fn(self, dirname, t):
        fname = self.root + '/sequences/%s/velodyne/%06d.bin' % (dirname, t)
        return fname

    # simple function for getting the xyz point-cloud w.r.t drive and time
    # note that the dataset conversion code did not save the reflection component.
    def _get_xyz(self, drive, time):
        fname = self._get_velodyne_fn(drive, time)
        xyz = np.fromfile(fname, dtype=np.float32).reshape(-1, 3)
        return xyz
    
    def __getitem__(self, idx):
        # Note that preparation procedures with or without complement frames are very much different,
        # we might as well just throw them in an if-else case, for simplicity of tuning and debugging
        prepare_timer, icp_timer, rot_crop_timer = Timer(), Timer(), Timer()
        prepare_timer.tic()
        drive, t_0, t_1 = self.files[idx]
        positions = self.get_video_odometry(drive, [t_0, t_1])

        pos_0 = positions[0].reshape(4,4)
        pos_1 = positions[1].reshape(4,4)

        # load two center point clouds
        xyz_0 = self._get_xyz(drive, t_0)
        xyz_1 = self._get_xyz(drive, t_1)

        _, sel_curr_0 = ME.utils.sparse_quantize(xyz_0 / 0.05,
                                                 return_index=True)
        _, sel_curr_1 = ME.utils.sparse_quantize(xyz_1 / 0.05,
                                                 return_index=True)
        pcd_0 = make_open3d_point_cloud(xyz_0[sel_curr_0])
        pcd_1 = make_open3d_point_cloud(xyz_1[sel_curr_1])
        del sel_curr_0
        del sel_curr_1
        prepare_timer.toc()

        icp_timer.tic()
        # determine icp result between t0 and t1
        # This item is not used during training. Only testing involves calculating RR using this label.
        M2 = np.linalg.inv(pos_1) @ pos_0
        icp_timer.toc()

        # add random rotation if needed, note that the same rotation is applied to both curr and nghb
        rot_crop_timer.tic()
        if self.random_rotation:
            T0 = sample_random_trans(xyz_0, self.randg, np.pi / 4)
            T1 = sample_random_trans(xyz_1, self.randg, np.pi / 4)
            trans = T1 @ M2 @ np.linalg.inv(T0)

            xyz_0 = self.apply_transform(xyz_0, T0)
            xyz_1 = self.apply_transform(xyz_1, T1)
        else:
            trans = M2
        rot_crop_timer.toc()

        # random scaling
        matching_search_voxel_size = self.matching_search_voxel_size
        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + \
                (self.max_scale - self.min_scale) * random.random()
            matching_search_voxel_size *= scale
            xyz_0 = scale * xyz_0
            xyz_1 = scale * xyz_1
            trans[:3, 3] = scale * trans[:3, 3]

        # voxelization
        xyz_0 = torch.from_numpy(xyz_0)
        xyz_1 = torch.from_numpy(xyz_1)

        # Make point clouds using voxelized points
        _, sel_0 = ME.utils.sparse_quantize(xyz_0 / self.voxel_size,
                                            return_index=True)
        _, sel_1 = ME.utils.sparse_quantize(xyz_1 / self.voxel_size,
                                            return_index=True)

        pcd_0 = make_open3d_point_cloud(xyz_0[sel_0])
        pcd_1 = make_open3d_point_cloud(xyz_1[sel_1])

        if self.MAX_DIST <= 1 and self.phase == "train":
            # Base mode training, we default an identity transformation.
            matches = get_matching_indices(pcd_0, pcd_1, np.identity(4), matching_search_voxel_size)
        elif self.phase != "train" or self.supervised == True:
            # Extension mode val/test, or manually forced supervised training during comparison, where GT pose is used.
            matches = get_matching_indices(pcd_0, pcd_1, trans, matching_search_voxel_size)
        else:
            # Extenstion mode training, fill in dummy value for compatibility
            matches = torch.zeros((1,2))

        # apply voxelization
        xyz_0_th = xyz_0[sel_0]
        xyz_1_th = xyz_1[sel_1]
        del sel_0
        del sel_1

        coords_0 = torch.floor(xyz_0_th / self.voxel_size)
        coords_1 = torch.floor(xyz_1_th / self.voxel_size)
        feats_0 = torch.ones((len(coords_0), 1))
        feats_1 = torch.ones((len(coords_1), 1))

        if self.transform:
            coords_0, feats_0 = self.transform(coords_0, feats_0)
            coords_1, feats_1 = self.transform(coords_1, feats_1)

        # note: we now provide voxelized neighbourhood.
        # whether unvoxelized pcd performs better is still unclear.
        return (xyz_0_th.float(), xyz_1_th.float(), coords_0.int(),
                coords_1.int(), feats_0.float(), feats_1.float(), matches, trans, t_1-t_0)
    

class KittiRandDistPairDataset(KittiNFramePairDataset):
    def parse_calibration(self, filename):
        calib = {}
        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()
        return calib
    
    def get_slam_odometry(self, drive, indices=None, return_all=False):
        data_path = self.root + '/sequences/%02d' % drive
        calib_filename = data_path + '/calib.txt'
        pose_filename = data_path + '/poses.txt'
        calibration = self.parse_calibration(calib_filename)

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        poses = []
        pose_file = open(pose_filename)
        for line in pose_file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
        
        if pose_filename not in kitti_icp_cache:
            kitti_icp_cache[pose_filename] = np.array(poses)
        if return_all:
            return kitti_icp_cache[pose_filename]
        else:
            return kitti_icp_cache[pose_filename][indices]

    def prepare_kitty_ply(self, phase):
        # load all frames that are several meters apart
        subset_names = open(self.DATA_FILES[phase]).read().split()
        for dirname in subset_names:
            drive_id = int(dirname)
            print(f"Processing drive {drive_id}")
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            all_pos = self.get_slam_odometry(drive_id, return_all=True)
            self.Ts = all_pos[:, :3, 3]

            curr_time = inames[0]

            np.random.seed(0)
            while curr_time in inames:
                dist_tmp = self.MIN_DIST + np.random.rand() * (self.MAX_DIST - self.MIN_DIST)

                right_dist = np.sqrt(((self.Ts[curr_time: curr_time+int(10*self.MAX_DIST)] - 
                                    self.Ts[curr_time].reshape(1, 3))**2).sum(-1))
                # Find the min index
                next_time = np.where(right_dist > dist_tmp)[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1
                    if next_time in inames:
                        self.files.append((drive_id, curr_time, next_time))
                        curr_time += 11
                    else:
                        curr_time += 1
        if phase == 'test':
            self.files = self.files[::3]



class NuscenesRandDistPairDataset(NuscenesNFramePairDataset):
    def get_video_odometry(self, dirname, indices=None, ext='.txt', return_all=False):
        data_path = os.path.join(self.root, 'sequences', dirname, 'poses.npy')
        if data_path not in pose_cache:
            pose_cache[data_path] = np.load(data_path)
        if return_all:
            return pose_cache[data_path]
        else:
            return pose_cache[data_path][indices]
            
    def prepare_nuscenes_ply(self, phase):
        # load all frames that are several meters apart
        subset_names = os.listdir(os.path.join(self.root, 'sequences'))
        for dirname in subset_names:
            print(f"Processing log {dirname}")
            fnames = glob.glob(self.root + '/sequences/%s/velodyne/*.bin' % dirname)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            all_pos = self.get_video_odometry(dirname, return_all=True)
            self.Ts = all_pos[:, :3, 3]

            curr_time = inames[0]
            np.random.seed(0)
            while curr_time in inames:
                # calculate the distance (by random or not)
                dist_tmp = self.MIN_DIST + np.random.rand() * (self.MAX_DIST - self.MIN_DIST)

                right_dist = np.sqrt(((self.Ts[curr_time: curr_time+int(10*self.MAX_DIST)] - 
                                    self.Ts[curr_time].reshape(1, 3))**2).sum(-1))
                # Find the min index
                next_time = np.where(right_dist > dist_tmp)[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1
                    if next_time in inames:
                        self.files.append((dirname, curr_time, next_time))
                        # curr_time = next_time + max(1, int(10 -self.MAX_DIST))
                        curr_time += 11
                    else:
                        curr_time += 1
        if phase == 'train':
            self.files = self.files[::3]
            self.files = self.files[:1200]

        if phase == 'test':
            self.files = self.files[::3]
        print(f"Data size for phase {phase}: {len(self.files)}")


class WaymoRandDistPairDataset(WaymoNFramePairDataset):
           
    def prepare_waymo_ply(self, phase):
        # load all frames that are several meters apart
        subset_names = os.listdir(os.path.join(self.root, 'sequences'))
        for i, dirname in enumerate(subset_names):
            if not i%100:
                print(f"Processing the {i+1}'th drive, {dirname=}")
            fnames = glob.glob(self.root +
                               '/sequences/%s/velodyne/*.bin' % dirname)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            all_pos = self.get_video_odometry(dirname, return_all=True)
            all_pos = all_pos.reshape(-1,4,4)

            self.Ts = all_pos[:, :3, 3]
            
            curr_time = inames[0]
            np.random.seed(0)
            while curr_time in inames:
                # calculate the distance (by random or not)
                dist_tmp = self.MIN_DIST + np.random.rand() * (self.MAX_DIST - self.MIN_DIST)

                right_dist = np.sqrt(((self.Ts[curr_time: curr_time+int(10*self.MAX_DIST)] - 
                                    self.Ts[curr_time].reshape(1, 3))**2).sum(-1))
                # Find the min index
                next_time = np.where(right_dist > dist_tmp)[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1
                    if next_time in inames:
                        self.files.append((dirname, curr_time, next_time))
                        # curr_time = next_time + max(1, int(10-self.MAX_DIST))
                        curr_time += 11
                    else:
                        curr_time += 1
        if phase == 'train':
            # pass
            print(
                f"Data size for phase {phase} before pruning: {len(self.files)}"
            )
            self.files = self.files[::4]
            # self.files = self.files[:1200]
        if phase == 'val':
            # pass
            print(
                f"Data size for phase {phase} before pruning: {len(self.files)}"
            )
            self.files = self.files[::4]
        print(f"Data size for phase {phase}: {len(self.files)}")    


class KittiContinuousFramePairDataset(KittiNFramePairDataset):
    '''
    The typical alternative of "gradual" extension.

    The current implementation adopts a simple linear mapping from epoch to frame distance.
    '''

    def __init__(self,
                 phase,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 config=None):
        # These parameters now have different meanings.
        # 'pair_min_dist' and 'pair_max_dist' now represent the maximum frame distance at the beginning and end of the training, respectively.
        # The original minimal frame distance will also be fixed at 'pair_min_dist' frames.
        # The dataset.files is gradually updated per epoch (or based on a preset epoch interval) by using 'update_extension_distance()' function.
        self.FIRST_DIST = config.pair_min_dist
        self.LAST_DIST = config.pair_max_dist

        if phase == 'train':
            config.pair_min_dist = self.FIRST_DIST
            config.pair_max_dist = self.FIRST_DIST

            # utilize previous code to help us load the initial dataset.
            KittiNFramePairDataset.__init__(self, phase, transform, random_rotation,
                                            random_scale, manual_seed, config)

            config.pair_min_dist = self.FIRST_DIST
            config.pair_max_dist = self.LAST_DIST
        else:
            # utilize previous code to help us load the initial dataset.
            KittiNFramePairDataset.__init__(self, phase, transform, random_rotation,
                                            random_scale, manual_seed, config)

        self.max_epoch = config.max_epoch - 1
        self.last_altered_epoch = 0

        if config.extension_steps > 0:  # treat it as the interval
            self.extension_epoch_interval = int(config.max_epoch / config.extension_steps)
        elif config.extension_steps == 0:  # simply alter the dataset at every epoch
            self.extension_epoch_interval = 1

    def update_extension_distance(self, epoch):
        # If the interval has not been reached since the last extension, then do nothing.
        if not (epoch - self.last_altered_epoch >= self.extension_epoch_interval):
            print(f"Checking {self.phase} dataset: No need to update ({epoch}, latest altered at {self.last_altered_epoch}).")
            return False
        
        # if fractional extension step is floored away and the old value remains, then do nothing.
        expected_max_dist = int((self.LAST_DIST - self.FIRST_DIST) * (epoch/self.max_epoch)) + self.FIRST_DIST
        if expected_max_dist == self.MAX_DIST:
            print(f"Checking {self.phase} dataset: No need to update ({epoch}, latest altered at {self.last_altered_epoch}).")
            return False

        # otherwise, update the dataset.
        self.MAX_DIST = expected_max_dist
        self.last_altered_epoch = epoch

        self.files = []
        self.prepare_kitty_ply(self.phase)
        print(f"Dataset extension for phase {self.phase}: {self.MAX_DIST=}")
        print(f"Data size for epoch {epoch} has been updated to: {len(self.files)}")
        return self.MAX_DIST

    def is_base_dataset(self):
        return self.MAX_DIST == 1


class WaymoContinuousFramePairDataset(WaymoNFramePairDataset):
    '''
    The typical alternative of "gradual" extension.
    This one is on Waymo dataset.
    The current implementation adopts a simple linear mapping from epoch to frame distance.
    '''

    def __init__(self,
                 phase,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 config=None):
        # These parameters now have different meanings.
        # pair_min_dist and pair_max_dist now represent the maximum frame distance at the beginning and end of the training, respectively.
        # The original minimal frame distance will also be fixed at 'pair_min_dist' frames.
        # The dataset.files is gradually updated per epoch (or based on a preset epoch interval) by using 'update_extension_distance()' function.
        self.FIRST_DIST = config.pair_min_dist
        self.LAST_DIST = config.pair_max_dist

        if phase == "train":
            config.pair_min_dist = self.FIRST_DIST
            config.pair_max_dist = self.FIRST_DIST

            # utilize previous code to help us load the initial dataset.
            WaymoNFramePairDataset.__init__(self, phase, transform, random_rotation,
                                            random_scale, manual_seed, config)

            config.pair_min_dist = self.FIRST_DIST
            config.pair_max_dist = self.LAST_DIST
        else:
            config.pair_min_dist = self.FIRST_DIST
            config.pair_max_dist = self.LAST_DIST

            # utilize previous code to help us load the initial dataset.
            WaymoNFramePairDataset.__init__(self, phase, transform, random_rotation,
                                            random_scale, manual_seed, config)

        self.max_epoch = config.max_epoch - 1
        self.last_altered_epoch = 0

        if config.extension_steps > 0:  # treat it as the interval
            self.extension_epoch_interval = int(config.max_epoch / config.extension_steps)
        elif config.extension_steps == 0:  # simply alter the dataset at every epoch
            self.extension_epoch_interval = 1

    def update_extension_distance(self, epoch):
        # If the interval has not been reached since the last extension, then do nothing.
        if not (epoch - self.last_altered_epoch >= self.extension_epoch_interval):
            print(f"Checking {self.phase} dataset: No need to update ({epoch}, latest altered at {self.last_altered_epoch}).")
            return False
        
        # if fractional extension step is floored away and the old value remains, then do nothing.
        expected_max_dist = int((self.LAST_DIST - self.FIRST_DIST) * (epoch/self.max_epoch)) + self.FIRST_DIST
        if expected_max_dist == self.MAX_DIST:
            print(f"Checking {self.phase} dataset: No need to update ({epoch}, latest altered at {self.last_altered_epoch}).")
            return False

        # otherwise, update the dataset.
        self.MAX_DIST = expected_max_dist
        self.last_altered_epoch = epoch

        self.files = []
        self.prepare_waymo_ply(self.phase)
        print(f"Dataset extension for phase {self.phase}: {self.MAX_DIST=}")
        print(f"Data size for epoch {epoch} has been updated to: {len(self.files)}")
        return self.MAX_DIST

    def is_base_dataset(self):
        return self.MAX_DIST == 1


class NuscenesContinuousFramePairDataset(NuscenesNFramePairDataset):
    '''
    The typical alternative of "gradual" extension.
    This one is on nuScenes dataset.
    The current implementation adopts a simple linear mapping from epoch to frame distance.
    '''

    def __init__(self,
                 phase,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 config=None):
        # These parameters now have different meanings.
        # pair_min_dist and pair_max_dist now represent the maximum frame distance at the beginning and end of the training, respectively.
        # The original minimal frame distance will also be fixed at 'pair_min_dist' frames.
        # The dataset.files is gradually updated per epoch (or based on a preset epoch interval) by using 'update_extension_distance()' function.
        self.FIRST_DIST = config.pair_min_dist
        self.LAST_DIST = config.pair_max_dist

        if phase == "train":
            config.pair_min_dist = self.FIRST_DIST
            config.pair_max_dist = self.FIRST_DIST

            # utilize previous code to help us load the initial dataset.
            NuscenesNFramePairDataset.__init__(self, phase, transform, random_rotation,
                                               random_scale, manual_seed, config)

            config.pair_min_dist = self.FIRST_DIST
            config.pair_max_dist = self.LAST_DIST
        else:
            config.pair_min_dist = self.FIRST_DIST
            config.pair_max_dist = self.LAST_DIST

            # load the complete val set during validation all at once, without extension
            NuscenesNFramePairDataset.__init__(self, phase, transform, random_rotation,
                                               random_scale, manual_seed, config)

        self.max_epoch = config.max_epoch - 1
        self.last_altered_epoch = 0

        if config.extension_steps > 0:  # treat it as the interval
            self.extension_epoch_interval = int(config.max_epoch / config.extension_steps)
        elif config.extension_steps == 0:  # simply alter the dataset at every epoch
            self.extension_epoch_interval = 1

    def update_extension_distance(self, epoch):
        # If the interval has not been reached since the last extension, then do nothing.
        if not (epoch - self.last_altered_epoch >= self.extension_epoch_interval):
            print(f"Checking {self.phase} dataset: No need to update ({epoch}, latest altered at {self.last_altered_epoch}).")
            return False
        
        # if fractional extension step is floored away and the old value remains, then do nothing.
        expected_max_dist = int((self.LAST_DIST - self.FIRST_DIST) * (epoch/self.max_epoch)) + self.FIRST_DIST
        if expected_max_dist == self.MAX_DIST:
            print(f"Checking {self.phase} dataset: No need to update ({epoch}, latest altered at {self.last_altered_epoch}).")
            return False

        # otherwise, update the dataset.
        self.MAX_DIST = expected_max_dist
        self.last_altered_epoch = epoch

        self.files = []
        self.prepare_nuscenes_ply(self.phase)
        print(f"Dataset extension for phase {self.phase}: {self.MAX_DIST=}")
        print(f"Data size for epoch {epoch} has been updated to: {len(self.files)}")
        return self.MAX_DIST

    def is_base_dataset(self):
        return self.MAX_DIST == 1


ALL_DATASETS = [
    ThreeDMatchPairDataset, KITTIPairDataset, 
    KittiNFramePairDataset, NuscenesNFramePairDataset, WaymoNFramePairDataset,
    KittiRandDistPairDataset, NuscenesRandDistPairDataset, WaymoRandDistPairDataset,
    KittiContinuousFramePairDataset, NuscenesContinuousFramePairDataset,
    WaymoContinuousFramePairDataset
]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config, phase, batch_size, num_threads=0, shuffle=None):
    assert phase in ['train', 'trainval', 'val', 'test']
    if shuffle is None:
        shuffle = phase != 'test'

    if config.dataset not in dataset_str_mapping.keys():
        logging.error(f'Dataset {config.dataset}, does not exists in ' +
                      ', '.join(dataset_str_mapping.keys()))

    collate_function = collate_pair_fn

    print(f"Using dataset: {config.dataset}")

    Dataset = dataset_str_mapping[config.dataset]

    use_random_scale = False
    use_random_rotation = False
    transforms = []
    if phase in ['train', 'trainval']:
        use_random_rotation = config.use_random_rotation
        use_random_scale = config.use_random_scale
        transforms += [t.Jitter()]

    dset = Dataset(phase,
                   transform=t.Compose(transforms),
                   random_scale=use_random_scale,
                   random_rotation=use_random_rotation,
                   manual_seed=True,
                   config=config)

    loader = torch.utils.data.DataLoader(dset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=num_threads,
                                         collate_fn=collate_function,
                                         pin_memory=False,
                                         drop_last=True)

    return loader
