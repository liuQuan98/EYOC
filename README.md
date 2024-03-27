# Extend Your Own Correspondences: Unsupervised Distant Point Cloud Registration by Progressive Distance Extension (CVPR 2024)

Registration of point clouds collected from a pair of distant vehicles provides a comprehensive and accurate 3D view of the driving scenario, which is vital for driving safety related applications, yet existing literature suffers from the expensive pose label acquisition and the deficiency to generalize to new data distributions. In this paper, we propose EYOC, an unsupervised distant point cloud registration method that adapts to new point cloud distributions on the fly, requiring no global pose labels. The core idea of EYOC is to train a feature extractor in a progressive fashion, where in each round, the feature extractor, trained with near point cloud pairs, can label slightly farther point cloud pairs, enabling self-supervision on such far point cloud pairs. This process continues until the derived extractor can be used to register distant point clouds. Particularly, to enable high-fidelity correspondence label generation, we devise an effective spatial filtering scheme to select the most representative correspondences to register a point cloud pair, and then utilize the aligned point clouds to discover more correct correspondences. Experiments show that EYOC can achieve comparable performance with state-of-the-art supervised methods at a lower training cost. Moreover, it outwits supervised methods regarding generalization performance on new data distributions.

Paper links: [Arxiv preprint](https://arxiv.org/abs/2403.03532)

## News

20240327 - Source code is released. We also checked and fixed a bug in WOD preprocessing--Remember to check our arxiv v2 for the updated results!

20240227 - Our paper has been accepted by CVPR'24!

## Overview of Group-wise Contrastive Learning (GCL)

<div align="center">
<img src=assets\arch.png>
</div>

## Requirements

- Ubuntu 14.04 or higher

- Ubuntu 14.04 or higher
- CUDA 11.1 or higher
- Python v3.7 or higher
- Pytorch v1.6 or higher
- [MinkowskiEngine](https://github.com/stanfordvl/MinkowskiEngine) v0.5 or higher

## Dataset Preparation

### KITTI & nuScenes

Please check [GCL](https://github.com/liuQuan98/GCL#dataset-preparation) for detailed preparation instructions.

### WOD

Please first download the LiDAR sequences and pose files of [WOD](https://www.waymo.com/open).

Then, convert the WOD dataset to kitti format using the following command:

```
python ./assets/convert_wod_to_kitti.py waymo_construct_kitti_PCR_data
```

Remember to alter the `root` and `waymo_kitti_dir` to your specific paths.

## Installation

We recommend conda for installation. First, we need to create a basic environment to setup MinkowskiEngine:

```
conda create -n eyoc python=3.7 pip=21.1
conda activate eyoc
conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia
pip install numpy
```

Then install [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) along with other dependencies:

```
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
pip install -r requirements.txt
```

### Setting the distance between two LiDARs (registration difficulty)

1. **Supervised training:**

We denote range of pairwise translation length $d$ with the parameter `--pair_min_dist` and `--pair_max_dist`, which can be found in `./scripts/train_fcgf_{$dataset}.sh`. For example, setting

```
--pair_min_dist 5 \
--pair_max_dist 20 \
```

will set $d\in [5m,20m]$. In other words, for every pair of point clouds, the ground-truth euclidean distance betwen two corresponding LiDAR positions obeys a uniform distribution between 5m and 20m.

2) **Unsupervised training of EYOC:**

For all EYOC scripts (`./scripts/train_fcgf_{$dataset}_continuous.sh`), setting

```
--pair_min_dist 1 \
--pair_max_dist 30 \
```

will set frame interval $I\in [1,1]$ which linearly grows to $I\in [1,30]$ during the course of training. In other words, for every pair of point clouds, the frame interval betwen two corresponding LiDAR frames obeys a uniform distribution in the current range $I\in [1,B]$.

### Launch the training

To train EYOC, run the following command inside conda environment `eyoc`:

```
./scripts/train_{$DATASET}_EYOC.sh
```

For example, kitti training can be initiated by running:

```
./scripts/train_kitti_EYOC.sh
```

The baseline method FCGF and FCGF+C can be trained similarly with our dataset. FCGF needs training twice while FCGF+C completes training only once and for all.

```
./scripts/train_{$DATASET}.sh
./scripts/train_{$DATASET}_FCGF+C.sh
```

### Testing

You can choose to use SC2-PCR to speedup the inference, by setting `use_RANSAC` to `false`. Otherwise, you can set `use_RANSAC` to `true` for fair comparison results. Do not forget to set  `OUT_DIR` to the specific model path and set  `LOG_DIR` to your preferred output directory.

All five test splits will be tested in parallel. If CUDA OOM occurs, please alter the `devices_list` according to your hardware specifications.

```
./scripts/test_{$DATASET}.sh
```

## Acknowlegdements

We thank [FCGF](https://github.com/chrischoy/FCGF) for the wonderful baseline, [SC2-PCR](https://github.com/ZhiChen902/SC2-PCR) for a powerful and fast alternative registration algorithm.

We would also like to thank [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) and [waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset) for the convenient dataset conversion codes.
