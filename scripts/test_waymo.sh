#! /bin/bash
export PATH_POSTFIX=$1
export MISC_ARGS=$2

# if you are using the automated procedure, this parameter is overrided
export CUDA_VISIBLE_DEVICES=3

# export DATASET=${DATASET:-NuscenesNFramePairDataset}
export DATASET=${DATASET:-WaymoRandDistPairDataset}
export KITTI_PATH="/mnt/disk/waymo_open_dataset_V2/waymo_kitti_firstReturn"
# export KITTI_PATH="/mnt/disk/waymo_open_dataset_V2/waymo_kitti"
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export VERSION=$(git rev-parse HEAD)
export OUT_DIR=${OUT_DIR:-./outputs/Experiments/WaymoContinuousFramePairDataset-v0.3/ContinuousCorrExtensionTrainer/ResUNetFatBN/SGD-lr5e-3-e200-b8i1-modelnout32/2024-03-17_22-31-05}
export PYTHONUNBUFFERED="True"

echo $OUT_DIR

mkdir -m 755 -p $OUT_DIR

LOG=${OUT_DIR}/log_${TIME}.txt

echo "Host: " $(hostname) | tee -a $LOG
echo "Conda " $(which conda) | tee -a $LOG
echo $(pwd) | tee -a $LOG
echo "Version: " $VERSION | tee -a $LOG
# echo "Git diff" | tee -a $LOG
# echo "" | tee -a $LOG
# git diff | tee -a $LOG
echo "" | tee -a $LOG
nvidia-smi | tee -a $LOG

# # Test
# python -m scripts.test_kitti \
# 	--kitti_root ${KITTI_PATH} \
# 	--dataset ${DATASET} \
# 	--LoWAYMO false \
# 	--rre_thresh 5 \
# 	--rte_thresh 2 \
# 	--pair_min_dist 5 \
# 	--pair_max_dist 10 \
# 	--use_RANSAC true \
# 	--save_dir ${OUT_DIR}
# 	#  | tee -a $LOG


# Automated procedure for all five tests
range_list=(5 10 20 30 40 50)
devices_list=(0 0 1 1 1)
RANSAC=true
LOG_DIR="./ablation/eyoc_waymo/FirstReturn/ema_train/finetune_from_GCL/test_GCL_finetune_30_30_epoch25"

if [ "$RANSAC" = true ] ; then
	REGISTRATOR="ransac"
else
	REGISTRATOR="sc2pcr"
fi

for ((i=0; i<5; i++)); do
    min_dist=${range_list[i]}
    max_dist=${range_list[i+1]}
	device=${devices_list[i]}
	
	export CUDA_VISIBLE_DEVICES=$device

    nohup python -m scripts.test_kitti \
		--kitti_root ${KITTI_PATH} \
		--LoWAYMO false \
		--rre_thresh 5.0 \
		--rte_thresh 2.0 \
		--pair_min_dist $min_dist \
		--pair_max_dist $max_dist \
		--use_RANSAC $RANSAC \
		--dataset $DATASET \
		--save_dir ${OUT_DIR} | tee -a $LOG \
		 > ${LOG_DIR}/test_waymo_${REGISTRATOR}_${min_dist}_${max_dist}.txt &

done