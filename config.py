import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')


logging_arg = add_argument_group('Logging')
logging_arg.add_argument('--out_dir', type=str, default='outputs')
logging_arg.add_argument('--labeler_dir', type=str, default='')
logging_arg.add_argument('--labeler_weight', type=str, default='')
logging_arg.add_argument('--pretraining_dataset', type=str, default='')

trainer_arg = add_argument_group('Trainer')
trainer_arg.add_argument('--trainer', type=str, default='HardestContrastiveLossTrainer')
trainer_arg.add_argument('--save_freq_epoch', type=int, default=1)
trainer_arg.add_argument('--batch_size', type=int, default=4)
trainer_arg.add_argument('--val_batch_size', type=int, default=1)
trainer_arg.add_argument('--extension_steps', type=int, default=10)
trainer_arg.add_argument('--sync_strategy', type=str, default='sync')
trainer_arg.add_argument('--ema_decay', type=float, default=0.99)

trainer_arg.add_argument('--use_sc2_filtering', type=str2bool, default=True)
trainer_arg.add_argument('--feature_filter', type=str, default='Lowe')
trainer_arg.add_argument('--spatial_filter', type=str, default="Spherical")

# Hard negative mining
trainer_arg.add_argument('--use_hard_negative', type=str2bool, default=True)
trainer_arg.add_argument('--hard_negative_sample_ratio', type=int, default=0.05)
trainer_arg.add_argument('--hard_negative_max_num', type=int, default=3000)
trainer_arg.add_argument('--num_pos_per_batch', type=int, default=1024)
trainer_arg.add_argument('--num_hn_samples_per_batch', type=int, default=256)

# Metric learning loss
trainer_arg.add_argument('--neg_thresh', type=float, default=1.4)
trainer_arg.add_argument('--pos_thresh', type=float, default=0.1)
trainer_arg.add_argument('--neg_weight', type=float, default=1)

# labeler config
trainer_arg.add_argument('--use_SC2_PCR', type=str2bool, default=False)

# Data augmentation
trainer_arg.add_argument('--use_random_scale', type=str2bool, default=False)
trainer_arg.add_argument('--min_scale', type=float, default=0.8)
trainer_arg.add_argument('--max_scale', type=float, default=1.2)
trainer_arg.add_argument('--use_random_rotation', type=str2bool, default=True)
trainer_arg.add_argument('--rotation_range', type=float, default=360)

# Data loader configs
trainer_arg.add_argument('--train_phase', type=str, default="train")
trainer_arg.add_argument('--val_phase', type=str, default="val")
trainer_arg.add_argument('--test_phase', type=str, default="test")

trainer_arg.add_argument('--stat_freq', type=int, default=40)
trainer_arg.add_argument('--test_valid', type=str2bool, default=True)
trainer_arg.add_argument('--val_max_iter', type=int, default=400)
trainer_arg.add_argument('--val_epoch_freq', type=int, default=1)
trainer_arg.add_argument(
    '--positive_pair_search_voxel_size_multiplier', type=float, default=1.5)

trainer_arg.add_argument('--hit_ratio_thresh', type=float, default=0.1)
trainer_arg.add_argument('--similarity_thresh', type=float, default=0.4)
trainer_arg.add_argument('--filter_radius', type=float, default=20)
trainer_arg.add_argument('--skip_initialization', type=str2bool, default=False)

# Triplets
trainer_arg.add_argument('--triplet_num_pos', type=int, default=256)
trainer_arg.add_argument('--triplet_num_hn', type=int, default=512)
trainer_arg.add_argument('--triplet_num_rand', type=int, default=1024)

# dNetwork specific configurations
net_arg = add_argument_group('Network')
net_arg.add_argument('--model', type=str, default='ResUNetBN2C')
net_arg.add_argument('--model_n_out', type=int, default=32, help='Feature dimension')
net_arg.add_argument('--conv1_kernel_size', type=int, default=5)
net_arg.add_argument('--normalize_feature', type=str2bool, default=True)
net_arg.add_argument('--dist_type', type=str, default='L2')
net_arg.add_argument('--best_val_metric', type=str, default='feat_match_ratio')

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD')
opt_arg.add_argument('--max_epoch', type=int, default=100)
opt_arg.add_argument('--lr', type=float, default=1e-1)
opt_arg.add_argument('--momentum', type=float, default=0.8)
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1)
opt_arg.add_argument('--adam_beta1', type=float, default=0.9)
opt_arg.add_argument('--adam_beta2', type=float, default=0.999)
# opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
opt_arg.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.05)
opt_arg.add_argument('--exp_gamma', type=float, default=0.99)
opt_arg.add_argument('--scheduler', type=str, default='ExpLR')
opt_arg.add_argument('--finetune_restart', type=str2bool, default=False)


misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--weights', type=str, default=None)
misc_arg.add_argument('--resume', type=str, default=None)
misc_arg.add_argument('--resume_dir', type=str, default=None)
misc_arg.add_argument('--train_num_thread', type=int, default=8)
misc_arg.add_argument('--val_num_thread', type=int, default=2)
misc_arg.add_argument('--test_num_thread', type=int, default=2)
misc_arg.add_argument(
    '--nn_max_n',
    type=int,
    default=500,
    help='The maximum number of features to find nearest neighbors in batch')

# Dataset specific configurations
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='ThreeDMatchPairDataset')
data_arg.add_argument('--voxel_size', type=float, default=0.025)
data_arg.add_argument(
    '--threed_match_dir', type=str, default="/home/chrischoy/datasets/FCGF/threedmatch")
data_arg.add_argument(
    '--kitti_root', type=str, default="/home/chrischoy/datasets/FCGF/kitti/")
data_arg.add_argument(
    '--kitti_max_time_diff',
    type=int,
    default=3,
    help='max time difference between pairs (non inclusive)')
data_arg.add_argument('--kitti_date', type=str, default='2011_09_26')

data_arg.add_argument('--pair_min_dist', type=int, default=-1)
data_arg.add_argument('--pair_max_dist', type=int, default=-1)
data_arg.add_argument('--LoKITTI', type=str2bool, default=False)
data_arg.add_argument('--supervised', type=str2bool, default=False) # set to True to enable supervised training on continuous datasets.
data_arg.add_argument('--percentage', type=float, default=1.0) # percentage of the dataset to use. This is used as an online-learning test.

# composition dataset setting
data_arg.add_argument('--use_kitti', type=str2bool, default=False)
data_arg.add_argument('--use_nuscenes', type=str2bool, default=False)
data_arg.add_argument('--use_waymo', type=str2bool, default=False)

def get_config():
  args = parser.parse_args()
  return args
