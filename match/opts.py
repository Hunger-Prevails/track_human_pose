import argparse

parser = argparse.ArgumentParser(description='list of train/test options')

# bool options
parser.add_argument('-decimal', action='store_true', help='whether to compute loss using decimeters instead')
parser.add_argument('-tracker', action='store_true', help='whether to employ tracker association and smoothen strategy')
parser.add_argument('-smoothen', action='store_true', help='whether to smoothen established tracks before writing annals to disk')

# required options
parser.add_argument('-data_name', required=True, help='Name of dataset')
parser.add_argument('-feat_path', required=True, help='path to inference features')
parser.add_argument('-partition', required=True, help='name of partition')

# integer options
parser.add_argument('-in_features', default=6, type=int, help='number of features for each joint')
parser.add_argument('-in_frames', default=31, type=int, help='length of tracklet as input of the network')
parser.add_argument('-num_joints', default=19, type=int, help='number of joints in the dataset')
parser.add_argument('-n_cudas', default=1, type=int, help='Number of cuda devices available')
parser.add_argument('-channels', default=1024, type=int, help='Number of convolution channels')
parser.add_argument('-inflation', default=2, type=int, help='inflation rate of dilation among convolutions')
parser.add_argument('-n_blocks', default=3, type=int, help='Number of blocks with skip connection')
parser.add_argument('-kernel', default=3, type=int, help='universal kernel size of the convolutions')
parser.add_argument('-n_matches', default=10, type=int, help='number of joints considered for a hard match')
parser.add_argument('-n_forecast', default=10, type=int, help='maximum number of frames we forecast into future')
parser.add_argument('-range_extrap', default=5, type=int, help='range of frames considered by extrapolation')
parser.add_argument('-range_smooth', default=10, type=int, help='range of frames considered by smoothen operation')
parser.add_argument('-base_index', default=2, type=int, help='index of base joint')
parser.add_argument('-on_frame', default=0, type=int, help='start of the video clip on which we have inferred features')
parser.add_argument('-on_stage', default=0, type=int, help='start of the video clip on which we perform tracking')
parser.add_argument('-duration', default=1200, type=int, help='duration of the video clip on which we perform tracking')

# float options
parser.add_argument('-thresh_rel', default=10.0, type=float, help='dist threshold on root-relative cam coord for a positive association')
parser.add_argument('-thresh_cam', default=20.0, type=float, help='dist threshold on cam coord for a positive association')
parser.add_argument('-thresh_score', default=15.0, type=float, help='threshold for score measurement')
parser.add_argument('-dropout', default=0.25, type=float, help='dropout rate')

# miscellaneous
parser.add_argument('-model', help='Backbone architecture')
parser.add_argument('-model_path', help='path to benchmark model')
parser.add_argument('-cam_name', help='name of camera')

args = parser.parse_args()
