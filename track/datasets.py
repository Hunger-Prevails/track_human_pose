import os
import re
import json
import static
import cameralib
import itertools
import matlabfile
import numpy as np
import torch.utils.data as data

from builtins import zip as xzip


def fetch_mpihp_tracks(args, phase):

	selection = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 3, 6, 4]

	camera_ids = [0, 1, 2, 4, 5, 6, 7, 8]

	partitions = [0, 1, 2, 3, 4, 5, 6] if phase == 'train' else [7]

	gt_tracks = []

	for par_id, seq_id in itertools.product(partitions, range(2)):

		seq_path = os.path.join(args.data_root, 'S' + str(par_id + 1), 'Seq' + str(seq_id + 1))

		annotations = matlabfile.load(os.path.join(seq_path, 'annot.mat'))['annot3']

		cam_coords = [anno.reshape([anno.shape[0], -1, 3])[:, selection] for anno in annotations]  # [(n_frames, 17, 3) x n_cams]

		for cam_id in camera_ids:

			gt_tracks.append(cam_coords[cam_id])

	return gt_tracks


def get_data_loader(args, phase):

	gt_tracks = globals()['fetch_' + args.data_name + '_tracks'](args, phase)

	dataset = Dataset(gt_tracks, args, phase)

	return data.DataLoader(dataset, args.batch_size, shuffle = True if phase == 'train' else False, num_workers = args.workers, pin_memory = True)
