import os
import re
import json
import h5py
import static
import cameralib
import itertools
import matlabfile
import numpy as np
import torch.utils.data as data

from builtins import zip as xzip

np.set_printoptions(precision = 3)

class Sample:

	def __init__(self, anchor, track_id):

		self.anchor = anchor
		self.track_id = track_id


def fetch_mpihp_tracks(args, phase):

	gt_tracks = []

	if phase == 'train':

		selection = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 3, 6, 4]

		camera_ids = [0, 1, 2, 4, 5, 6, 7, 8]

		for par_id, seq_id in itertools.product(xrange(8), xrange(2)):

			seq_path = os.path.join(args.data_root, 'S' + str(par_id + 1), 'Seq' + str(seq_id + 1))

			annotations = matlabfile.load(os.path.join(seq_path, 'annot.mat'))['annot3']

			cam_coords = [anno.reshape([anno.shape[0], -1, 3])[:, selection] for anno in annotations]  # [(n_frames, 17, 3) x n_cams]

			for cam_id in camera_ids:

				gt_tracks.append((cam_coords[cam_id] / 1000.0).astype(np.float32))
	else:
		selection = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 14]

		for par_id in xrange(6):

			seq_path = os.path.join(args.data_root, 'TS' + str(par_id + 1))

			anno_path = os.path.join(seq_path, 'annot_data-corr.mat') if par_id == 5 else os.path.join(seq_path, 'annot_data.mat')

			with h5py.File(anno_path, 'r') as anno:

				cam_coords = np.asarray(anno['annot3'])[:, 0, selection]

				gt_tracks.append((cam_coords / 1000.0).astype(np.float32))

				anno.close()

	return gt_tracks


def get_data_loader(args, phase):

	gt_tracks = globals()['fetch_' + args.data_name + '_tracks'](args, phase)

	dataset = Dataset(gt_tracks, args, phase)

	return data.DataLoader(dataset, args.batch_size, shuffle = True if phase == 'train' else False, num_workers = args.workers, pin_memory = True)


class Dataset(data.Dataset):

	def __init__(self, gt_tracks, args, phase):

		self.in_frames = args.in_frames
		self.n_joints = args.n_joints

		self.gt_tracks = gt_tracks

		self.on_test = phase != 'train'

		self.sigma = args.sigma

		self.sigma_root_xy = args.sigma_root_xy
		self.sigma_root_zz = args.sigma_root_zz

		self.beta = args.beta

		self.samples = []

		for track_id, track in enumerate(gt_tracks):

			self.samples += [Sample(frame, track_id) for frame in xrange(0, track.shape[0] - args.in_frames, args.stride)]

	def parse_sample(self, sample):

		sample = self.gt_tracks[sample.track_id][sample.anchor:sample.anchor + self.in_frames]  # (in_frames, 17, 3)

		cam_gt = sample[-1].flatten()

		jitter = np.random.normal(loc = 0.0, scale = self.sigma, size = (self.in_frames, self.n_joints - 1, 3))

		jitter_root_xy = np.random.normal(loc = 0.0, scale = self.sigma_root_xy, size = (self.in_frames, 1, 2))
		jitter_root_zz = np.random.normal(loc = 0.0, scale = self.sigma_root_zz, size = (self.in_frames, 1, 1))

		jitter_root = np.concatenate([jitter_root_xy, jitter_root_zz], axis = -1)

		# sample[:, :-1] += jitter
		# sample += jitter_root

		mask = np.ones(self.in_frames).astype(np.float32)

		occ_anchor = np.random.randint(low = 0, high = self.in_frames + 1)

		occ_duration = int(np.round(np.random.exponential(scale = self.beta)))

		blind = self.in_frames <= occ_anchor + occ_duration and occ_anchor < self.in_frames

		# mask[occ_anchor:occ_anchor + occ_duration] = 0.0

		sample = sample.reshape(self.in_frames, -1).transpose()

		mask = mask.reshape(-1, self.in_frames)

		if self.on_test:
			return sample * mask, mask, cam_gt, np.uint8([blind])
		else:
			return sample * mask, mask, cam_gt

	def __getitem__(self, index):
		return self.parse_sample(self.samples[index])


	def __len__(self):
		return len(self.samples)
