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
import matplotlib.pyplot as plt

from builtins import zip as xzip


np.set_printoptions(precision = 3)


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

	dataset = Lecture(gt_tracks, args) if phase == 'train' else Exam(gt_tracks, args)

	return data.DataLoader(dataset, args.batch_size, shuffle = True if phase == 'train' else False, num_workers = args.workers, pin_memory = True)


class Sample:

	def __init__(self, anchor, track_id):

		self.anchor = anchor
		self.track_id = track_id


class Lecture(data.Dataset):

	def __init__(self, gt_tracks, args):

		self.in_frames = args.in_frames
		self.n_joints = args.n_joints

		self.gt_tracks = gt_tracks

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

		sample[:, :-1] += jitter

		jitter_root_xy = np.random.normal(loc = 0.0, scale = self.sigma_root_xy, size = (self.in_frames, 1, 2))
		jitter_root_zz = np.random.normal(loc = 0.0, scale = self.sigma_root_zz, size = (self.in_frames, 1, 1))

		jitter_root = np.concatenate([jitter_root_xy, jitter_root_zz], axis = -1)

		sample += jitter_root

		mask = np.ones(self.in_frames).astype(np.float32)

		occ_anchor = np.random.randint(low = 0, high = self.in_frames + 1)

		occ_duration = int(np.round(np.random.exponential(scale = self.beta)))

		blind = self.in_frames <= occ_anchor + occ_duration and occ_anchor < self.in_frames

		mask[occ_anchor:occ_anchor + occ_duration] = 0.0

		sample = sample.reshape(self.in_frames, -1).transpose()

		mask = mask.reshape(-1, self.in_frames)

		return sample * mask, mask, cam_gt


	def __getitem__(self, index):
		return self.parse_sample(self.samples[index])


	def __len__(self):
		return len(self.samples)


class Exam(data.Dataset):

	def __init__(self, gt_tracks, args):

		self.in_frames = args.in_frames
		self.n_joints = args.n_joints

		self.samples = []

		self.shift_tracks = []

		root_random = '/globalwork/liu/mpihp_random'

		for track_id, track in enumerate(gt_tracks):

			self.samples += [Sample(frame, track_id) for frame in xrange(0, track.shape[0] - args.in_frames, args.stride)]

		path_occ_anchor = os.path.join(root_random, 'occ_anchor.npy')

		if os.path.exists(path_occ_anchor):

			occ_anchor = np.load(path_occ_anchor)
		else:
			occ_anchor = np.random.randint(low = 0, high = args.in_frames + 1, size = len(self.samples))

			np.save(path_occ_anchor, occ_anchor)

		path_occ_duration = os.path.join(root_random, 'occ_duration.npy')

		if os.path.exists(path_occ_duration):

			occ_duration = np.load(path_occ_duration)
		else:
			occ_duration = np.random.exponential(scale = args.beta, size = len(self.samples)).astype(np.int)

			np.save(path_occ_duration, occ_duration)

		for sample_id, sample in enumerate(self.samples):

			sample.cam_gt = gt_tracks[sample.track_id][sample.anchor + args.in_frames - 1].copy()

			sample.occ_anchor = occ_anchor[sample_id]

			sample.occ_duration = occ_duration[sample_id]

		for track_id, track in enumerate(gt_tracks):

			path_jitter = os.path.join(root_random, 'jitter_' + str(track_id) + '.npy')

			if os.path.exists(path_jitter):

				jitter = np.load(path_jitter)
			else:
				jitter = np.random.normal(loc = 0.0, scale = args.sigma, size = (track.shape[0], args.n_joints - 1, 3))

				np.save(path_jitter, jitter)

			path_jitter_root = os.path.join(root_random, 'jitter_root_' + str(track_id) + '.npy')

			if os.path.exists(path_jitter_root):

				jitter_root = np.load(path_jitter_root)
			else:
				jitter_root_xy = np.random.normal(loc = 0.0, scale = args.sigma_root_xy, size = (track.shape[0], 1, 2))

				jitter_root_zz = np.random.normal(loc = 0.0, scale = args.sigma_root_zz, size = (track.shape[0], 1, 1))

				jitter_root = np.concatenate([jitter_root_xy, jitter_root_zz], axis = -1)

				np.save(path_jitter_root, jitter_root)

			shift_track = track.copy()

			shift_track[:, :-1] += jitter

			shift_track += jitter_root

			self.shift_tracks.append(shift_track)


	def parse_sample(self, sample):

		shift_track = self.shift_tracks[sample.track_id][sample.anchor:sample.anchor + self.in_frames]  # (in_frames, 17, 3)

		cam_gt = sample.cam_gt.flatten()

		mask = np.ones(self.in_frames).astype(np.float32)

		occ_anchor = sample.occ_anchor

		occ_duration = sample.occ_duration

		blind = self.in_frames <= occ_anchor + occ_duration and occ_anchor < self.in_frames

		mask[occ_anchor:occ_anchor + occ_duration] = 0.0

		shift_track = shift_track.reshape(self.in_frames, -1).transpose()

		mask = mask.reshape(-1, self.in_frames)

		return shift_track * mask, mask, cam_gt, np.uint8([blind])


	def __getitem__(self, index):
		return self.parse_sample(self.samples[index])


	def __len__(self):
		return len(self.samples)
