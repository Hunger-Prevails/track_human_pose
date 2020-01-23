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

	def __init__(self, anchor, i_track):

		self.anchor = anchor
		self.i_track = i_track


class Lecture(data.Dataset):

	def __init__(self, gt_tracks, args):

		self.in_frames = args.in_frames
		self.n_joints = args.n_joints

		self.gt_tracks = gt_tracks

		self.sigma = args.sigma
		self.zeta = args.zeta
		self.sigma_root_xy = args.sigma_root_xy
		self.sigma_root_zz = args.sigma_root_zz
		self.beta = args.beta

		self.samples = []

		for i_track, track in enumerate(gt_tracks):

			self.samples += [Sample(frame, i_track) for frame in xrange(0, track.shape[0] - args.in_frames, args.stride)]

		self.samples = [sample for sample in self.samples if self.is_dynamic(sample, gt_tracks)]

		landmarks = np.linspace(-1.0, 1.0, args.in_frames).reshape(-1, 1)

		self.kernel = landmarks ** 2 + landmarks.T ** 2 - 2 * np.dot(landmarks, landmarks.T)

		self.kernel = np.exp(- 0.5 / args.zeta * self.kernel)


	def is_dynamic(self, sample, gt_tracks):

		gt_track = gt_tracks[sample.i_track][sample.anchor:sample.anchor + self.in_frames]  # (in_frames, 17, 3)

		displace = np.linalg.norm(gt_track[1:] - gt_track[:-1], axis = -1)  # (in_frames - 1, 17)

		return np.mean(np.max(displace, axis = -1) > 0.01) > 0.2


	def parse_sample(self, sample):

		gt_track = self.gt_tracks[sample.i_track][sample.anchor:sample.anchor + self.in_frames]  # (in_frames, 17, 3)

		cam_gt = gt_track[(self.in_frames - 1) / 2]

		track = gt_track.copy()

		jitter = np.random.normal(loc = 0.0, scale = self.sigma, size = (self.in_frames, self.n_joints - 1, 3))

		jitter_root_xy = np.random.normal(loc = 0.0, scale = self.sigma_root_xy, size = (self.in_frames, 1, 2))
		jitter_root_zz = np.random.normal(loc = 0.0, scale = self.sigma_root_zz, size = (self.in_frames, 1, 1))

		jitter_root = np.concatenate([jitter_root_xy, jitter_root_zz], axis = -1)

		# track[:, :-1] += np.einsum('ai,ibc->abc', self.kernel, jitter)

		# track += np.einsum('ai,ibc->abc', self.kernel, jitter_root)
		'''
		from plot import show_cam

		import matplotlib.pyplot as plt

		plt.figure(figsize = (16, 12))
		ax = plt.subplot(1, 1, 1, projection = '3d')

		show_cam(track[2::6].transpose(0, 2, 1) * 100.0, ax, color = np.array([0.0, 0.0, 1.0]))
		show_cam(gt_track[2::6].transpose(0, 2, 1) * 100.0, ax, color = np.array([0.0, 1.0, 0.0]))

		plt.show()
		'''
		mask = np.ones(self.in_frames).astype(np.float32)
		# if np.random.random() < 0.5:

		# 	span_pool = np.arange(11)

		# 	chances = 1.0 / (span_pool + 1)

		# 	occ_span = np.random.choice(span_pool, p = chances / chances.sum())

		# 	mask[(self.in_frames - 1) / 2 - occ_span:(self.in_frames - 1) / 2 + occ_span + 1] = 0.0

		track = track.transpose(1, 2, 0).reshape(-1, self.in_frames)

		mask = mask.reshape(-1, self.in_frames)

		track = track * mask

		return track, mask, cam_gt


	def __getitem__(self, index):
		return self.parse_sample(self.samples[index])


	def __len__(self):
		return len(self.samples)


class Exam(data.Dataset):

	def __init__(self, gt_tracks, args):

		self.occ_span = args.occ_span
		self.in_frames = args.in_frames
		self.n_joints = args.n_joints

		self.samples = []

		self.tracks = []

		root_random = '/globalwork/liu/mpihp_random'

		for i_track, track in enumerate(gt_tracks):

			self.samples += [Sample(frame, i_track) for frame in xrange(0, track.shape[0] - args.in_frames, args.stride)]

		for sample in self.samples:

			sample.cam_gt = gt_tracks[sample.i_track][sample.anchor + (args.in_frames - 1) / 2]

		for i_track, track in enumerate(gt_tracks):

			track = track.copy()

			path_jitter = os.path.join(root_random, 'jitter_' + str(i_track) + '.npy')

			if os.path.exists(path_jitter):

				jitter = np.load(path_jitter)
			else:
				jitter = np.random.normal(loc = 0.0, scale = args.sigma, size = (track.shape[0], args.n_joints - 1, 3))

				np.save(path_jitter, jitter)

			path_jitter_root = os.path.join(root_random, 'jitter_root_' + str(i_track) + '.npy')

			if os.path.exists(path_jitter_root):

				jitter_root = np.load(path_jitter_root)
			else:
				jitter_root_xy = np.random.normal(loc = 0.0, scale = args.sigma_root_xy, size = (track.shape[0], 1, 2))

				jitter_root_zz = np.random.normal(loc = 0.0, scale = args.sigma_root_zz, size = (track.shape[0], 1, 1))

				jitter_root = np.concatenate([jitter_root_xy, jitter_root_zz], axis = -1)

				np.save(path_jitter_root, jitter_root)

			landmarks = np.linspace(- track.shape[0] / args.in_frames, track.shape[0] / args.in_frames, track.shape[0]).reshape(-1, 1)

			kernel = landmarks ** 2 + landmarks.T ** 2 - 2 * np.dot(landmarks, landmarks.T)

			kernel = np.exp(- 0.5 / args.zeta * kernel)

			# track[:, :-1] += np.einsum('ai,ibc->abc', kernel, jitter)

			# track += np.einsum('ai,ibc->abc', kernel, jitter_root)

			self.tracks.append(track)


	def parse_sample(self, sample):

		track = self.tracks[sample.i_track][sample.anchor:sample.anchor + self.in_frames]  # (in_frames, 17, 3)

		mask = np.ones(self.in_frames).astype(np.float32)

		# mask[(self.in_frames - 1) / 2 - self.occ_span:(self.in_frames - 1) / 2 + self.occ_span + 1] = 0.0

		track = track.transpose(1, 2, 0).reshape(-1, self.in_frames)

		mask = mask.reshape(-1, self.in_frames)

		track = track * mask

		return track, mask, sample.cam_gt


	def __getitem__(self, index):
		return self.parse_sample(self.samples[index])


	def __len__(self):
		return len(self.samples)
