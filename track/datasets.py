import os
import json
import static
import cameralib
import numpy as np
import torch.utils.data as data

from utils import TrackSample
from builtins import zip as xzip


np.seterr(all='warn')


def get_cameras(json_file, cam_names):

	calibration = json.load(open(json_file))

	cameras = [cam for cam in calibration['cameras'] if cam['panel'] == 0]

	return dict(
			[
				(
					cam['name'],
					cameralib.Camera(
							np.matmul(np.array(cam['R']).T, - np.array(cam['t'])),
							np.array(cam['R']),
							np.array(cam['K']),
							np.array(cam['distCoef']),
							(0, -1, 0)
					)
				) for cam in cameras if cam['name'] in cam_names
			]
		)


def is_near_entry(cam_gt, entry_center, args):

	valid = args.thresh_mask <= cam_gt[:, 3]

	if np.sum(valid) < args.num_valid:
		return True

	mass_center = np.array(
		[
			np.mean(cam_gt[valid, 0]),
			np.mean(cam_gt[valid, 2])
		]
	)
	return np.linalg.norm(mass_center - entry_center, axis = -1) <= 80


def compare(cam_gt, cam_spec, args):
	valid = args.thresh_mask <= cam_gt[:, 3]

	cam_gt = cam_gt[:, :3]

	rel_gt = cam_gt - cam_gt[args.base_index]  # (num_joints, 3)

	rel_spec = cam_spec - cam_spec[args.base_index]  # (num_joints, 3)

	dist_rel = np.linalg.norm(rel_gt[valid] - rel_spec[valid], axis = -1)
	dist_cam = np.linalg.norm(cam_gt[valid] - cam_spec[valid], axis = -1)

	_rel_ = np.argsort(dist_rel, axis = -1)[:args.num_valid]  # (args.num_valid,)
	_cam_ = np.argsort(dist_cam, axis = -1)[:args.num_valid]  # (args.num_valid,)

	cond_a = np.mean(dist_rel[_rel_]) <= args.thresh_rel
	cond_b = np.mean(dist_cam[_cam_]) <= args.thresh_cam

	return cond_a, cond_b


def associate(cam_gt, cam_spec, args):
	'''
	Associates a ground truth cam pose to one of the predictions of the current frame

	Args:
		cam_gt: (num_joints, 4)
		cam_spec: (n_det, num_joints, 3)
	Ret:
		int: associate index or (-1) if association fails
	'''
	if cam_spec is None:
		return -1

	valid = args.thresh_mask <= cam_gt[:, 3]

	cam_gt = cam_gt[:, :3]

	rel_gt = cam_gt - cam_gt[args.base_index]  # (num_joints, 3)

	rel_spec = cam_spec - cam_spec[:, args.base_index:args.base_index + 1]  # (n_det, num_joints, 3)

	dist_rel = np.linalg.norm(rel_gt[valid] - rel_spec[:, valid], axis = -1)  # (n_det, k)
	dist_cam = np.linalg.norm(cam_gt[valid] - cam_spec[:, valid], axis = -1)  # (n_det, k)

	_rel_ = np.argsort(dist_rel, axis = -1)[:, :args.num_valid]  # (n_det, args.num_valid)
	_cam_ = np.argsort(dist_cam, axis = -1)[:, :args.num_valid]  # (n_det, args.num_valid)

	dist_rel = np.stack([dist[indices] for dist, indices in xzip(dist_rel, _rel_)])  # (n_det, args.num_valid)
	dist_cam = np.stack([dist[indices] for dist, indices in xzip(dist_cam, _cam_)])  # (n_det, args.num_valid)

	best_fit = np.argmin(np.sum(dist_rel + dist_cam, axis = -1))

	cond_a = np.mean(dist_rel[best_fit]) <= args.thresh_rel
	cond_b = np.mean(dist_cam[best_fit]) <= args.thresh_cam

	return (cond_a & cond_b) * best_fit + (cond_a & cond_b) - 1


def fetch_samples(args, phase):
	sequences = dict(
		train = [
			'170221_haggling_b1',
			'170221_haggling_b2',
			'170221_haggling_b3',
			'170221_haggling_m1',
			'170221_haggling_m2',
			'170221_haggling_m3',
			'170224_haggling_a2',
			'170224_haggling_a3',
			'170224_haggling_b1',
			'170224_haggling_b2',
			'170224_haggling_b3',
			'170228_haggling_a1',
			'170228_haggling_a2',
			'170228_haggling_a3',
			'170228_haggling_b1',
			'170228_haggling_b2',
			'170228_haggling_b3',
			'170404_haggling_a1',
			'170404_haggling_a2',
			'170404_haggling_a3',
			'170404_haggling_b1',
			'170404_haggling_b2',
			'170404_haggling_b3',
			'170407_haggling_a1',
			'170407_haggling_a2',
			'170407_haggling_a3',
			'170407_haggling_b1',
			'170407_haggling_b2',
			'170407_haggling_b3'
		],
		test = [
			'160224_haggling1',
			'160226_haggling1'
		]
	)
	all_spec_atns = dict()
	all_spec_mats = dict()
	all_spec_cams = dict()

	samples = []

	for sequence in sequences[phase]:

		print '=> entering sequence:', sequence

		root_seq = os.path.join(static.root_anno, sequence)

		root_anno = os.path.join(root_seq, 'hdPose3d_stage1_coco19')

		root_image = os.path.join(root_seq, 'hdImgs')

		cam_folders = [os.path.join(root_image, folder) for folder in os.listdir(root_image)]
		cam_folders = [folder for folder in cam_folders if os.path.isdir(folder)]
		cam_folders.sort()

		cam_names = [os.path.basename(folder) for folder in cam_folders]

		root_feat = os.path.join(static.root_feat, sequence)

		spec_atns = dict()
		spec_mats = dict()
		spec_cams = dict()

		on_frame = static.on_frames[sequence]
		ab_frame = static.ab_frames[sequence]

		print '=> reading features'

		for cam_name in cam_names:

			print '=> => reading features from camera:', cam_name

			spec_atn = os.path.join(root_feat, 'expand_atn_' + cam_name + '.json')
			spec_atn = json.load(open(spec_atn))

			assert on_frame == spec_atn['start_frame']
			assert ab_frame == spec_atn['end_frame']

			spec_atns[cam_name] = [None if value is None else np.float32(value) for value in spec_atn['atn']]

			spec_mat = os.path.join(root_feat, 'expand_mat_' + cam_name + '.json')
			spec_mat = json.load(open(spec_mat))

			assert on_frame == spec_mat['start_frame']
			assert ab_frame == spec_mat['end_frame']

			spec_mats[cam_name] = [None if value is None else np.float32(value) for value in spec_mat['mat']]

			spec_cam = os.path.join(root_feat, 'expand_cam_' + cam_name + '.json')
			spec_cam = json.load(open(spec_cam))

			assert on_frame == spec_cam['start_frame']
			assert ab_frame == spec_cam['end_frame']

			spec_cams[cam_name] = [None if value is None else np.float32(value) for value in spec_cam['cam']]

		all_spec_atns[sequence] = spec_atns
		all_spec_mats[sequence] = spec_mats
		all_spec_cams[sequence] = spec_cams

		print '=> features are ready'

		cameras = get_cameras(os.path.join(root_seq, 'calibration_' + sequence + '.json'), cam_names)

		entry_center = np.array([30.35, -254.3])

		body_poses = dict()
		near_entry = dict()
		assc_index = dict()

		print '=> associating instances'

		for frame in xrange(start_frame, end_frame):

			if frame % 100 == 0:
				print '=> => frame [', start_frame, '-', frame, '|', end_frame, ']'

			bodies = os.path.join(root_anno, 'body3DScene_' + str(frame).zfill(8) + '.json')
			bodies = json.load(open(bodies))['bodies']

			bodies = dict([(skeleton['id'], np.array(skeleton['joints19']).reshape(-1, 4)) for skeleton in bodies])

			for body_id in bodies:

				if body_id not in body_poses:

					body_poses[body_id] = dict()
					near_entry[body_id] = dict()
					assc_index[body_id] = dict()

				near_entry[body_id][frame] = is_near_entry(bodies[body_id], entry_center, args)

				if not near_entry[body_id][frame]:

					body_poses[body_id][frame] = dict()
					assc_index[body_id][frame] = dict()

					confid = bodies[body_id][:, 3:]

					for cam_name in cam_names:

						cam_gt = cameras[cam_name].world_to_camera(bodies[body_id][:, :3])

						cam_gt = np.hstack([cam_gt, confid]).astype(np.float32)

						body_poses[body_id][frame][cam_name] = cam_gt

						_spec_cams = spec_cams[cam_name][frame - start_frame]

						assc_index[body_id][frame][cam_name] = associate(cam_gt, _spec_cams, args)

		print '=> association done'

		super_hard_negatives = 0

		print '=> assigning tracks'

		n_persons = len(body_poses)

		for idx, body_id in enumerate(body_poses):

			print '=> => assigning tracks to person [', str(idx) + '/' + str(n_persons), ']'

			occur_frames = body_poses[body_id].keys()
			occur_frames.sort()

			emerge_frame = occur_frames[0]
			vanish_frame = occur_frames[-1]

			for anchor_frame in xrange(emerge_frame, vanish_frame, args.stride):

				window = range(anchor_frame, anchor_frame + args.in_frames)

				occur_flag = [frame in occur_frames for frame in window]

				if not np.all(occur_flag):
					continue

				entry_flag = [near_entry[body_id][frame] for frame in window]

				if sum(entry_flag) != 0:
					continue

				for cam_name in cam_names:
					assc_flag = [assc_index[body_id][frame][cam_name] != -1 for frame in window]

					if not np.all(assc_flag):
						continue

					cam_gt = body_poses[body_id][window[-1]][cam_name]

					_assc_index = [assc_index[body_id][frame][cam_name] for frame in window]

					anchor = anchor_frame - start_frame

					samples.append(TrackSample(sequence, cam_name, anchor, cam_gt, _assc_index))

	print '=> tracks are ready'

	print '=> n_positive:', len(samples)

	return samples, all_spec_atns, all_spec_mats, all_spec_cams


def get_data_loader(args, phase):
	samples, spec_atns, spec_mats, spec_cams = fetch_samples(args, phase)

	dataset = Dataset(samples, spec_atns, spec_mats, spec_cams, args)

	if args.balance and phase == 'train':
		n_samples = dataset.n_positives + dataset.n_negatives

		weight_positives = [float(n_samples) / dataset.n_positives] * dataset.n_positives
		weight_negatives = [float(n_samples) / dataset.n_negatives] * dataset.n_negatives

		sampler = data.WeightedRandomSampler(weight_positives + weight_negatives, n_samples)
	else:
		sampler = data.RandomSampler(dataset)

	return data.DataLoader(dataset, args.batch_size, sampler = sampler, num_workers = args.workers, pin_memory = True)


class Dataset(data.Dataset):

	def __init__(self, samples, spec_atns, spec_mats, spec_cams, args):

		self.samples = samples

		self.spec_atns = spec_atns
		self.spec_mats = spec_mats
		self.spec_cams = spec_cams

		self.in_frames = args.in_frames
		self.num_joints = args.num_joints
		self.thresh_mask = args.thresh_mask


	def parse_sample(self, sample):

		spec_atns = self.spec_atns[sample.sequence][sample.cam_name]
		spec_mats = self.spec_mats[sample.sequence][sample.cam_name]
		spec_cams = self.spec_cams[sample.sequence][sample.cam_name]

		window = range(sample.anchor, sample.anchor + self.in_frames)

		assert len(window) == len(sample.assc_index)

		assc_atns = [spec_atns[anchor][index] for anchor, index in xzip(window, sample.assc_index)]
		assc_mats = [spec_mats[anchor][index] for anchor, index in xzip(window, sample.assc_index)]
		assc_cams = [spec_cams[anchor][index] for anchor, index in xzip(window, sample.assc_index)]

		assc_atns = np.stack(assc_atns, axis = -1).reshape(-1, self.in_frames)
		assc_mats = np.stack(assc_mats, axis = -1).reshape(-1, self.in_frames)
		assc_cams = np.stack(assc_cams, axis = -1).reshape(-1, self.in_frames) / 10.0

		track_feat = np.vstack([assc_atns, assc_mats, assc_cams])

		return track_feat[:, :-1], track_feat[:, -1], sample.cam_gt[:, :3] / 10.0, np.uint8(self.thresh_mask <= sample.cam_gt[:, 3])


	def __getitem__(self, index):
		return self.parse_sample(self.samples[index])


	def __len__(self):
		return len(self.samples)
