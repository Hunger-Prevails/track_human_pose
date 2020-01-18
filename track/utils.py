import numpy as np


def analyze(cam_spec, cam_gt, blind, thresh):
	'''
	Analyzes tracking performance of a single batch.

	Args:
		cam_spec: (batch, n_joints x 3) <float32>
		cam_gt: (batch, n_joints x 3) <float32>
		blind: (batch,) <bool>
	'''
	batch = cam_gt.shape[0]

	cam_gt = cam_gt.reshape(batch, -1, 3) * 100.0
	cam_spec = cam_spec.reshape(batch, -1, 3) * 100.0

	rootrel_gt = cam_gt[:, :-1] - cam_gt[:, -1:]  # (batch, n_joints - 1, 3)
	rootrel_spec = cam_spec[:, :-1] - cam_spec[:, -1:]  # (batch, n_joints - 1, 3)

	root_dist = np.linalg.norm(cam_gt[:, -1] - cam_spec[:, -1], axis = -1)  # (batch,)

	rootrel_dist = np.linalg.norm(rootrel_gt - rootrel_spec, axis = -1)  # (batch, n_joints - 1)

	score_pck = np.mean(rootrel_dist / thresh <= 1.0)
	score_auc = np.mean(np.maximum(0, 1 - rootrel_dist / thresh))

	return dict(
		root = np.mean(root_dist),
		mean = np.mean(rootrel_dist),
		score_auc = score_auc,
		score_pck = score_pck,
		batch_size = batch
	)


def parse_epoch(stats):

	keys = ['root', 'mean', 'score_auc', 'score_pck', 'batch_size']

	values = np.array([[patch[key] for patch in stats] for key in keys])

	return dict(
		zip(
			keys[:-1],
			np.sum(values[-1] * values[:-1], axis = 1) / np.sum(values[-1])
		)
	)
