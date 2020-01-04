import numpy as np


def analyze(cam_spec, cam_gt, blind, n_joints, thresh):
	'''
	Analyzes tracking performance of a single batch.

	Args:
		cam_spec: (batch, n_joints x 3) <float32>
		cam_gt: (batch, n_joints x 3) <float32>
		blind: (batch,) <bool>
	'''
	cam_gt = cam_gt.reshape(-1, n_joints, 3) * 100.0
	cam_spec = cam_spec.reshape(-1, n_joints, 3) * 100.0

	dist = np.linalg.norm(cam_spec - cam_gt, axis = -1).flatten()

	score_pck = np.mean(dist / thresh <= 1.0)
	score_auc = np.mean(np.maximum(0, 1 - dist / thresh))

	return dict(
		mean = np.mean(dist),
		score_auc = score_auc,
		score_pck = score_pck,
		batch = dist.shape[0]
	)


def parse_epoch(stats):

	keys = ['mean', 'score_auc', 'score_pck', 'batch']

	values = np.array([[patch[key] for patch in stats] for key in keys])

	return dict(
		zip(
			keys[:-1],
			np.sum(values[-1] * values[:-1], axis = 1) / np.sum(values[-1])
		)
	)
