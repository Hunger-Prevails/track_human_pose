import numpy as np


def analyze(accept, refine, agnost, true_cam, verdict, key_mask, thresh):
	'''
	Analyzes tracking performance of a single batch.

	Args:
		accept: (batch, 1) <float32>
		refine: (batch, n_joints x 3) <float32>
		agnost: (batch, n_joints x 3) <float32>
		true_cam: (batch, n_joints, 3) <float32>
		verdict: (batch, 1) <bool>
		key_mask: (batch, n_joints) <bool>
	'''
	accept = 0 <= accept

	CT = float(np.sum(accept & verdict) + np.sum(~accept & ~verdict))
	TP = float(np.sum(accept & verdict))
	FP = float(np.sum(accept & ~verdict))
	FN = float(np.sum(~accept & verdict))

	assert CT + FP + FN != 0
	assert TP + FP != 0
	assert TP + FN != 0

	accuracy = CT / (CT + FP + FN)
	precision = TP / (TP + FP)
	recall = TP / (TP + FN)
	f_measure = 2 * precision * recall / (precision + recall)

	refine_mask = (key_mask & verdict).flatten()
	agnost_mask = (key_mask & ~verdict).flatten()

	refine_dist = np.linalg.norm(refine.reshape(-1, 3)[refine_mask] - true_cam.reshape(-1, 3)[refine_mask], axis = -1)
	agnost_dist = np.linalg.norm(agnost.reshape(-1, 3)[agnost_mask] - true_cam.reshape(-1, 3)[agnost_mask], axis = -1)

	refine_dist *= 10.0
	agnost_dist *= 10.0

	return dict(
		accuracy = accuracy,
		precision = precision,
		recall = recall,
		f_measure = f_measure,
		batch = accept.shape[0],
		refine_mean = np.mean(refine_dist),
		refine_auc = np.mean(np.maximum(0, 1 - refine_dist / thresh)),
		refine_pck = np.mean((refine_dist / thresh) <= 1.0),
		refine_count = refine_dist.shape[0],
		agnost_mean = np.mean(agnost_dist),
		agnost_auc = np.mean(np.maximum(0, 1 - agnost_dist / thresh)),
		agnost_pck = np.mean((agnost_dist / thresh) <= 1.0),
		agnost_count = agnost_dist.shape[0]
	)


def parse_epoch(stats):

	keysets = []
	keysets += [('accuracy', 'precision', 'recall', 'f_measure', 'batch')]
	keysets += [('refine_mean', 'refine_auc', 'refine_pck', 'refine_count')]
	keysets += [('agnost_mean', 'agnost_auc', 'agnost_pck', 'agnost_count')]

	all_stats = dict()

	for keys in keysets:

		values = np.array([[patch[key] for patch in stats] for key in keys])

		all_stats.update(
			dict(
				zip(
					keys[:-1],
					np.sum(values[-1] * values[:-1], axis = 1) / np.sum(values[-1])
				)
			)
		)

	return all_stats
