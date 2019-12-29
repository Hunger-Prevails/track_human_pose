import torch
import numpy as np

from builtins import zip as xzip

class Tracklet:

	def __init__(self, atn_det, mat_det, cam_det, start_frame, args):

		self.atn_dets = []
		self.mat_dets = []
		self.cam_dets = []

		self.atn_dets.append(atn_det)
		self.mat_dets.append(mat_det)
		self.cam_dets.append(cam_det)

		self.on_stage = start_frame
		self.off_stage = start_frame + 1

		self.in_frames = args.in_frames
		self.b_forecast = args.n_forecast
		self.n_forecast = args.n_forecast
		self.range_extrap = args.range_extrap

		self.tracker = args.tracker
		self.n_matches = args.n_matches
		self.base_index = args.base_index

		self.thresh_rel = args.thresh_rel
		self.thresh_cam = args.thresh_cam


	def _gather(self):
		'''
		retrieves inferred features from the latest past and wrap them up with a Tensor
		'''
		feat_atn = np.stack(self.atn_dets[- self.in_frames:], axis = -1).reshape(-1, self.in_frames)
		feat_mat = np.stack(self.mat_dets[- self.in_frames:], axis = -1).reshape(-1, self.in_frames)
		feat_cam = np.stack(self.cam_dets[- self.in_frames:], axis = -1).reshape(-1, self.in_frames)

		return torch.as_tensor(np.vstack([feat_atn, feat_mat, feat_cam]))


	def _accepts(self, model, atn_specs, mat_specs, cam_specs):
		'''
		return association score based on accept head

		Args:
			atn_specs: (n_det, num_joints)
			mat_specs: (n_det, num_joints, 2)
			cam_specs: (n_det, num_joints, 3)
		'''
		assert self.in_frames <= len(self)

		n_det = atn_specs.shape[0]

		det_feat = np.hstack([atn_specs.reshape(n_det, -1), mat_specs.reshape(n_det, -1), cam_specs.reshape(n_det, -1)])

		with torch.no_grad():

			assc_score = model(self._gather().repeat(n_det, 1, 1).cuda(), torch.as_tensor(det_feat).cuda())[0]

		return assc_score.cpu().numpy().flatten()


	def _belongs(self, cam_specs):
		'''
		return association score based on distance measure

		Args:
			cam_specs: (n_det, num_joints, 3)
		'''
		cam_extra = self.cam_dets[-1] if len(self) == 1 else self._extrapolate()

		rel_specs = cam_specs - cam_specs[:, self.base_index:self.base_index + 1]  # (n_det, num_joints, 3)
		rel_extra = cam_extra - cam_extra[self.base_index]  # (num_joints, 3)

		dist_rel = np.linalg.norm(rel_specs - rel_extra, axis = -1)  # (n_det, num_joints)
		dist_cam = np.linalg.norm(cam_specs - cam_extra, axis = -1)  # (n_det, num_joints)

		_rel_ = np.argsort(dist_rel, axis = -1)[:, :self.n_matches]  # (n_det, n_matches)
		_cam_ = np.argsort(dist_cam, axis = -1)[:, :self.n_matches]  # (n_det, n_matches)

		dist_rel = np.stack([dist[indices] for dist, indices in xzip(dist_rel, _rel_)])  # (n_det, n_matches)
		dist_cam = np.stack([dist[indices] for dist, indices in xzip(dist_cam, _cam_)])  # (n_det, n_matches)

		mean_rel = np.mean(dist_rel, axis = -1)
		mean_cam = np.mean(dist_cam, axis = -1)

		return 2.0 ** (- mean_rel / self.thresh_rel - 1) + 2.0 ** (- mean_cam / self.thresh_cam - 1)


	def _extrapolate(self):
		'''
		extrapolates camera pose for the next frame from detections on the latest (range_extrap + 1) frames
		'''
		assert 2 <= len(self)

		range_extrap = min(self.range_extrap, len(self) - 1)

		cam_dets = np.stack(self.cam_dets[- range_extrap - 1:])  # (range_extrap + 1, 19, 3)

		cam_extra = 2.0 * cam_dets[-1:] - cam_dets[:-1]  # (range_extrap, 19, 3)

		alphas = np.arange(range_extrap)

		alphas = 1.0 / (range_extrap - alphas).reshape(-1, 1, 1)  # (range_extrap, 1, 1)

		cam_extra = cam_extra * alphas + cam_dets[-1:] * (1.0 - alphas)  # (range_extrap, 19, 3)

		return np.mean(cam_extra, axis = 0)  # (19, 3)


	def associate(self, model, atn_specs, mat_specs, cam_specs):
		'''
		return association score for each of the detections

		Args:
			atn_specs: (n_det, num_joints)
			mat_specs: (n_det, num_joints, 2)
			cam_specs: (n_det, num_joints, 3)
		'''
		return self._accepts(model, atn_specs, mat_specs, cam_specs) if self.in_frames <= len(self) and self.tracker else self._belongs(cam_specs)


	def forecast(self, model):

		assert self.in_frames <= len(self)

		if self.tracker:

			with torch.no_grad():

				next_cam = model(self._gather().unsqueeze(0).cuda())

			next_cam = next_cam.cpu().numpy().reshape(-1, 3)

		else:
			next_cam = self._extrapolate()

		next_mat = next_cam[:, :2] / next_cam[:, 2:]

		next_atn = self.atn_dets[-1].copy()

		self.atn_dets.append(next_atn)
		self.mat_dets.append(next_mat)
		self.cam_dets.append(next_cam)

		self.b_forecast -= 1
		self.off_stage += 1


	def incorporate(self, model, atn_spec, mat_spec, cam_spec):

		self.atn_dets.append(atn_spec)

		if self.in_frames <= len(self) and self.tracker:

			det_feat = np.hstack([atn_spec.flatten(), mat_spec.flatten(), cam_spec.flatten()])

			det_feat = torch.as_tensor(det_feat).unsqueeze(0).cuda()

			with torch.no_grad():

				next_cam = model(self._gather().unsqueeze(0).cuda(), det_feat)[1]

			next_cam = next_cam.cpu().numpy().reshape(-1, 3)

			next_mat = next_cam[:, :2] / next_cam[:, 2:]

			self.mat_dets.append(next_mat)
			self.cam_dets.append(next_cam)
		else:
			self.mat_dets.append(mat_spec)
			self.cam_dets.append(cam_spec)

		self.b_forecast = self.n_forecast
		self.off_stage += 1


	def __len__(self):
		return self.off_stage - self.on_stage
