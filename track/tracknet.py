import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialNet(nn.Module):

	def __init__(self, args):

		super(PartialNet, self).__init__()

		self.drop = nn.Dropout(args.dropout)

		self.dilate_convs = []
		self.dilate_bns = []

		self.mask_convs = []

		self.smooth_convs = []
		self.smooth_bns = []

		self.n_blocks = args.n_blocks

		self.dilations = []

		self.expand_conv = nn.Conv1d(args.n_joints * args.in_features, args.channels, 3, bias = False)
		self.expand_bn = nn.BatchNorm1d(args.channels)

		dilation = args.inflation

		for k in xrange(args.n_blocks):

			self.dilations.append(dilation)

			self.dilate_convs.append(nn.Conv1d(args.channels, args.channels, 3, dilation = dilation, bias = False))
			self.dilate_bns.append(nn.BatchNorm1d(args.channels))

			self.smooth_convs.append(nn.Conv1d(args.channels, args.channels, 1, bias = False))
			self.smooth_bns.append(nn.BatchNorm1d(args.channels))

			dilation *= args.inflation

		self.dilate_convs = nn.ModuleList(self.dilate_convs)
		self.dilate_bns = nn.ModuleList(self.dilate_bns)

		self.smooth_convs = nn.ModuleList(self.smooth_convs)
		self.smooth_bns = nn.ModuleList(self.smooth_bns)

		self.a_lin = nn.Linear(args.channels, args.channels)
		self.b_lin = nn.Linear(args.channels, args.n_joints * 3)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)


	def forward(self, x, mask, ones):
		'''
		Args:
			x: (batch, n_joints x in_features, n_frames)
			mask: (batch, 1, n_frames)
			ones: (1, 1, 3)
		Return:
			(batch, n_joints x 3)
		'''
		assert x.size(-1) == sum(self.dilations) * 2 + 3

		batch = x.size(0)

		expand_x = self.expand_conv(x)

		expand_mask = F.conv1d(mask, ones)

		mask = (expand_mask != 0).float()

		multiplier = (1.0 / (expand_mask + 1e-6)) * mask

		expand_x *= multiplier

		x = F.relu(self.expand_bn(expand_x))

		for k in xrange(self.n_blocks):

			res = x[:, :, self.dilations[k] * 2:]

			dilate_x = self.dilate_convs[k](x)

			dilate_mask = F.conv1d(mask, ones, dilation = self.dilations[k])

			mask = (dilate_mask != 0).float()

			multiplier = (1.0 / (dilate_mask + 1e-6)) * mask

			dilate_x *= multiplier

			x = F.relu(self.dilate_bns[k](dilate_x))

			x = F.relu(self.smooth_bns[k](self.smooth_convs[k](x)))

			x += res

		x = x.squeeze(-1)

		x = F.relu(self.a_lin(x))

		return self.b_lin(x)
