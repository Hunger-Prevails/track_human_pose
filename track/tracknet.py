import torch
import torch.nn as nn
import torch.nn.functional as F


class StrideNet(nn.Module):

	def __init__(self, args):

		super(StrideNet, self).__init__()

		self.stride_convs = []
		self.stride_bns = []

		self.smooth_convs = []
		self.smooth_bns = []

		self.n_blocks = args.n_blocks

		self.inflation = args.inflation

		self.expand_conv = nn.Conv1d(args.n_joints * args.in_features, args.channels, 3, stride = args.inflation, bias = False)
		self.expand_bn = nn.BatchNorm1d(args.channels)

		for k in xrange(args.n_blocks):

			self.stride_convs.append(nn.Conv1d(args.channels, args.channels, 3, stride = args.inflation, bias = False))
			self.stride_bns.append(nn.BatchNorm1d(args.channels))

			self.smooth_convs.append(nn.Conv1d(args.channels, args.channels, 1, bias = False))
			self.smooth_bns.append(nn.BatchNorm1d(args.channels))

		self.stride_convs = nn.ModuleList(self.stride_convs)
		self.stride_bns = nn.ModuleList(self.stride_bns)

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
		expand_x = self.expand_conv(x)

		expand_mask = F.conv1d(mask, ones, stride = self.inflation)

		mask = (expand_mask != 0).float()

		multiplier = (1.0 / (expand_mask + 1e-6)) * mask

		expand_x *= multiplier

		x = F.relu(self.expand_bn(expand_x))

		for k in xrange(self.n_blocks):

			res = x[:, :, 2::self.inflation]

			stride_x = self.stride_convs[k](x)

			stride_mask = F.conv1d(mask, ones, stride = self.inflation)

			mask = (stride_mask != 0).float()

			multiplier = (1.0 / (stride_mask + 1e-6)) * mask

			stride_x *= multiplier

			x = F.relu(self.stride_bns[k](stride_x))

			x = F.relu(self.smooth_bns[k](self.smooth_convs[k](x)))

			x += res

		x = x.squeeze(-1)

		x = F.relu(self.a_lin(x))

		return self.b_lin(x)


class DiffNet(nn.Module):

	def __init__(self, args):

		super(DiffNet, self).__init__()

		self.n_blocks = args.n_blocks

		self.inflation = args.inflation

		self.expand_conv = nn.Conv1d((args.n_joints - 1) * 3, args.channels, 1, bias = False)
		self.root_expand_conv = nn.Conv1d(3, args.root_channels, 1, bias = False)

		self.expand_bn = nn.BatchNorm1d(args.channels)
		self.root_expand_bn = nn.BatchNorm1d(args.root_channels)

		self.stride_convs = []
		self.root_stride_convs = []

		self.stride_bns = []
		self.root_stride_bns = []

		self.smooth_convs = []
		self.root_smooth_convs = []

		self.smooth_bns = []
		self.root_smooth_bns = []

		for k in xrange(args.n_blocks):

			self.stride_convs.append(nn.Conv1d(args.channels, args.channels, 3, stride = args.inflation, bias = False))
			self.root_stride_convs.append(nn.Conv1d(args.root_channels, args.root_channels, 3, stride = args.inflation, bias = False))

			self.stride_bns.append(nn.BatchNorm1d(args.channels))
			self.root_stride_bns.append(nn.BatchNorm1d(args.root_channels))

			self.smooth_convs.append(nn.Conv1d(args.channels, args.channels, 1, bias = False))
			self.root_smooth_convs.append(nn.Conv1d(args.root_channels, args.root_channels, 1, bias = False))

			self.smooth_bns.append(nn.BatchNorm1d(args.channels))
			self.root_smooth_bns.append(nn.BatchNorm1d(args.root_channels))

		self.stride_convs = nn.ModuleList(self.stride_convs)
		self.root_stride_convs = nn.ModuleList(self.root_stride_convs)

		self.stride_bns = nn.ModuleList(self.stride_bns)
		self.root_stride_bns = nn.ModuleList(self.root_stride_bns)

		self.smooth_convs = nn.ModuleList(self.smooth_convs)
		self.root_smooth_convs = nn.ModuleList(self.root_smooth_convs)

		self.smooth_bns = nn.ModuleList(self.smooth_bns)
		self.root_smooth_bns = nn.ModuleList(self.root_smooth_bns)

		self.a_lin = nn.Linear(args.channels, args.channels)
		self.root_a_lin = nn.Linear(args.root_channels, args.root_channels)

		self.b_lin = nn.Linear(args.channels, (args.n_joints - 1) * 3)
		self.root_b_lin = nn.Linear(args.root_channels, 3)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)


	def forward(self, x, y, mask, ones):
		'''
		Args:
			x: (batch, 16 x 3, n_frames)
			y: (batch, 3, n_frames)
			mask: (batch, 1, n_frames)
			ones: (1, 1, 3)
		Return:
			(batch, 16 x 3)
			(batch, 3)
		'''
		mean_y = y.mean(dim = -1, keepdim = True)

		x = F.relu(self.expand_bn(self.expand_conv(x)))
		y = F.relu(self.root_expand_bn(self.root_expand_conv(y - mean_y)))

		for k in xrange(self.n_blocks):

			res_x = x[:, :, 2::self.inflation]
			res_y = y[:, :, 2::self.inflation]

			stride_x = self.stride_convs[k](x)
			stride_y = self.root_stride_convs[k](y)

			stride_mask = F.conv1d(mask, ones, stride = self.inflation)

			mask = (stride_mask != 0).float()

			multiplier = (1.0 / (stride_mask + 1e-6)) * mask

			stride_x *= multiplier
			stride_y *= multiplier

			x = F.relu(self.stride_bns[k](stride_x))
			y = F.relu(self.root_stride_bns[k](stride_y))

			x = F.relu(self.smooth_bns[k](self.smooth_convs[k](x)))
			y = F.relu(self.root_smooth_bns[k](self.root_smooth_convs[k](y)))

			x += res_x
			y += res_y

		x = x.squeeze(-1)
		y = y.squeeze(-1)
		mean_y = mean_y.squeeze(-1)

		return self.b_lin(F.relu(self.a_lin(x))), self.root_b_lin(F.relu(self.root_a_lin(y))) + mean_y