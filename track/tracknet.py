import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveNet(nn.Module):

	def __init__(self, args):

		super(NaiveNet, self).__init__()

		self.expand_conv = nn.Conv1d(args.num_joints * args.in_features, args.channels, 3, bias = False)
		self.expand_bn = nn.BatchNorm1d(args.channels)

		self.drop = nn.Dropout(args.dropout)

		self.dilate_convs = []
		self.dilate_bns = []

		self.smooth_convs = []
		self.smooth_bns = []

		dilation = args.inflation

		self.n_blocks = args.n_blocks
		self.shift = []

		for k in xrange(args.n_blocks):

			self.shift.append(args.kernel * dilation - dilation)

			self.dilate_convs.append(nn.Conv1d(args.channels, args.channels, args.kernel, dilation = dilation, bias = False))
			self.dilate_bns.append(nn.BatchNorm1d(args.channels))

			self.smooth_convs.append(nn.Conv1d(args.channels, args.channels, 1, bias = False))
			self.smooth_bns.append(nn.BatchNorm1d(args.channels))

			dilation *= args.inflation

		self.dilate_convs = nn.ModuleList(self.dilate_convs)
		self.dilate_bns = nn.ModuleList(self.dilate_bns)

		self.smooth_convs = nn.ModuleList(self.smooth_convs)
		self.smooth_bns = nn.ModuleList(self.smooth_bns)

		self.lin_a = nn.Linear(args.channels + args.num_joints * args.in_features, args.channels)
		self.lin_b = nn.Linear(args.channels, args.channels)

		self.accept = nn.Linear(args.channels, 1)
		self.refine = nn.Linear(args.channels, args.num_joints * 3)
		self.agnost = nn.Linear(args.channels, args.num_joints * 3)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, mean = 0.0, std = (2.0 / (m.in_features + m.out_features)) ** 0.5)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)


	def forward(self, x, y):
		'''
		Args:
			x: (batch, num_joints x in_features, n_frames)
			y: (batch, num_joints x in_features)
		Return:
			accept: (batch, 1)
			refine: (batch, num_joints x 3)
			agnost: (batch, num_joints x 3)
		'''
		assert x.size(-1) == sum(self.shift) + 3

		batch = x.size(0)

		x = self.drop(F.relu(self.expand_bn(self.expand_conv(x))))

		for k in xrange(self.n_blocks):

			res = x[:, :, self.shift[k]:]

			x = self.drop(F.relu(self.dilate_bns[k](self.dilate_convs[k](x))))
			x = self.drop(F.relu(self.smooth_bns[k](self.smooth_convs[k](x))))

			x += res

		x = x.squeeze()

		concat = torch.cat([x, y], dim = -1)

		concat = F.relu(self.lin_a(concat))
		concat = F.relu(self.lin_b(concat))

		return self.accept(concat), self.refine(concat), self.agnost(x)
