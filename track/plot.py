import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

short_names = [
	'htop',
	'neck',
	'rsho',
	'relb',
	'rwri',
	'lsho',
	'lelb',
	'lwri',
	'rhip',
	'rkne',
	'rank',
	'lhip',
	'lkne',
	'lank',
	'spin',
	'head',
	'pelv'
]
parent = dict(
	[
		('htop', 'head'),
		('head', 'neck'),
		('rsho', 'neck'),
		('relb', 'rsho'),
		('rwri', 'relb'),
		('lsho', 'neck'),
		('lelb', 'lsho'),
		('lwri', 'lelb'),
		('rhip', 'pelv'),
		('rkne', 'rhip'),
		('rank', 'rkne'),
		('lhip', 'pelv'),
		('lkne', 'lhip'),
		('lank', 'lkne'),
		('neck', 'spin'),
		('spin', 'pelv'),
		('pelv', 'pelv')
	]
)

def show_cam(cam_coords, ax, color):
	'''
	Shows coco19 skeleton(3d)

	Args:
		cam_coords: (N, 3, num_joints)
	'''
	assert len(short_names) == cam_coords.shape[-1]

	mapper = dict(zip(short_names, range(len(short_names))))

	body_edges = [mapper[parent[name]] for name in short_names]
	body_edges = np.hstack(
		[
			np.arange(len(body_edges)).reshape(-1, 1),
			np.array(body_edges).reshape(-1, 1)
		]
	)
	ax.view_init(10, -80)
	ax.set_xlim3d(-200, 200)
	ax.set_ylim3d(0, 500)
	ax.set_zlim3d(-200, 100)

	for i, cam_coord in enumerate(cam_coords):

		ax.plot(cam_coord[0], - cam_coord[1], cam_coord[2], '.', zdir = 'y', color = 0.5 + (color - 0.5) * (float(i + 1) / len(cam_coords)))

		for edge in body_edges:
			ax.plot(cam_coord[0, edge], - cam_coord[1, edge], cam_coord[2, edge], zdir = 'y', color = 0.5 + (color - 0.5) * (float(i + 1) / len(cam_coords)))