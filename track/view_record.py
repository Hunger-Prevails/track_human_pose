import torch
import sys
import numpy as np

def main(model, suffix, *keys):
	record = torch.load('/globalwork/liu/track_camera_pose/' + model + '-' + suffix + '/protocol.pth')

	for key in keys:
		strings = ['%2.5f' % (val) for val in record[key]]
		max_idx = np.argmax(np.array(record[key]))
		min_idx = np.argmin(np.array(record[key]))

		print key + ':'
		print strings
		print 'max:', max_idx + 1, strings[max_idx]
		print 'min:', min_idx + 1, strings[min_idx]

	best = torch.load('/globalwork/liu/track_camera_pose/' + model + '-' + suffix + '/best.pth')['best']

	print '-' * 40
	print 'best:', best

	for key in keys:
		strings = ['%2.5f' % (val) for val in record[key]]
		print key + ':', strings[best - 1]

if __name__ == '__main__':
	main(*sys.argv[1:])
