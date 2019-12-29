import os
import json
import torch
import pickle
import tracknet
import cameralib
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matlabfile as mlf

from opts import args
from utils import Tracklet
from builtins import zip as xzip


def cmu_camera(json_file, cam_name):

    calibration = json.load(open(json_file))

    for cam in calibration['cameras']:
        if cam['name'] == cam_name:
            return cameralib.Camera(
                    np.matmul(np.array(cam['R']).T, - np.array(cam['t'])),
                    np.array(cam['R']),
                    np.array(cam['K']),
                    np.array(cam['distCoef']),
                    (0, -1, 0)
            )

def mupots_camera(json_file, partition):

    calibration = json.load(open(json_file))

    return cameralib.Camera(
        np.zeros(3),
        np.eye(3),
        np.array(calibration[partition]),
        np.zeros(5),
        (0, -1, 0)
    )


def match(model, all_atn_specs, all_mat_specs, all_cam_specs, args):

    trajects = []

    tracklets = []

    for frame in xrange(args.on_stage, args.on_stage + args.duration):

        cam_specs = all_cam_specs[frame - args.on_frame]

        if cam_specs is None:

            for tracklet in tracklets:

                if len(tracklet) < args.in_frames:

                    tracklets.remove(tracklet)

                elif tracklet.b_forecast == 0:

                    trajects.append(tracklet)
                    tracklets.remove(tracklet)

                else:
                    tracklet.forecast(model)

            continue

        atn_specs = all_atn_specs[frame - args.on_frame]
        mat_specs = all_mat_specs[frame - args.on_frame]

        atn_specs = np.array(atn_specs).astype(np.float32)
        mat_specs = np.array(mat_specs).astype(np.float32)
        cam_specs = np.array(cam_specs).astype(np.float32)

        if not tracklets:

            tracklets += [Tracklet(atn_spec, mat_spec, cam_spec, frame, args) for atn_spec, mat_spec, cam_spec in xzip(atn_specs, mat_specs, cam_specs)]
            continue

        assc_score = np.stack([tracklet.associate(model, atn_specs, mat_specs, cam_specs) for tracklet in tracklets])  # (n_track, n_det)

        assc_match = np.zeros_like(assc_score).astype(np.int)

        assc_match[np.arange(assc_score.shape[0]), np.argmax(assc_score, axis = 1)] += 1
        assc_match[np.argmax(assc_score, axis = 0), np.arange(assc_score.shape[1])] += 1

        assc_match = (assc_match == 2)

        for i, tracklet in enumerate(tracklets):

            if np.any(assc_match[i]):

                atn_spec = atn_specs[assc_match[i]].squeeze()
                mat_spec = mat_specs[assc_match[i]].squeeze()
                cam_spec = cam_specs[assc_match[i]].squeeze()

                tracklet.incorporate(model, atn_spec, mat_spec, cam_spec)

            elif len(tracklet) < args.in_frames:

                tracklets.remove(tracklet)

            elif tracklet.b_forecast == 0:

                trajects.append(tracklet)
                tracklets.remove(tracklet)

            else:
                tracklet.forecast(model)

        tracklets += [Tracklet(atn_specs[k], mat_specs[k], cam_spec, frame, args) for k, cam_spec in enumerate(cam_specs) if not np.any(assc_match[:, k])]

    trajects += tracklets

    return trajects


def create_model(args):

    model = getattr(tracknet, args.model + 'Net')(args)

    checkpoint = torch.load(args.model_path)['model']
    
    model.load_state_dict(checkpoint)

    if args.n_cudas:
        cudnn.benchmark = True
        model = model.cuda() if args.n_cudas == 1 else nn.DataParallel(model, device_ids = range(args.n_cudas)).cuda()

    model.eval()

    return model


def chronicle(trajects, args):

    annals = dict(on_stage = args.on_stage, duration = args.duration, cam = dict(), mat = dict())

    for tid, traject in enumerate(trajects):

        print '=> => traject [', tid, ']:', traject.on_stage, '->', traject.off_stage

        assert len(traject) == len(traject.cam_dets)

        for frame in xrange(traject.on_stage, traject.off_stage):

            if args.smoothen:
                range_smooth = min(args.range_smooth, frame - traject.on_stage, traject.off_stage - frame - 1)

                cam_dets = traject.cam_dets[frame - traject.on_stage - range_smooth:frame - traject.on_stage + range_smooth + 1]
                mat_dets = traject.mat_dets[frame - traject.on_stage - range_smooth:frame - traject.on_stage + range_smooth + 1]

                curr_cam = np.mean(np.stack(cam_dets, axis = -1), axis = -1)
                curr_mat = np.mean(np.stack(mat_dets, axis = -1), axis = -1)
            else:
                curr_cam = traject.cam_dets[frame - traject.on_stage]
                curr_mat = traject.mat_dets[frame - traject.on_stage]

            if frame in annals['cam']:
                annals['cam'][frame][tid] = curr_cam.tolist()
                annals['mat'][frame][tid] = curr_mat.tolist()
            else:
                annals['cam'][frame] = dict(tid = curr_cam.tolist())
                annals['mat'][frame] = dict(tid = curr_mat.tolist())

    return annals


def save_as_array(annals, camera, args):

    cam_spec_array = []
    mat_spec_array = []

    for frame in xrange(args.on_frame, args.on_frame + args.duration):

        if frame not in annals['mat']:

            cam_spec_array.append(None)
            mat_spec_array.append(None)

            continue

        tids = annals['cam'][frame].keys()

        mat_specs = np.stack([annals['mat'][frame][tid] for tid in tids])
        cam_specs = np.stack([annals['cam'][frame][tid] for tid in tids])

        ones = np.ones((mat_specs.size / 2, 1))

        mat_specs = camera.camera_to_image(np.hstack([mat_specs.reshape(-1, 2), ones]))

        mat_spec_array.append(mat_specs.reshape(-1, args.num_joints, 2).tolist())
        cam_spec_array.append(cam_specs.reshape(-1, args.num_joints, 3).tolist())

    if args.data_name == 'cmu':
        path_det = os.path.join('./cmu_outcome', args.partition)

        if not os.path.exists(path_det):
            os.mkdir(path_det)

        path_mat = os.path.join(path_det, 'mat_' + args.cam_name + '_1.json')
        path_cam = os.path.join(path_det, 'cam_' + args.cam_name + '_1.json')

        with open(path_mat, 'w') as file:
            print >> file, json.dumps(
                dict(
                    mat = mat_spec_array,
                    start_frame = args.on_frame,
                    end_frame = args.on_frame + args.duration,
                )
            )
            file.close()

        with open(path_cam, 'w') as file:
            print >> file, json.dumps(
                dict(
                    cam = cam_spec_array,
                    start_frame = args.on_frame,
                    end_frame = args.on_frame + args.duration,
                )
            )
            file.close()

    else:
        path_mat = os.path.join('./mupots_outcome', args.partition + '_mat_1.json')
        path_cam = os.path.join('./mupots_outcome', args.partition + '_cam_1.json')

        with open(path_mat, 'w') as file:
            file.write(json.dumps(dict(mat = mat_spec_array)))
            file.close()

        with open(path_cam, 'w') as file:
            file.write(json.dumps(dict(cam = cam_spec_array)))
            file.close()


def main():
    model = create_model(args) if args.tracker else None

    if args.data_name == 'cmu':
        feat_path = os.path.join(args.feat_path, args.partition)

        path_atn = os.path.join(feat_path, 'detect_atn_' + args.cam_name + '.json')
        path_mat = os.path.join(feat_path, 'detect_mat_' + args.cam_name + '.json')
        path_cam = os.path.join(feat_path, 'detect_cam_' + args.cam_name + '.json')
    else:
        path_atn = os.path.join(args.feat_path, args.partition + '_atn.json')
        path_mat = os.path.join(args.feat_path, args.partition + '_mat.json')
        path_cam = os.path.join(args.feat_path, args.partition + '_cam.json')

    all_atn_specs = json.load(open(path_atn))
    all_mat_specs = json.load(open(path_mat))
    all_cam_specs = json.load(open(path_cam))

    if args.data_name == 'cmu':
        args.on_frame = atn_specs['start_frame']

        assert args.on_frame <= args.on_stage

        assert args.on_stage + args.duration <= atn_specs['end_frame']

        calib_file = '/globalwork/data/new-panoptic'

        calib_file = os.path.join(calib_file, args.partition, 'calibration_' + args.partition + '.json')

        camera = cmu_camera(calib_file, args.cam_name)
    else:
        root_data = '/globalwork/data/mupots'

        anno_file = os.path.join(root_data, args.partition, 'annot.mat')

        annotations = mlf.load(anno_file)['annotations']

        args.duration = len(annotations)

        calib_file = os.path.join(root_data, 'cam_params.json')

        camera = mupots_camera(calib_file, args.partition)

    print '\n=> collecting trajects'

    trajects = match(model, all_atn_specs['atn'], all_mat_specs['mat'], all_cam_specs['cam'], args)

    print '=> [', len(trajects), '] trajects collected\n'

    print '\n=> generating annals'

    annals = chronicle(trajects, args)

    print '=> annals generated\n'

    print '\n=> saving to array'

    save_as_array(annals, camera, args)

    print '=> arrays saved\n'


if __name__ == '__main__':
	main()
