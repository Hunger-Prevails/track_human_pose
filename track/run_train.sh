python main.py \
				-save_record \
				-resume \
				-model_path /globalwork/liu/track_camera_pose/Partial-base/model_16.pth \
				-model Partial \
				-suffix both \
				-save_path /globalwork/liu/track_camera_pose \
				-criterion SmoothL1 \
				-data_name mpihp \
				-data_root /globalwork/data/mpi_inf_3dhp
