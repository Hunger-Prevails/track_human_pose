python main.py \
				-save_record \
				-model Stride \
				-suffix auto_off \
				-n_epochs 40 \
				-save_path /globalwork/liu/track_camera_pose \
				-criterion SmoothL1 \
				-data_name mpihp \
				-data_root /globalwork/data/mpi_inf_3dhp
