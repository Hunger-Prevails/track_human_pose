python main.py \
				-decimal \
				-balance \
				-save_record \
				-model Naive \
				-suffix debug \
				-save_path /globalwork/liu/video_track \
				-accept_crit BCEWithLogits \
				-refine_crit SmoothL1 \
				-agnost_crit SmoothL1
