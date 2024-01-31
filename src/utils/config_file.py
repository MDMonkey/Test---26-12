import argparse


def load_args():
	
	parser = argparse.ArgumentParser()

	#Training file
	
	parser.add_argument('--file_dir', type=str, default=r'C:\Users\totti\OneDrive\Work\ITA-PUC\Data\Meio_canal_Retau180')
	
	#Location
	parser.add_argument('--file_location', default=r"C:\Users\totti\OneDrive\Work\ITA-PUC\Data\Meio_canal_Retau180\analysis_s2.h5")

	#Working Directory
	parser.add_argument('--working_dir', default=r'/Code/Emps/Deepxde')
	parser.add_argument('--PERC_TRAIN', type=float, default=0.75)
	parser.add_argument('--PERC_VAL', type=float, default=0.15)
	parser.add_argument('--PERC_TEST', type=float, default=0.10)

	#Model
	parser.add_argument('--N_VARS_OUT', type=int, default=3)
	parser.add_argument('--N_VARS_IN', type=int, default=3)
	parser.add_argument('--N_PLANES_OUT', type=int, default=2)
	parser.add_argument('--VARS_NAME_IN', type=list, default=['u','w','p'])
	parser.add_argument('--VARS_NAME_OUT', type=list, default=['u','v','w'])

	parser.add_argument('--NORMALIZE_INPUT', type=bool, default=True)
	parser.add_argument('--PRED_FLUCT', type=bool, default=True)
	parser.add_argument('--SCALE_OUTPUT', type=bool, default=True)
	parser.add_argument('--PADDING_INPUT', type=bool, default=True)
	parser.add_argument('--PADDING_SIZE', type=int, default=16)

	
    # Hardware definition
	parser.add_argument('--ON_GPU', type=bool, default=True)
	parser.add_argument('--N_GPU', type=int, default=1)
	parser.add_argument('--WHICH_GPU_TRAIN', type=int, default=0)
	parser.add_argument('--WHICH_GPU_TEST', type=int, default=1)

    #Training
	parser.add_argument('--FLUCTUATION_PRED', type=bool, default=False)
	parser.add_argument('--RELU_THRESHOLD', type=float, default=-1.0)

	parser.add_argument('--TRAIN_YP', type=int, default=0)
	parser.add_argument('--TARGET_YP', type=int, default=50)

	parser.add_argument('--N_EPOCHS', type=int, default=1)
	parser.add_argument('--BATCH_SIZE', type=int, default=32)
	parser.add_argument('--VAL_SPLIT', type=float, default=0.2)
	parser.add_argument('--TEST_SPLIT', type=float, default=0.2)


	parser.add_argument('--INIT_LR', type=float, default=0.001)
	parser.add_argument('--LR_DROP', type=float, default=0.5)
	parser.add_argument('--LR_EPDROP', type=float, default=40.0)

	args = parser.parse_args()

	return args