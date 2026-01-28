from dataclasses import dataclass

@dataclass
class Config:
    RANDOM_SEED: int = 123

    TRAIN_DATA_PATH: str = 'train_data'
    TEST_DATA_PATH: str = 'test_data'

    GT_PATH: str = 'train_gt.csv'
    WL_PATH: str = 'wavelengths.csv'

    SUBMISSION_DIR: str = '.'
    TARGET_COLS: tuple =('P', 'K', 'Mg', 'pH')

    SIGMA_PARAM = (0, 2, 2) 
    SG_WINDOW = 11
    SG_POLY = 2
    SG_DERIV = 1

    RFE_N_FEATURES = 40

    CV_FOLDS = 5
    CV_ITER = 80
    CV_SCORING_MSE = 'neg_mean_squared_error'
    CV_SCORING_MAE = 'neg_mean_absolute_error'

    MODEL_TYPE: str = 'rf'
    


