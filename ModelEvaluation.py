from sklearn.metrics import r2_score as R2, mean_squared_error as MSE, mean_absolute_error as MAE
import numpy as np
import pandas as pd

from config import Config

class Evaluator:
    def __init__(self, config: Config):
        self.cfg = config       

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, verbose: bool=False):
        target_cols = list(self.cfg.TARGET_COLS)
        metrics = {}

        for i, target_param in enumerate(target_cols):
            y_true_target = y_true[:,i]
            y_pred_target = y_pred[:,i]
        
            r2 = R2(y_true_target, y_pred_target)
            mae = MAE(y_true_target, y_pred_target)
            mse = MSE(y_true_target, y_pred_target)
            rmse = np.sqrt(mse)

            metrics[target_param] = {'R2':r2, 'MAE':mae, 'MSE':mse, 'RMSE':rmse}

            if verbose:
                print(f'[{target_param}] parameter model scorings - R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f} RMSE: {rmse:.4f}\n')

        return metrics

    def calculate_hyperview_score(self, y_true: np.ndarray, y_pred: np.ndarray, y_baseline ,verbose:bool = False):
        target_cols = list(self.cfg.TARGET_COLS)
        scores = {}

        baselines_mse = np.mean((y_true - y_baseline) ** 2, axis=0)
        models_mse = np.mean((y_true - y_pred) ** 2, axis=0)

        scores = models_mse / baselines_mse
        final_score = np.mean(scores)

        for score, target_param in zip(scores, target_cols):
            scores[target_param] = score
            if verbose:
                print(f"Class {target_param} score: {score}")

        scores['FINAL_SCORE'] = final_score
        if verbose:
            print(f"Final HYPERVIEW score: {final_score}")

        return scores