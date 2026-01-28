import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from config import Config

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

class ModelManager:
    def __init__(self, config: Config, model_type: str = "rf"):
        self.cfg = config
        self.models = {}
        self.model_type = model_type
        self.selector = None  
        self.param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [None, 10, 20]
        }
    
    def _get_model_conf(self):
        # Helper method for model configuration (not used in inference)
        pass
        
    def tune_fit(self, X_train, y_train, verbose=False):
        # Training logic (not used in inference)
        pass

    def predict(self, X_test: np.ndarray) -> pd.DataFrame:
        """Main prediction method with Dictionary Selector support."""
        target_cols = list(self.cfg.TARGET_COLS)

        if not self.models:
            raise RuntimeError('You need to fit or load models first!')
        
        predictions = {}

        for target_param in target_cols:
            if target_param in self.models:
                model = self.models[target_param]
                
                # --- PREPARE DATA FOR SPECIFIC TARGET ---
                X_input = X_test.copy() 

                # 1. Check if we have a dictionary of selectors (per target)
                current_selector = None
                
                if isinstance(self.selector, dict):
                    # Get selector for P, K, Mg...
                    current_selector = self.selector.get(target_param)
                else:
                    # Or use global selector
                    current_selector = self.selector

                # 2. If selector exists, apply transformation
                if current_selector is not None:
                    try:
                        # Case A: Wrapper or RFE (has transform method)
                        if hasattr(current_selector, 'transform'):
                            X_input = current_selector.transform(X_input)
                        # Case B: Raw object with support_ mask
                        elif hasattr(current_selector, 'support_'):
                            X_input = X_input[:, current_selector.support_]
                        # Case C: Nested Wrapper
                        elif hasattr(current_selector, 'selector_') and hasattr(current_selector.selector_, 'transform'):
                             X_input = current_selector.selector_.transform(X_input)
                    except Exception as e:
                        print(f"Warning: Selector failed for {target_param}: {e}")

                # 3. Predict on (potentially) reduced data
                y_pred = model.predict(X_input)
                predictions[target_param] = y_pred

        return pd.DataFrame(predictions)
    
    def load_models(self, filename:str = 'models.pkl'):
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        path = filename
        # Robust path searching logic
        if not os.path.exists(path): path = os.path.join(self.cfg.SUBMISSION_DIR, 'Models', filename)
        if not os.path.exists(path): path = filename 
        if not os.path.exists(path): path = os.path.join('Models', filename)
            
        if not os.path.exists(path):
            raise FileNotFoundError(f'Model file not found: {filename}')
        
        self.models = joblib.load(path)
        print(f"Loaded models from: {path}")

        # Try loading selector
        selector_path = path.replace('.pkl', '_selector.pkl')
        if os.path.exists(selector_path):
            try:
                self.selector = joblib.load(selector_path)
                print(f"Loaded selector: {selector_path}")
            except Exception as e:
                print(f"Selector load error: {e}")
                self.selector = None
        else:
            self.selector = None