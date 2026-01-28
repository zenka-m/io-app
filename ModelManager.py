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
        pass
        
    def tune_fit(self, X_train, y_train, verbose=False):
        pass

    def predict(self, X_test: np.ndarray) -> pd.DataFrame:
        """Main prediction method with Robust Selector support."""
        target_cols = list(self.cfg.TARGET_COLS)

        if not self.models:
            raise RuntimeError('You need to fit or load models first!')
        
        predictions = {}

        for target_param in target_cols:
            if target_param in self.models:
                model = self.models[target_param]
                
                # --- PREPARE DATA FOR SPECIFIC TARGET ---
                X_input = X_test.copy() 

                # 1. Determine which selector to use
                current_selector = None
                
                if isinstance(self.selector, dict):
                    # Try to find exact match (e.g. 'P' for 'P')
                    if target_param in self.selector:
                        current_selector = self.selector[target_param]
                    # Fallback: If dictionary has only 1 item, use it for all targets
                    elif len(self.selector) == 1:
                        key = list(self.selector.keys())[0]
                        current_selector = self.selector[key]
                        # Only print this warning once per batch to avoid spam
                        if target_param == target_cols[0]: 
                            print(f"Warning: Exact match for '{target_param}' not found in selector. Using fallback: '{key}'")
                else:
                    # If it's not a dict, it's a single global selector object
                    current_selector = self.selector

                # 2. Apply transformation if selector exists
                if current_selector is not None:
                    try:
                        # Case A: Wrapper or RFE (has transform method)
                        if hasattr(current_selector, 'transform'):
                            X_input = current_selector.transform(X_input)
                        # Case B: Raw object with support_ mask
                        elif hasattr(current_selector, 'support_'):
                            X_input = X_input[:, current_selector.support_]
                        # Case C: Nested Wrapper (wrapper.selector_.transform)
                        elif hasattr(current_selector, 'selector_') and hasattr(current_selector.selector_, 'transform'):
                             X_input = current_selector.selector_.transform(X_input)
                    except Exception as e:
                        print(f"Error: Selector transformation failed for {target_param}: {e}")
                
                # Debug print if shapes still mismatch
                # if X_input.shape[1] != 50:
                #     print(f"DEBUG: {target_param} input shape: {X_input.shape}")

                # 3. Predict
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
