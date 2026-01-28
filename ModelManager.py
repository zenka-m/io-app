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
        self.param_grid = {} 
    
    def _get_model_conf(self):
        pass
        
    def tune_fit(self, X_train, y_train, verbose=False):
        pass

    def predict(self, X_test: np.ndarray) -> pd.DataFrame:
        """Prediction method using the global feature selector."""
        # print(f"DEBUG: Input shape before selection: {X_test.shape}")
        
        target_cols = list(self.cfg.TARGET_COLS)

        if not self.models:
            raise RuntimeError('You need to load models first!')
        
        predictions = {}

        for target_param in target_cols:
            if target_param in self.models:
                model = self.models[target_param]
                X_input = X_test.copy() 

                # --- APPLY FEATURE SELECTION ---
                if self.selector is not None:
                    try:
                        # Common selector methods (RFE, Boruta, etc.)
                        if hasattr(self.selector, 'transform'):
                            X_input = self.selector.transform(X_input)
                        elif hasattr(self.selector, 'support_'):
                            X_input = X_input[:, self.selector.support_]
                        elif hasattr(self.selector, 'selector_') and hasattr(self.selector.selector_, 'transform'):
                             # Nested wrapper case
                             X_input = self.selector.selector_.transform(X_input)
                        
                        # print(f"DEBUG: Shape after selection for {target_param}: {X_input.shape}")
                        
                    except Exception as e:
                        print(f"Error applying selector for {target_param}: {e}")
                # -------------------------------

                # Predict
                y_pred = model.predict(X_input)
                predictions[target_param] = y_pred

        return pd.DataFrame(predictions)
    
    def load_models(self, filename:str = 'models.pkl'):
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        path = filename
        if not os.path.exists(path): path = os.path.join(self.cfg.SUBMISSION_DIR, 'Models', filename)
        if not os.path.exists(path): path = filename 
        if not os.path.exists(path): path = os.path.join('Models', filename)
            
        if not os.path.exists(path):
            raise FileNotFoundError(f'Model file not found: {filename}')
        
        print(f"Loading models from: {path}")
        self.models = joblib.load(path)

        # --- SMART SELECTOR LOADING ---
        selector_path = path.replace('.pkl', '_selector.pkl')
        
        if os.path.exists(selector_path):
            try:
                loaded_obj = joblib.load(selector_path)
                
                # CHECK: Is this the dictionary with metadata?
                if isinstance(loaded_obj, dict) and 'selector' in loaded_obj:
                    print("Unpacking 'selector' object from dictionary metadata.")
                    self.selector = loaded_obj['selector']
                else:
                    # It's either a direct object or a per-target dict (less likely now)
                    self.selector = loaded_obj
                
                print(f"Selector loaded successfully. Type: {type(self.selector)}")
            except Exception as e:
                print(f"Selector load error: {e}")
                self.selector = None
        else:
            print("Selector file not found.")
            self.selector = None