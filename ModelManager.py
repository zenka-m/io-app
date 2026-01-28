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
        """Robust prediction method handling specific dictionary structures."""
        print("\n--- STARTING PREDICTION ---")
        print(f"DEBUG: Input shape (X_test): {X_test.shape}")
        
        target_cols = list(self.cfg.TARGET_COLS)

        if not self.models:
            raise RuntimeError('You need to load models first!')
        
        predictions = {}

        # --- SELECTOR LOGIC START ---
        # Determine the active selector object
        
        active_selector = None
        
        if self.selector is not None:
            if isinstance(self.selector, dict):
                keys = list(self.selector.keys())
                print(f"DEBUG: Selector dict keys: {keys}")
                
                # CASE 1: The 'selector' key exists (Fix for your specific file)
                if 'selector' in self.selector:
                    print("DEBUG: Found 'selector' key. Extracting the object inside.")
                    active_selector = self.selector['selector']
                
                # CASE 2: Single key fallback
                elif len(self.selector) == 1:
                    key = keys[0]
                    print(f"DEBUG: Using single available key: '{key}'")
                    active_selector = self.selector[key]
                
                # CASE 3: Per-target dictionary (handled inside loop)
                else:
                    print("DEBUG: Dictionary might be per-target. Will check inside loop.")
            else:
                # CASE 4: Direct object
                print("DEBUG: Selector is a direct object (not a dict).")
                active_selector = self.selector
        else:
             print("DEBUG: self.selector is None.")
        # ----------------------------

        for target_param in target_cols:
            if target_param in self.models:
                model = self.models[target_param]
                X_input = X_test.copy() 

                # Determine final selector for this specific target
                current_selector = active_selector
                
                # Fallback: Check if the dict has a key for this target (e.g. 'P')
                if current_selector is None and isinstance(self.selector, dict):
                    if target_param in self.selector:
                        current_selector = self.selector[target_param]

                # APPLY TRANSFORMATION
                if current_selector is not None:
                    try:
                        # Logic to extract the mask or transform
                        if hasattr(current_selector, 'transform'):
                            X_input = current_selector.transform(X_input)
                        elif hasattr(current_selector, 'support_'):
                            X_input = X_input[:, current_selector.support_]
                        elif hasattr(current_selector, 'selector_') and hasattr(current_selector.selector_, 'transform'):
                             X_input = current_selector.selector_.transform(X_input)
                        elif hasattr(current_selector, 'selector_') and hasattr(current_selector.selector_, 'support_'):
                             X_input = X_input[:, current_selector.selector_.support_]
                        
                        print(f"DEBUG: Transformation successful for {target_param}. New shape: {X_input.shape}")
                        
                    except Exception as e:
                        print(f"DEBUG: Transformation error for {target_param}: {e}")
                
                # Final check on shape
                if X_input.shape[1] > 100:
                     print(f"WARNING: Feature reduction might have failed. Features count: {X_input.shape[1]}")

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
        
        print(f"DEBUG: Loading model from: {path}")
        self.models = joblib.load(path)

        selector_path = path.replace('.pkl', '_selector.pkl')
        print(f"DEBUG: Looking for selector in: {selector_path}")
        
        if os.path.exists(selector_path):
            try:
                self.selector = joblib.load(selector_path)
                print(f"DEBUG: Loaded selector. Type: {type(self.selector)}")
            except Exception as e:
                print(f"DEBUG: Load error: {e}")
                self.selector = None
        else:
            print("DEBUG: Selector file not found.")
            self.selector = None