import pandas as pd
import numpy as np
from typing import List, Optional , Tuple, Dict
import os
from config import Config




class DataManager:

    def __init__(self, config: Config):
        self.cfg = config
        self._ensure_submission_dir()

    def _ensure_submission_dir(self):
        """Creates submission directory if it doesn't exist.
        """
        os.makedirs(self.cfg.SUBMISSION_DIR, exist_ok=True)

    def load_ground_truth(self) -> pd.DataFrame: 
        """Loads the ground truth data for the training set from CSV.

        Raises:
            FileNotFoundError: If the GT file path does not exist

        Returns:
            pd.DataFrame: DF containing ground truth values
        """
        gt_path = self.cfg.GT_PATH

        if not os.path.exists(gt_path):
            raise FileNotFoundError(f'Could not find GT file: {gt_path}')
        
        return pd.read_csv(gt_path, index_col=0)
    
    def load_wavelengths(self) -> pd.DataFrame:
        """Loads wavelenth information from CSV.
        
        Raises:
            FileNotFoundError: If the wavelengths file path does not exist.

        Returns:
            pd.DataFrame: DataFrame structured: (wave index, wave length) 
        """
        wl_path = self.cfg.WL_PATH

        if not os.path.exists(wl_path):
            raise FileNotFoundError(f'Could not find GT file: {wl_path}')
        
        return pd.read_csv(wl_path, index_col=0)
    
    def load_image(self, image_id:int, file_path:str = None) -> np.ma.MaskedArray:
        """Loads a hypersepctral image from an .npz file based on its ID.

        Args:
            image_id (int): ID of the image
            file_path (str, optional): Custom directory path. Defaults to None.

        Returns:
            np.ma.MaskedArray: Masked array of the loaded spectral image 
        """
        if file_path is None:
            file_path = self.cfg.TRAIN_DATA_PATH

        img_path = os.path.join(file_path, f'{image_id}.npz')

        if not os.path.exists(img_path):
            print(f'Could not find file: {img_path}')
            return None
        
        try: 
            with np.load(img_path) as npz:
                spectral_arr = np.ma.MaskedArray(**npz)
                return spectral_arr
        except Exception as e:
            print(f'Error while loading: {img_path}')
            return None
        
    def get_data_indices(self, file_path: str):
        """Retrieves a list of image IDs found in a directory

        Args:
            file_path (str): The directory path

        Returns:
            List[int]: Sorted List of integer IDs retrieved from filenames in the directory  
        """
        if not os.path.exists(file_path):
            print(f'Could not find directory: {file_path}')
            return None
        
        image_ids = []

        for filename in os.listdir(file_path):
            if filename.endswith('.npz'):
                try:
                    img_id = int(filename.replace('.npz', ''))
                    image_ids.append(img_id)
                except ValueError:
                    print(f'Warning: Could not parse image ID from {filename}')
                    
        image_ids.sort()
        return image_ids #np.array(image_ids, dtype=np.int16)

    def save_submission(self, predictions: np.ndarray, filename: str = "submission.csv"):     
        """Saves predictions to a CSV file.

        Args:
            predictions (np.ndarray): Soil parameters predictions
            filename (str, optional): Output file name. Defaults to "submission.csv".
        """
        if not filename.endswith('.csv'):
            filename += '.csv'
        path = os.path.join(self.cfg.SUBMISSION_DIR, filename)

        columns = list(self.cfg.TARGET_COLS)
        submission_df = pd.DataFrame(data=predictions, columns=columns)

        submission_df.to_csv(path, index_label="sample_index")

        print(f'Succesfully saved submission file to: {path}')


      ################## NOWE ##############
    def load_image_from_path(self, file_path: str) -> np.ma.MaskedArray:
        """Loads a hyperspectral image directly from a full file path.

        Args:
            file_path (str): Full path to the .npz file

        Returns:
            np.ma.MaskedArray: Loaded spectral image
        """

        if not file_path.lower().endswith('.npz'):
            print(f"Error: file {file_path} is not .npz")
            return None

        if not os.path.exists(file_path):
            print(f"Error: file does not exist: {file_path}")
            return None



        try:
            with np.load(file_path) as npz:
                spectral_arr = np.ma.MaskedArray(**npz)
                return spectral_arr
        except Exception as e:
            print(f'Error while loading: {file_path}')
            return None





