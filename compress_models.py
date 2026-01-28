import joblib
import os

# We import FeatureSelector to ensure the custom class is recognized during loading
try:
    from FeatureSelection import FeatureSelector
except ImportError:
    print("Warning: FeatureSelection module not found. Proceeding anyway.")

def compress_file(filename):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    original_size = os.path.getsize(filename) / (1024 * 1024)
    print(f"Loading: {filename} ({original_size:.2f} MB)...")
    
    try:
        # 1. Load the heavy file
        data = joblib.load(filename)
        
        # 2. Save it back with compression (level 3 is optimal)
        joblib.dump(data, filename, compress=3)
        
        new_size = os.path.getsize(filename) / (1024 * 1024)
        print(f"Success: {filename} compressed to {new_size:.2f} MB")
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # List of files to compress
    files_to_compress = [
        "best_rf_model.pkl",
        "best_rf_model_selector.pkl"
    ]
    
    for f in files_to_compress:
        compress_file(f)