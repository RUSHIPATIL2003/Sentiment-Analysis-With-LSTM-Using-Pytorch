import json
import os

# --- CHANGE THIS TO YOUR FILE NAME ---
filename = 'D:\\Pytorch\\Personal_Projects\\SENTEMENT_ANALYSIS_WITH_LSTM_IN_PYTORCH\\notebook\\SENTEMENT_ANALYSIS_WITH_LSTM_IN_PYTORCH.ipynb' 
# -------------------------------------

def strip_widget_metadata(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if the problematic key exists and delete it
    if 'metadata' in data and 'widgets' in data['metadata']:
        del data['metadata']['widgets']
        
        # Save the cleaned file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print(f"Success! 'metadata.widgets' removed from {file_path}")
    else:
        print("No 'widgets' metadata found. The file might already be clean.")

if __name__ == "__main__":
    strip_widget_metadata(filename)
