import synapseclient
import synapseutils
import os
import json

import tarfile

def download_data(token: str, data_dir: str = 'data', limit: int = 1):
    """
    Logs into Synapse using the provided token and downloads a sample dataset.
    Returns True on success, False on failure.
    """
    if not token:
        print("Synapse token not provided.")
        return False

    try:
        print("Logging in to Synapse...")
        syn = synapseclient.Synapse()
        syn.login(authToken=token, silent=True)
        
        target_file = "BraTS-GLI-fastlane.tar.gz"
        print(f"Searching for sample dataset '{target_file}' in 'syn64952532'...")
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Walk through the hierarchy to find files
        walked = synapseutils.walk(syn, 'syn64952532')
        
        found = False
        for dirpath, dirnames, filenames in walked:
            for filename in filenames:
                fname, syn_id = filename
                if fname == target_file:
                    print(f"Found sample dataset: {fname} (ID: {syn_id})")
                    print(f"Downloading to {data_dir}...")
                    syn.get(syn_id, downloadLocation=data_dir)
                    found = True
                    break
            if found:
                break
        
        if not found:
            print(f"Sample file '{target_file}' not found.")
            return False

        print("Download complete. Extracting...")
        
        # Extract the tar.gz file
        tar_path = os.path.join(data_dir, target_file)
        if tarfile.is_tarfile(tar_path):
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=data_dir)
            print(f"Extracted {target_file}.")
            
            # Optional: Clean up the tar file to save space
            # os.remove(tar_path) 
        else:
            print(f"{target_file} is not a valid tar file.")

        return True

    except Exception as e:
        print(f"An error occurred during data download: {e}")
        return False

if __name__ == "__main__":
    try:
        with open('config.json') as f:
            config = json.load(f)
        synapse_token = config.get('dataset_synapse_token')
        if synapse_token:
            # Download only 1 file for testing/small batch
            download_data(synapse_token, limit=1)
        else:
            print("Synapse token not found in config.json.")
    except FileNotFoundError:
        print("config.json not found. Please create it with your Synapse token.")