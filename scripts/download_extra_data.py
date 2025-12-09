import synapseclient
import os
import json
import tarfile

def download_extra_data():
    try:
        with open('config.json') as f:
            config = json.load(f)
        token = config.get('dataset_synapse_token')
        
        if not token:
            print("No token found.")
            return

        print("Logging in...")
        syn = synapseclient.Synapse()
        syn.login(authToken=token, silent=True)
        
        target_id = "syn52276405" # BraTS-Local-Synthesis-fastlane.tar.gz
        output_dir = "data/extra_data"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Downloading {target_id} to {output_dir}...")
        file_entity = syn.get(target_id, downloadLocation=output_dir)
        
        print(f"Downloaded: {file_entity.path}")
        
        if file_entity.path.endswith("tar.gz"):
            print("Extracting...")
            with tarfile.open(file_entity.path, "r:gz") as tar:
                tar.extractall(path="data/data") # Extract into main data folder
            print("Extraction complete.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_extra_data()
