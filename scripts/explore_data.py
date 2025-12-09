import synapseclient
import synapseutils
import json
import os

def list_files():
    try:
        with open('config.json') as f:
            config = json.load(f)
        token = config.get('dataset_synapse_token')
        
        if not token:
            print("No token found in config.json")
            return

        print("Logging in...")
        syn = synapseclient.Synapse()
        syn.login(authToken=token, silent=True)
        
        print("Listing files in project syn64952532...")
        walked = synapseutils.walk(syn, 'syn64952532')
        
        files = []
        for dirpath, dirnames, filenames in walked:
            for filename in filenames:
                files.append(filename)
                print(f"Found: {filename[0]} (ID: {filename[1]})")
                
        if not files:
            print("No files found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_files()
