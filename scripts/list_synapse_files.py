
import synapseclient
import json

with open('config.json') as f:
    config = json.load(f)
token = config.get('dataset_synapse_token')

syn = synapseclient.Synapse()
syn.login(authToken=token, silent=True)

entity = syn.get('syn64952532')
manifest = syn.gettablecolumns(entity)

for row in manifest:
    print(f"File: {row['name']}, Size: {row['contentSize'] / 1024 / 1024:.2f} MB")
