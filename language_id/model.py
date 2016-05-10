#!/usr/bin/env python
import json
import httplib2
from apiclient.discovery import build
from oauth2client.file import Storage
from oauth2client.client import OAuth2WebServerFlow 
from oauth2client.tools import run

# Initial setup 
keys = json.load(open("secrets.json"))
client_id = keys["web"]["client_id"]
client_secret = keys["web"]["client_secret"]

scope = {'https://www.googleapis.com/auth/devstorage.full_control', 
            'https://www.googleapis.com/auth/devstorage.read_only',
            'https://www.googleapis.com/auth/devstorage.read_write',        
            'https://www.googleapis.com/auth/prediction'}
            
flow = OAuth2WebServerFlow(client_id, client_secret, scope)

storage = Storage("credentials.dat")
credentials = storage.get()
if credentials is None or credentials.invalid:
    credentials = run(flow, storage)
           
http = credentials.authorize(httplib2.Http())
service = build("prediction", "v1.6", http=http)


class HostedModel(object):

    Hosted_model_id = 40204645227

    def predict(self, model_id, csv_instances):
        body = {
                    input: { "csvInstance": csv_instances }
                }
        return service.hostedmodels().predict()


class TrainedModel(object):

    def __init__ (self, project_id, model_id):
        self.p = project_id
        self.m = model_id

    # Upload training data from the given path and train the system
    def insert(self, storage_data_location=None, sentence=None, label=None):
        body = {    
                    "id": self.m,
                    "storageDataLocation": storage_data_location,
                    "trainingInstances": [
                        {   
                            "csvInstance": sentence,
                            "output": label
                        }
                    ]
                }
        return service.trainedmodels().insert(project=self.p, body=body).execute()

    # Upload training data instances and train the system
    def insert_dataset(self, training_data):
        body = {
                    "id": self.m,
                    "trainingInstances": training_data
                }
        return service.trainedmodels().insert(project=self.p, body=body).execute()

    # Check training status
    def get(self):
        return service.trainedmodels().get(project=self.p, id=self.m).execute()

    # Identify language ID given a source sentence
    def predict(self, sentence):
        body = {
                    "input": { "csvInstance": sentence }
                }
        return service.trainedmodels().predict(project=self.p, id=self.m, body=body).execute()