#!/usr/bin/env python

'''
 @author sapphirejyt@gmail.com

 Language Identifier for English, French, Spanish using Google Prediction API
 
 input: source sentence
 output: language ID
 usage: python main.py -i input_file -p project_id -m model_id -d training_data_path

'''

import optparse
import sys
import model

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="input.txt", help="input file")
optparser.add_option("-p", "--project_id", dest="pid", default="just-episode-97221", help="project ID")
optparser.add_option("-m", "--model_id", dest="mid", default="languageidentifier", help="model ID")
optparser.add_option("-d", "--data_loc", dest="data", default="traindata24/language_id.txt", help="training data location")
opts = optparser.parse_args()[0]

# Initialize language_id model
lm = model.TrainedModel(opts.pid, opts.mid)
# Upload training data and start training
lm.insert(opts.data)

# Check the training status 
status = lm.get()
while "DONE" not in status["trainingStatus"]:
    status = lm.get()

sentences = [line.strip() for line in open(opts.input).readlines()]
for sentence in sentences:
    label = lm.predict(sentence.split())['outputLabel']
    print "The language ID for \"", sentence, "\" is", label