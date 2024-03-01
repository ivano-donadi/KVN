import json

import numpy as np
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Tool for plotting the learning curves of a pvnet training session',
                                     epilog="")

parser.add_argument('-c', '--curves', 
                        help='Json file containing the learning curves data', required=True) 

args = parser.parse_args()

json_file = args.curves

with open(json_file, 'r') as f:
  data = json.load(f)

train_seg = []
train_vote = []
train_loss = []


val_seg = []
val_vote = []
val_loss = []

for point in data :
    train_seg.append(point['train']['seg_loss'])
    train_vote.append(point['train']['vote_loss'])
    train_loss.append(point['train']['loss'])
    val_seg.append(point['val']['seg_loss'])
    val_vote.append(point['val']['vote_loss'])
    val_loss.append(point['val']['loss'])

plt.title('Segmentation loss')
plt.plot(train_seg, label = 'training')
plt.plot(val_seg, label = 'validation')
plt.legend()
plt.show()

plt.title('Vote loss')
plt.plot(train_vote, label = 'training')
plt.plot(val_vote, label = 'validation')
plt.legend()
plt.show()

plt.title('Overall loss')
plt.plot(train_loss, label = 'training')
plt.plot(val_loss, label = 'validation')
plt.show()

