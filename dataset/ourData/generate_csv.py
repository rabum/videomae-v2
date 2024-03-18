import os
import random
import pandas as pd

def generate(index_videos):

    #train : val : test = 2:1:1
    train_file = open('train.csv', 'w')
    val_file = open('val.csv', 'w')
    test_file = open('test.csv', 'w')
    for label, files in index_videos.items():
        n = len(files)
        random.shuffle(files)
        train, val, test = files[:n//2],files[n//2:n*3//4],files[n*3//4:]
        for path in train: train_file.write(f'{path} {label}\n')
        for path in val: val_file.write(f'{path} {label}\n')
        for path in test: test_file.write(f'{path} {label}\n')
    train_file.close()
    val_file.close()
    test_file.close()

labels = ['door_closed', 'door_opened']
indexs = {'door_closed': 0, 'door_opened' : 1}
index_videos = {} #{0: ['1.mp4', '2.mp4'], ...}
for label in labels:
    files = os.listdir(f'{label}')
    label_index = indexs[f'{label}']
    if label_index not in index_videos:
        index_videos[label_index] = []
    for file in files:
        index_videos[label_index].append(f'{label}/{file}')

generate(index_videos)
        
        

