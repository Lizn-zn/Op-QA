import os
import shutil

f = open('validation_label.txt', 'r')
lines = f.readlines()
for line in lines:
    name, label = line.split(' ')
    print(name)
    label = label.replace('\n', '')
    if not os.path.exists('val-data/'+label):
        os.makedirs('val-data/'+label)
    shutil.copyfile('val-data/temp/'+name,'val-data/'+label+'/'+name)
