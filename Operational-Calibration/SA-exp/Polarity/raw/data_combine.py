import os

path = 'tar/'

temp = 'pos'

files = os.listdir(path + temp)

f = open(path + 'tar.' + temp, 'a+')

for file in files:
	t = open(path + temp + '/' + file, 'rb')
	lines = t.readlines()
	f.write(str(lines[0]) + '\n')

f.close()
