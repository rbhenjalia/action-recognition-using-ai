import os
import shutil 
path='/home/abhishek/Major Project/dataset/UCF-101'
path_test='/home/abhishek/Major Project/dataset/UCF-101_test'

for filename in os.listdir(path):
	count=0
	counter=0
	path1=path+'/'+filename
	length=len(os.listdir(path1))
	counter=int(length*0.2)
	os.makedirs(path_test+"/"+filename)
	for videos in os.listdir(path1):
		shutil.move(path1+"/"+videos,path_test+"/"+filename+"/"+videos)
		count+=1
		if count==counter:
			break
			






