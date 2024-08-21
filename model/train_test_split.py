import os
import shutil 
path='D:\\Major-Project\\CNNnLSTM\\Project\\UCF-101'
path_test='D:\\Major-Project\\CNNnLSTM\\Project\\UCF-101_test'

for foldername in os.listdir(path):
	count=0
	counter=0
	folderpath=path+"\\"+foldername
	length=len(os.listdir(folderpath))
	counter=int(length*0.2)
	os.makedirs(path_test+"\\"+foldername)
	for videos in os.listdir(folderpath):
		shutil.move(folderpath+"\\"+videos,path_test+"\\"+foldername+"\\"+videos)
		count+=1
		if count==counter:
			break
