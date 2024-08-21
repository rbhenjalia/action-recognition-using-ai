from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
import os
from tqdm import tqdm
import fnmatch
from time import time


path_sequence='D:\\Major-Project\\CNNnLSTM\\Project\\UCF-101_train_seq'
path_features='D:\\Major-Project\\CNNnLSTM\\Project\\UCF-101_train_features'
size=40
count=0

print('Loading Model....')
base_model=InceptionV3(weights='imagenet',include_top=True)
model = Model(inputs=base_model.input,outputs=base_model.get_layer('avg_pool').output)
print('Model Loaded!')

def extract(image_path,model):
	img=image.load_img(image_path,target_size=(299,299))
	x=image.img_to_array(img)
	x=np.expand_dims(x,axis=0)
	x=preprocess_input(x)
	features=model.predict(x)

	return features[0]



for actname in os.listdir(path_sequence):

	path1=path_sequence+"\\"+actname
	actno=len(fnmatch.filter(os.listdir(path_sequence), '*'))
	if(not os.path.exists(path_features+"\\"+actname)):
		os.makedirs(path_features+"\\"+actname)
	
	for vidfolder in os.listdir(path1):
		
		path2=path1+"\\"+vidfolder
		vidno=len(fnmatch.filter(os.listdir(path1), '*'))
		path3=path_features+'/'+actname+'/'+vidfolder

		length=len(fnmatch.filter(os.listdir(path2), '*.jpg'))
		total = (length if length<size else size)
		pbar = tqdm(total=total)
		sequence=[]
		count=0
		skip=1
		img_count=0
		old_time = time()
		for video_img in os.listdir(path2):
		
			img_path=path2+"\\"+video_img

			if length<size:
				sequence.append(extract(img_path,model))
				pbar.update(1)
				img_count+=1
			else:
				count+=1
				skip=length//size
				if count%skip==0:
					sequence.append(extract(img_path,model))
					pbar.update(1)
					img_count+=1
				else:
					continue
			if(img_count==40):break

		np.save(path3,sequence)
		new_time = time()
		print('Done in {} secs!'.format(new_time - old_time))
		break
	break

pbar.close()