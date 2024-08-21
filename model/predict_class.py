from time import time
print("\n[INFO] Importing Modules and Declaring Global variables...\n")
old = time()
import numpy as np
import os
import fnmatch
from subprocess import call

from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
from keras.models import model_from_json
new = time()
print('Time : %.2f' % (new - old))

laptop = 'meet' # or 'harsh' or 'pillai'


if laptop=='meet':
	project_path = 'D:\\Major-Project\\CNNnLSTM\\Project'
elif laptop=='harsh':
	project_path = 'S:\\meet_pillai\\Project'
elif laptop=='pillai':
	# remove [None] with project path in pillai's laptop
	project_path=None

lstm_json_path = os.path.join(project_path,'Model','modele5.json')
lstm_hdf5_path = os.path.join(project_path,'Model','weights1e5.hdf5')
pred_path = os.path.join(project_path,'predict')


classes = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo']



def extract(image_path,model):
	img=image.load_img(image_path,target_size=(299,299))
	x=image.img_to_array(img)
	x=np.expand_dims(x,axis=0)
	x=preprocess_input(x)
	features=model.predict(x)
	return features[0]



print("\n[INFO] Loading InceptionV3 Model...\n")

old = time()
base_model=InceptionV3(weights='imagenet',include_top=True)
model = Model(inputs=base_model.input,outputs=base_model.get_layer('avg_pool').output)
new = time()
print('Time : %.2f' % (new - old))



print("\n[INFO] Loading LSTM Model...\n")

old = time()
json_file=open(lstm_json_path,'r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)
loaded_model.load_weights(lstm_hdf5_path)
new = time()
print('Time : %.2f' % (new - old))


while True:
	filename = input('Enter FileName : ')

	filename_no_ext = filename.split('.')[0]
	src = os.path.join(pred_path,filename)
	dest_dir = os.path.join(pred_path,'sequence',filename_no_ext)
	if(not os.path.exists(dest_dir)):
		os.makedirs(dest_dir)
	dest = os.path.join(dest_dir, filename_no_ext+'-%04d.jpg')



	print("\n[INFO] Spliting Video into Frames...\n")

	old = time()
	call(['ffmpeg', '-i', src, dest, '-hide_banner'])
	new = time()
	print('Time : %.2f' % (new - old))


	print("\n[INFO] Generating Spatial Features...\n")

	old = time()
	size = 40
	length=len(fnmatch.filter(os.listdir(dest_dir), '*.jpg'))

	sequence = []
	count=0
	skip=1
	img_count=0

	for video_img in os.listdir(dest_dir):
		img_path=os.path.join(dest_dir,video_img)
		if length<size:
			sequence.append(extract(img_path,model))
			img_count+=1
		else:
			count+=1
			skip=length//size
			if count%skip==0:
				sequence.append(extract(img_path,model))
				img_count+=1
			else:
				continue
		if(img_count==40):break
	new = time()
	print('Time : %.2f' % (new - old))


	print("\n[INFO] Generating Temporal Features...\n")

	sample=np.array(sequence)
	sample=np.reshape(sample,(sample.shape[0],1,sample.shape[1]))
	print(sample.shape,"SAMPLE SHAPE")

	print("\n[INFO] Predicting Final Class...\n")

	old = time()
	prediction=loaded_model.predict(sample)
	new = time()
	print('Time : %.2f' % (new - old))

	sum1=prediction[0]
	for i in range(1,len(prediction)):
		sum1=np.add(sum1,prediction[i])

	sum1=np.array(sum1)
	sum1=sum1/40

	print("Predicted class : ",classes[np.argmax(sum1)],"\n")



	decision = input("Do you want to continue Prediction? [y/n] : ")
	if decision== 'n':
		break
