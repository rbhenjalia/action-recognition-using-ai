# project_path1 = 'S:\\meet_pillai\\Project'
# project_path = 'D:\\Major-Project\\CNNnLSTM\\Project'
# import csv
# from keras.callbacks import ModelCheckpoint
# from keras.preprocessing.image import ImageDataGenerator
# from collections import deque
# import sys
# from tqdm import tqdm
# from keras.models import model_from_json


import numpy as np
from keras.layers import Dense, Dropout, Flatten  # , ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential # , load_model
import random
from pprint import pprint
from keras.utils import to_categorical
from keras import metrics

laptop = 'harsh' # or 'harsh' or 'pillai'


if laptop=='meet':
	project_path = 'D:\\Major-Project\\CNNnLSTM\\Project'
	classname_index = 5
elif laptop=='harsh':
	project_path = 'S:\\meet_pillai\\Project'
	classname_index = 4
elif laptop=='pillai':
	# remove [None] with project path & classname index in pillai's laptop
	project_path=None 
	classname_index=None

dataset_path = project_path + '\\dataset'

classes = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo']
classes = sorted(classes)



def label_encode(path):
	path=str(path).split("\\")
	encode=classes.index(path[classname_index])
	one_hot=to_categorical(encode,len(classes))
	return one_hot


def feature_generator(data, batch_size):
	while 1:
		x, y = [], []
		for _ in range(batch_size):
			sequence = None
			sample = random.choice(data)
			sample = sample[:-1]
			sequence = np.load(sample)
			x.extend(sequence)
			# print(sequence.shape)
			tmp_y = [label_encode(sample) for i in range(sequence.shape[0])]
			y.extend(tmp_y)

		x = np.array(x)
		# print(x.shape)
		# print(y.shape)
		x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
		y = np.array(y)
		yield x, y





print('\n[INFO] : Reading MetaData....')

with open(project_path + '\\train_address.txt','r') as fin:
	train_data=fin.readlines()
	# train_data=list(reader)

with open(project_path + '\\test_address.txt','r') as fin1:
	test_data=fin1.readlines()

print('[INFO] : MetaData Read Complete.!!')
# print('')


# print('\nPreparing train and test input dataset...')

# train_pbar = tqdm(total=len(train_data))
# train_pbar.set_description('Train')

# X,y=[],[]
# for row in train_data:
# 	row=row[:-1]
# 	sequence=np.load(row)
# 	X.append(sequence)
# 	y.append(label_encode(row))
# 	train_pbar.update(1)
# train_pbar.close()

# print('')
# test_pbar = tqdm(total=len(test_data))
# test_pbar.set_description('Test')

# X_test,y_test=[],[]
# for row in test_data:
# 	row = row[:-1]
# 	sequence=np.load(row)
# 	X_test.append(sequence)
# 	y_test.append(label_encode(row))
# 	test_pbar.update(1)
# test_pbar.close()

# print('\nTrain and test input dataset prepared.!')





print('\n[INFO] : Defining and Compiling Model...\n')

nb_classes=len(classes)
batch_size=32
steps_per_epoch_train= len(train_data) // batch_size
steps_per_epoch_test= len(test_data) // batch_size
nb_epoch = 5


model=Sequential()
model.add(LSTM(2048,return_sequences=False,input_shape=(1,2048),dropout=0.2))
# model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

print('\n[INFO] : Model Definition and Compilation Done.!')


train_generator = feature_generator(train_data,batch_size)
val_generator = feature_generator(test_data,batch_size)



print('\n[INFO] : Training Model...\n')


model.fit_generator(
    generator=train_generator, # feature_generator(train_data,batch_size),  # 
    steps_per_epoch=steps_per_epoch_train,
    epochs=nb_epoch,
    verbose=1,
    # callbacks=[tb, early_stopper, csv_logger, checkpointer],
    validation_data=val_generator, # feature_generator(test_data,batch_size),  # 
    validation_steps=steps_per_epoch_test,
    workers=4,
	# use_multiprocessing=True
    )

print('\n[INFO] : Model Trained.!')

# model.fit(X,y,
# 	batch_size=30,
# 	validation_data=(X_test,y_test),
# 	verbose=1,
# 	epochs=3)



print('\n[INFO] : Saving Model.!')

model_json = model.to_json()
with open("modele5.json", "w") as json_file:
	json_file.write(model_json)

fname="weights1e5.hdf5"
model.save_weights(fname,overwrite=True)
	
print('\n[INFO] : Model Saved.!')
print('\n\n\n!.DONE.!')