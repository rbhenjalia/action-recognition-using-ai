from keras.models import model_from_json
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from collections import deque
import sys
import os

nb_classes=101
model=Sequential()
# input_shape doubt hai
model.add(LSTM(2048,return_sequences=False,input_shape=(40,2048),dropout=0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes, activation='softmax')


checkpointer = ModelCheckpoint(
	filepath=os.path.join('Model', 'Checkpoints', 'model-1', 'model-{epoch:03d}-{val_loss:.3f}.hdf5'),
	verbose=1,
	save_best_only=True)


# wrong input
# correct input is 2048 dimensional vector 
# no need to create image generator class and 
# no need for flow_from_directory
aug = ImageDataGenerator(rotation_range=25	, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest", vertical_flip=True)


train_generator=aug.flow_from_directory(
		directory=r"", # directory not available
		target_size=(299,299),
		color_mode="rgb",
		batch_size=32,
		class_mode="categorical",
		shuffle=True,
		seed=None
	)


valid_generator=aug.flow_from_directory(
		directory=r"", #directory not available 
		target_size=(299,299),
		color_mode="rgb",
		batch_size=32,
		class_mode="categorical",
		shuffle=True,
		seed=None
	)


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
						steps_per_epoch=STEP_SIZE_TRAIN,
						validation_data=valid_generator,
						validation_steps=STEP_SIZE_VALID,
						use_multiprocessing=True,
						callbacks=[checkpointer],
						workers=4,
						verbose=1,
						epochs=12

					)


model_json = model.to_json()


# model folder created on drive for reference
json_model_filepath = os.path.join('Model', 'json', 'model1.json')

with open(json_model_filepath, "w") as json_file:
	json_file.write(model_json)

model_weights_filepath = os.path.join('Model', 'weights', 'weights1.hdf5')

model.save_weights(model_weights_filepath,overwrite=True)
	

