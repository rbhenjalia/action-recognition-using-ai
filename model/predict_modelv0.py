# from keras.model import load_model
import numpy as np 
from keras.models import model_from_json
from time import time
import operator

start_time=time()

classes = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo']
json_file=open('/home/abhishek/Major Project/modele5.json','r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)


loaded_model.load_weights("/home/abhishek/Major Project/weights1e5.hdf5")

print("MODEL LOADED")
sample=np.load("/home/abhishek/Major Project/UCF-101_train_features/Hammering/v_Hammering_g13_c02.npy")
# sample = np.expand_dims(sample,axis=0)
sample=np.array(sample)
sample=np.reshape(sample,(sample.shape[0],1,sample.shape[1]))
print(sample.shape,"SAMPLE SHAPE")
print("predicting")
prediction=loaded_model.predict(sample)

sum1=prediction[0]
for i in range(1,len(prediction)):
	sum1=np.add(sum1,prediction[i])

sum1=np.array(sum1)
sum1=sum1/40

print("Predicted class",classes[np.argmax(sum1)])

# print(sum1,"PREDICTION ARRAY")




# label_predictions = {}
# for i, label in enumerate(classes):
# 	label_predictions[label] = sum1[i]



# sorted_lps = sorted(label_predictions.items(),key=operator.itemgetter(1),reverse=True)

# for i, class_prediction in enumerate(sorted_lps):
# 	# print(i,"I")
# 	# print(class_prediction,"CLASS PREDICTION")
# 	if i > 5 or class_prediction[1] == 0.0:
# 		break
# 	print("%s: %.2f" % (class_prediction[0], class_prediction[1]))

end_time=time()
print("TOTAL TIME TAKEN",end_time-start_time)
