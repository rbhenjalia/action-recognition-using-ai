import numpy as np
import cv2

## capture video from camera
# cap = cv2.VideoCapture(0)

# if(not cap.isOpened()): cap.open()

# while(True):
# 	ret, frame = cap.read()
# 	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 	cv2.imshow('frame',gray)
# 	if cv2.waitKey(1) & 0xFF == ord('q'): break

# cap.release()
# cv2.destroyAllWindows()


## playing video from file
# cap = cv2.VideoCapture('../test.mp4')

# frate = cap.get(5)

# while(cap.isOpened()):
# 	ret, vframe = cap.read()
# 	if(ret):
# 		cv2.imshow('video-frame',vframe)
# 	else:
# 		break
# 	if cv2.waitKey(int(10**3/frate)) & 0xFF == ord('q'): break # int(10**3/frate)

# cap.release()
# cv2.destroyAllWindows()


## saving webcam stream to video file
# cap = cv2.VideoCapture(0)

# # define the codec and create video writer object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('../webcam-output.avi', fourcc, 20.0, (640,480))

# while(cap.isOpened()):
# 	ret, frame = cap.read()
# 	if(ret):
# 		frame = cv2.flip(frame,1)

# 		# write flipped frame to output file
# 		out.write(frame)

# 		# display frame
# 		cv2.imshow('Live Stream',frame)

# 	if(cv2.waitKey(1)) & 0xFF == ord('q'): break

# cap.release()
# out.release()
# cv2.destroyAllWindows()