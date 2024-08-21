import numpy as np
import cv2

blankframe = np.zeros((512,512,3), np.uint8)

# cv2.ellipse(blankframe, (256,256), (100,100), 270, 45, 360, 255, -1)
# cv2.imshow('Animation-frame', blankframe)
# cv2.waitKey(0)

angle = 0
color = 255
txtstrt = 0
while(1):
	cv2.ellipse(blankframe, (256,256), (100,100), 270, 0, (angle%360), color, -1)
	cv2.imshow('Animation-frame', blankframe)
	angle+=1
	if angle%360 == 0:
		if color==255: color=0
		else: color=255

	if(cv2.waitKey(5)) & 0xFF == ord('q'): break


cv2.destroyAllWindows()
