import numpy as np
import cv2

blankframe = np.zeros((512,512,3), np.uint8)

label = 'HUMAN-MEET'
x1 = 100
y1 = 100
x2 = 400
y2 = 200

cv2.rectangle(blankframe, (x1,y1), (x2,y2), (0,255,255), 2)
cv2.rectangle(blankframe, (x1,y1-20), ((x1 + len(label)*11),y1), (0,255,255), -1)
font = cv2.FONT_HERSHEY_PLAIN
cv2.putText(
		img=blankframe,
		text=label,
		org=(x1,y1-5),
		fontFace=font,
		fontScale=1,
		color=(0,0,0),
		thickness=1,
		lineType=cv2.LINE_AA
	)

cv2.imshow('Label-Box', blankframe)

cv2.waitKey(0)
cv2.destroyAllWindows()
