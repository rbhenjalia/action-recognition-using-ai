import numpy as np
import cv2

blankframe = np.zeros((512,512,3), np.uint8)

# cv2.line(blankframe, (0,0), (511,511), (0,255,0), 5)
# cv2.rectangle(blankframe, (100,100), (200,300), (0,255,0), 1)
# cv2.circle(blankframe, (300,300), 100, (0,0,255), 1)
# cv2.ellipse(blankframe, (256,256), (100,50), 0, 0, 270, 255, -1)
# cv2.ellipse(blankframe, (100,100), (100,50), 0, 45, 270, 255, -1)

# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(
# 	img=blankframe,
# 	text='OpenCV',
# 	org=(100,256),
# 	fontFace=font,
# 	fontScale=2,
# 	color=(255,255,255),
# 	thickness=2,
# 	lineType=cv2.LINE_AA
# 	)



cv2.imshow('Drawing-frame',blankframe)

cv2.waitKey(0)
cv2.destroyAllWindows()