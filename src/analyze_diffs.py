#!/usr/bin/python3 -i

import numpy as np
import pickle
import cv2
import sys
from matplotlib import pyplot as plt

with open('diffs.pkl', 'rb') as f:
    lbefore, rbefore, lafter, rafter = pickle.load(f)

ldiff = cv2.absdiff(lafter, lbefore)
"""
cv2.imshow("opencv", ldiff)
while (cv2.waitKey(0) != 113):
    pass
"""

rdiff = cv2.absdiff(rafter, rbefore)
"""
cv2.imshow("opencv", rdiff)
while (cv2.waitKey(0) != 113):
    pass
"""

"""
histogram = cv2.calcHist([rdiff], [0], None, [256], [0, 256])
plt.plot(histogram, color='k')
plt.show()
"""

lthresh, limg_threshDiff = cv2.threshold(ldiff, 45, 255, cv2.THRESH_BINARY)
print('lthresh = ', lthresh)
cv2.imwrite('limg_threshDiff.png', limg_threshDiff)

rthresh, rimg_threshDiff = cv2.threshold(rdiff, 45, 255, cv2.THRESH_BINARY)
print('rthresh = ', rthresh)
cv2.imwrite('rimg_threshDiff.png', rimg_threshDiff)
