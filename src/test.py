import numpy as np 
import linear_combination as lc
import cv2 as cv

x = np.array([[1,2,3],
              [2,3,4],
              [2,4,5],
              [4,3,2],
              [2,3,5],
              [1,2,4]]).astype(float)
y = np.array([[1],[2],[3],[4],[5],[6]]).astype(float)
sol = np.linalg.lstsq(x,y)[0]
x = (sol[0] < 0)

z = np.zeros((3))
print(z)