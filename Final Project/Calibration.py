import numpy as np
import cv2
import glob
from imutils import paths

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

cbrow = 6
cbcol = 9

# prepare 3D object points for a square chessboard: (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cbrow * cbcol, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

pathname = '/Users/amanda/Desktop/Final Project Data/Checkerboard/Camera'

for fname in paths.list_images(pathname):
    print('Processing image: ' + fname)
    img = cv2.imread(fname)

    img = cv2.resize(img, None, fx=0.25, fy=0.25)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (cbcol, cbrow), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print('\tAccepted image: ' + fname)
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        #cv2.imshow(fname, img)
        #cv2.waitKey(0)  # Press any key on the image window to continue

cv2.destroyAllWindows()

# Estimate the camera calibration matrix: mtx
# The distortion coefficients: dist
# The rotation and translation parameters: rvecs, tvecs
#ret, corners = cv2.findChessboardCorners(gray, (cbcol, cbrow), None)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("camera matrix:")
print(mtx)
print("distortion coefficients:")
print(dist)

img = cv2.imread('/Users/amanda/Desktop/Final Project Data/Checkerboard/Web Cam/opencv_frame_0.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, None, fx=0.25, fy=0.25)
h,  w = img.shape[:2]

# Refine the camera matrix and obtain a "valid" region of interest
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

print("new camera matrix:")
print(newcameramtx)

# undistort, use the refined camera matrix newcameramtx
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
#cv2.imshow('original', img)
#cv2.imshow('corrected', dst)
#cv2.waitKey(0)

# Calculate the reprojection error. The smaller the value the more
# accurate the estimated parameters are.
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: " + str(mean_error/len(objpoints)))

# Save the camera calibration parameters
np.savez("webcam-matrixParams_mbp.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)