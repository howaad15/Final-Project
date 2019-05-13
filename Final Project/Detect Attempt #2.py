import cv2
import numpy as np
import math
import os
from objloader_simple import *

MIN_MATCHES = 15


def main():
    homog = None

    camera_parameters = np.load('webcam-matrixParams_mbp.npz')
    camera_parameters = camera_parameters['mtx']

    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher()

    dir_name = '/Users/amanda/PycharmProjects/Final Project/Data/'
    model = cv2.imread('/Users/amanda/PycharmProjects/Final Project/Monkey-3.png', 0)
    kp_model, des_model = sift.detectAndCompute(model, None)
    obj = OBJ(os.path.join(dir_name, 'blue-dog.obj'))

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()
        cv2.imshow("test", frame)

        if not ret:
            print("camera sucks")
            return

        kp_frame, des_frame = sift.detectAndCompute(frame, None)
        matches = bf.knnMatch(des_model, des_frame, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        matches = good

        if len(matches) > MIN_MATCHES:

            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            homog, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            h, w = model.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, homog)
            frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            if homog is not None:
                try:
                    proj = projection_matrix(camera_parameters, homog)
                    frame = render(frame, obj, proj, model, False)
                except:
                    pass

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print("Not enough matches found - %d/%d" % (len(matches), MIN_MATCHES))

    cap.release()
    cv2.destroyAllWindows()
    return 0


def render(img, obj, projection, model, color=False):

    vertices = obj.vertecies
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.face:
        face=obj.face
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

        return img


def projection_matrix(camera_parameters, homog):

    homog = homog * (-1)

    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homog)

    col1 = rot_and_transl[:, 0]
    col2 = rot_and_transl[:, 1]
    col3 = rot_and_transl[:, 2]

    l = math.sqrt(np.linalg.norm(col1, 2) * np.linalg.norm(col2, 2))

    rot1 = col1 / l
    rot2 = col2 / l

    translation = col3 / l

    c = rot1 + rot2
    p = np.cross(rot1, rot2)
    d = np.cross(c, p)

    rot1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot3 = np.cross(rot1, rot2)

    proj = np.stack((rot1, rot2, rot3, translation)).T
    return np.dot(camera_parameters, proj)


def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


if __name__ == '__main__':
    main()
