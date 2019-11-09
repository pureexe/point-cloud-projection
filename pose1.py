import cv2
import numpy as np
import matplotlib.pyplot as plt
def plot_pose_box(image, P, kpt, color=(0, 255, 0), line_width=2):
    ''' Draw a 3D box as annotation of pose. Ref:https://github.com/yinguobing/head-pose-estimation/blob/master/pose_estimator.py
    Args: 
        image: the input image
        P: (3, 4). Affine Camera Matrix.
        kpt: (68, 3).
    '''
    image = image.copy()

    point_3d = []
    rear_size = 90
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 105
    front_depth = 110
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d image points
    point_3d_homo = np.hstack((point_3d, np.ones([point_3d.shape[0],1]))) #n x 4
    point_2d = point_3d_homo.dot(P.T)[:,:2]
    print(np.mean(point_2d[:4,:2], 0))
    point_2d[:,:2] = point_2d[:,:2] - np.mean(point_2d[:4,:2], 0) + np.mean(kpt[:27,:2], 0)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)

    return image

image = plt.imread('data/0004.jpg')
kpt = np.loadtxt('data/0004_kpt_ori.txt')
P = np.loadtxt('data/0004_camera_matrix_ori.txt')


plt.imshow(plot_pose_box(image,P,kpt))
plt.show()