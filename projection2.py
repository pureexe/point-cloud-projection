###
# Render work
# But without extrinsic paramter
#

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


mat = loadmat('data/0208_mesh_rightfar.mat')
color = mat['colors']
vertex = mat['vertices']
camera_matrix = np.loadtxt('data/0208_camera_matrix_rightfar.txt')
canonical = np.load('data/canonical_vertices_righthand_far.npy')
vertices_homo = np.hstack((vertex, np.ones([vertex.shape[0],1]))) #n x 4
P = np.linalg.lstsq(vertices_homo, canonical)[0].T # Affine matrix. 3 x 4
front_vertices = vertices_homo.dot(P.T)

vertex = front_vertices

# Extensic
extrinsic = camera_matrix

#Intrinsic
focal = 2000*128
intrinsic = np.array([
    [focal, 0, 128],
    [0, focal, 128],
    [0, 0, 1]
])

#Add 1 into vextex last channel
position = np.ones([vertex.shape[0], 4])
position[:,:3] = vertex

# APPLY CAMERA MODEL
#projected = np.matmul(intrinsic,np.matmul(extrinsic,position.transpose()))
projected = np.matmul(intrinsic,position.transpose()[:3])


#NORMALLIZE
projected = (projected / projected[2,:]).T
projected = np.round(projected).astype(np.int32)
image = np.zeros((256,256,3))
for i in range(len(color)):
    try:
        u,v,_ = projected[i]
        image[v,u,:] = color[i]
    except:
        pass
plt.imshow(image)
plt.show()

