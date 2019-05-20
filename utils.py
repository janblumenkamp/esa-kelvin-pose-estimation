import numpy as np
import json
import os
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
from matplotlib.collections import PolyCollection

class Camera:

    """" Utility class for accessing camera parameters. """

    fx = 0.0176  # focal length[m]
    fy = 0.0176  # focal length[m]
    nu = 1920  # number of horizontal[pixels]
    nv = 1200  # number of vertical[pixels]
    ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
    ppy = ppx  # vertical pixel pitch[m / pixel]
    fpx = fx / ppx  # horizontal focal length[pixels]
    fpy = fy / ppy  # vertical focal length[pixels]
    k = [[fpx,   0, nu / 2],
         [0,   fpy, nv / 2],
         [0,     0,      1]]
    K = np.array(k)


def process_json_dataset(root_dir):
    with open(os.path.join(root_dir, 'train.json'), 'r') as f:
        train_images_labels = json.load(f)

    with open(os.path.join(root_dir, 'test.json'), 'r') as f:
        test_image_list = json.load(f)

    with open(os.path.join(root_dir, 'real_test.json'), 'r') as f:
        real_test_image_list = json.load(f)

    partitions = {'test': [], 'train': [], 'real_test': []}
    labels = {}

    for image_ann in train_images_labels:
        partitions['train'].append(image_ann['filename'])
        labels[image_ann['filename']] = {'q': image_ann['q_vbs2tango'], 'r': image_ann['r_Vo2To_vbs_true']}

    for image in test_image_list:
        partitions['test'].append(image['filename'])

    for image in real_test_image_list:
        partitions['real_test'].append(image['filename'])

    return partitions, labels


def quat2dcm(q):

    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm

def pointInTriangle(t, p):
    #https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
    a = 0.5 *(-t[1][1]*t[2][0] + t[0][1]*(-t[1][0] + t[2][0]) + t[0][0]*(t[1][1] - t[2][1]) + t[1][0]*t[2][1]);
    s = 1/(2*a)*(t[0][1]*t[2][0] - t[0][0]*t[2][1] + (t[2][1] - t[0][1])*p[0] + (t[0][0] - t[2][0])*p[1]);
    u = 1/(2*a)*(t[0][0]*t[1][1] - t[0][1]*t[1][0] + (t[0][1] - t[1][1])*p[0] + (t[1][0] - t[0][0])*p[1]);
    return s > 0 and u > 0 and (1-s-u) > 0
    
class Plane:
    def __init__(self, points):
        if len(points) != 3:
            raise ValueError("Plane always consists of three points")
        
        self.points = np.asarray(points)
        n = np.cross(points[1] - points[0], points[2] - points[0])
        self.normal = n / np.linalg.norm(n, 2)
    
    def intersect(self, v):
        ndotu = self.normal.dot(v)
        if abs(ndotu) < 1e-6:
            raise ValueError("Line is parallel to plane")
 
        return -self.points[0] - (self.normal.dot(-self.points[0]) / ndotu) * v + self.points[0]
    
    def intersects(self, v):
        # First calculate intersection point of vector and this plane:
        try:
            intersection = self.intersect(v)
        except ValueError:
            # The vector is parallel to the line, so always return false (could be completely on it or completely off)
            return False
        
        # We have a rotated 3D plane (i.e. z coordinates are level) and want to remove the
        # z coordinates while keeping relations (i.e. project 3D plane to xy-plane)
        # https://stackoverflow.com/questions/1023948/rotate-normal-vector-onto-axis-plane
        zAxisNew = self.normal
        xAxisOld = np.array([1,0,0])
        if np.array_equal(np.absolute(zAxisNew), xAxisOld):
            # the old x axis cannot be the same as the normal (the new z axis) since then the
            # coordinate system is perpendicular to the xy plane. Therefore change x and z then
            xAxisOld = np.array([0,0,1])
        yAxisOld = np.array([0,1,0])
        yAxisNew = np.cross(xAxisOld, zAxisNew)
        xAxisNew = np.cross(zAxisNew, yAxisNew)
        yAxisNew /= np.linalg.norm(yAxisNew, 2)
        xAxisNew /= np.linalg.norm(xAxisNew, 2)
        projected2dtriangle = np.asarray([[p.dot(xAxisNew), p.dot(yAxisNew)] for p in self.points])
        # Now we know the 2d projection of the points of the polygon. Also project the 3d intersection point to the same plane
        projected2dpoint = np.asarray([intersection.dot(xAxisNew), intersection.dot(yAxisNew)])
        return intersection, pointInTriangle(projected2dtriangle, projected2dpoint)

def getSatelliteModel():
    b = 0.6
    a = 0.75
    d = 0.8
    c = 0.32

    #     0         1      
    #     +---a-----+
    #  d-/|   u    /|-c
    # 3 +---------+ | 2
    #   |w| y  z  |x|     (y: front, z: back)
    # 4 | +-------|-+ 5
    #   |/ v (0,0)|/-b 
    # 7 +---------+ 6
    # reference points in satellite frame for drawing axes
    return np.array([
        [-a / 2,  d / 2, c], # 0
        [ a / 2,  d / 2, c], # 1
        [ a / 2, -d / 2, c], # 2
        [-a / 2, -d / 2, c], # 3
        [-a / 2,  b / 2, 0], # 4
        [ a / 2,  b / 2, 0], # 5
        [ a / 2, -b / 2, 0], # 6
        [-a / 2, -b / 2, 0]  # 7
    ]), np.array([
        [0, 1, 2], [0, 3, 2], # u
        [4, 5, 6], [4, 7, 6], # v
        [0, 3, 7], [0, 4, 7], # w
        [1, 2, 6], [1, 5, 6], # x
        [3, 2, 6], [3, 7, 6], # y
        [0, 1, 5], [0, 4, 5], # z
    ])

def projectModel(q, r, plot=False):
    """ Projecting points to image frame to draw axes """
    model_coordinates, cube_polygon_indices = getSatelliteModel()
    p_axes = np.ones((model_coordinates.shape[0], model_coordinates.shape[1] + 1))
    p_axes[:,:-1] = model_coordinates
    points_body = np.transpose(p_axes)

    # transformation to camera frame
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
    p_cam = np.dot(pose_mat, points_body)

    # Indices of points describing 3 point triangles of the cube
    # No point should intersect any of these triangles to be visible in the camera

    if plot:
        fig = plt.figure()
        ax = Axes3D(fig)

    points_camera_t = p_cam.transpose()
    points_camera_collision_indices = []
    for polygon_indices in cube_polygon_indices:
        points_polygon = points_camera_t[polygon_indices]
        plane = Plane(points_polygon)

        if plot:
            tri = a3.art3d.Poly3DCollection([plane.points], alpha=0.2)
            tri.set_color([1,0,0])
            tri.set_edgecolor('k')
            ax.add_collection3d(tri)

        for i, p in enumerate(points_camera_t):
            intersection, intersects = plane.intersects(p)
            if(intersects):
                # The vector between camera origin and cube vertice intersects any of the 12 cube polygons.
                # There are two border cases to check:
                # 1) Sometimes an actual vertice intersects a neighboring polygon
                # 2) The vector between camera and point intersects a polygon that actually is behind the point
                dist_intersection = np.linalg.norm(intersection, 2)
                dist_point = np.linalg.norm(p, 2)
                if abs(dist_intersection - dist_point) > 0.01 and dist_intersection < dist_point and not i in points_camera_collision_indices:
                    points_camera_collision_indices.append(i)
                    if plot:
                        ax.scatter([intersection[0]], [intersection[1]], [intersection[2]])

    visible_points = np.ones(len(p_axes), dtype=bool)
    visible_points[points_camera_collision_indices] = False

    if plot:
        for p in points_camera_t[visible_points]:
            ax.plot([0, p[0]], [0, p[1]], [0, p[2]])

        #ax.set_xlim(-1, 1)
        #ax.set_ylim(-1, 1)
        #ax.set_zlim(5, 7)
        ax.autoscale()
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        plt.show()

    p_cam = points_camera_t.transpose()

    # getting homogeneous coordinates
    points_camera_frame = p_cam / p_cam[2]
    # projection to image plane
    points_image_plane = Camera.K.dot(points_camera_frame)

    x, y = (points_image_plane[0], points_image_plane[1])
    return x, y, visible_points

def projectAxes(q, r):

    """ Projecting points to image frame to draw axes """

    # reference points in satellite frame for drawing axes
    p_axes = np.array([[0, 0, 0, 1],
                       [1, 0, 0, 1],
                       [0, 1, 0, 1],
                       [0, 0, 1, 1]])
    points_body = np.transpose(p_axes)

    # transformation to camera frame
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
    p_cam = np.dot(pose_mat, points_body)

    # getting homogeneous coordinates
    points_camera_frame = p_cam / p_cam[2]

    # projection to image plane
    points_image_plane = Camera.K.dot(points_camera_frame)

    x, y = (points_image_plane[0], points_image_plane[1])
    return x, y


class SatellitePoseEstimationDataset:

    """ Class for dataset inspection: easily accessing single images, and corresponding ground truth pose data. """

    def __init__(self, root_dir='/datasets/speed_debug'):
        self.partitions, self.labels = process_json_dataset(root_dir)
        self.root_dir = root_dir

    def get_image(self, i=0, split='train'):

        """ Loading image as PIL image. """

        img_name = self.partitions[split][i]
        img_name = os.path.join(self.root_dir, 'images', split, img_name)
        image = Image.open(img_name).convert('RGB')
        return image

    def get_pose(self, i=0):

        """ Getting pose label for image. """

        img_id = self.partitions['train'][i]
        q, r = self.labels[img_id]['q'], self.labels[img_id]['r']
        return q, r

    def visualize(self, i, partition='train', ax=None):

        """ Visualizing image, with ground truth pose with axes projected to training image. """

        if ax is None:
            ax = plt.gca()
        img = self.get_image(i)
        ax.imshow(img)

        # no pose label for test
        if partition == 'train':
            q, r = self.get_pose(i)
            xa, ya = projectAxes(q, r)
            ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=30, color='r')
            ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=30, color='g')
            ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=30, color='b')

        return
