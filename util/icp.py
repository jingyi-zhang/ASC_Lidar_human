import math
import numpy as np
from sklearn.neighbors import NearestNeighbors


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    _, indices = neigh.kneighbors(src, return_distance=True)
    return indices.ravel()


def icp(A, B, max_iterations=20):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    res = np.identity(m + 1)

    initial_offset = np.mean(B, axis=0) - np.mean(A, axis=0)
    A += initial_offset
    res[:m, m] = initial_offset

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        T[2][3] = 0.0  # Tz = 0
        if T[2][0] < 1.0:
            theta = -math.asin(T[2][0])
            cos_theta = math.cos(theta)
            phi = math.atan2(-T[1][0] / cos_theta, T[0][0] / cos_theta)
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)
            T[0][0] = T[1][1] = cos_phi
            T[1][0] = sin_phi
            T[0][1] = -sin_phi
            T[0][2] = T[1][2] = T[2][0] = T[2][1] = 0
            T[2][2] = 1

        # update the current source
        src = np.dot(T, src)
        res = np.dot(T, res)

    # calculate final transformation
    return res.T
