import numpy as np
import cv2
from PIL import Image
import random

from descriptors import myLocalDescriptor

def descriptorMatching(points1, points2, percentageThreshold):
    im1 = Image.open("im1.png")
    im1 = np.asarray(im1)
    I1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)

    im2 = Image.open("im2.png")
    im2 = np.asarray(im2)
    I2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

    rhom = 5
    rhoM = 105
    rhostep = 2
    N = 180

    desc_len = int((rhoM - rhom) / rhostep)

    # calculate euclidean distance between every possible point descriptor pair 
    p1 = points1.shape[0]
    p2 = points2.shape[0]

    # precalculate descriptors for speed
    descriptors_a = np.zeros((p1, desc_len))
    descriptors_b = np.zeros((p2, desc_len))

    for x1 in range(p1):
        point_a = points1[x1]
        descriptor = myLocalDescriptor(I1, point_a, rhom, rhoM, rhostep, N)
        if descriptor is None:
            descriptor = np.full((desc_len, 1), np.inf)
        descriptors_a[x1] = np.squeeze(np.array(descriptor))

    for x2 in range(p2):
        point_b = points2[x2]
        descriptor = myLocalDescriptor(I2, point_b, rhom, rhoM, rhostep, N)
        if descriptor is None:
            descriptor = np.full((desc_len, 1), np.inf)
        descriptors_b[x2] = np.squeeze(np.array(descriptor))

    # euclidean distance matrix
    # in flattened array form
    edm = np.zeros((p1 * p2, 3))

    for x1 in range(p1):
        for x2 in range(p2):
            point_a = points1[x1]
            point_b = points2[x2]

            # get descriptors for both points
            descriptor_a = descriptors_a[x1]
            descriptor_b = descriptors_b[x2]

            inf = sum(np.isinf(descriptor_a)) + sum(np.isinf(descriptor_b))
            if inf > 0:
                dist = np.inf
            else:
                # avoid square root: take distance squared
                descriptor_diff = np.array(descriptor_a) - np.array(descriptor_b)
                dist = np.sum(descriptor_diff * descriptor_diff)

            idx = x1 + x2 * p1
            edm[idx][0] = dist
            edm[idx][1] = x1
            edm[idx][2] = x2

    # take percentage threshold (given as a number between 0 and 100) smaller distances
    smaller_dists = round(p1 * p2 * percentageThreshold / 100.0)

    sorted_dists = sorted(edm, key=lambda x: x[0])

    points = sorted_dists[0:smaller_dists]

    points = list(map(lambda x: [x[1], x[2]], points))

    return np.array(points)

def myRANSAC(points1, points2, matchingPoints, r, N):
    # perform algorithm
    number_of_points = matchingPoints.shape[0]
    scores = np.zeros((N, 4)) # keep score, theta, d1, d2

    for i in range(N):
        # pick two point pairs at random
        points = random.sample(range(number_of_points), 2)
        point_pair_a = matchingPoints[points[0]]
        point_pair_b = matchingPoints[points[1]]

        # determine theta, d
        p1 = np.array([points1[int(point_pair_a[0])], points2[int(point_pair_a[1])]])
        p2 = np.array([points1[int(point_pair_b[0])], points2[int(point_pair_b[1])]])

        center1 = np.mean(p1, axis=0)
        center2 = np.mean(p2, axis=0)

        p1_centered = p1 - center1
        p2_centered = p2 - center2

        cov = np.dot(p2_centered.T, p1_centered)
        U, _, Vt = np.linalg.svd(cov)
        d = np.dot(Vt.T, U)
        d = np.linalg.det(d)
        if d >= 0:
            d = 1
        else:   
            d = -1
        R = np.dot(Vt.T, np.dot(np.array([[1, 0], [0, d]]), U.T))
        d = center2 - np.dot(R, center1)

        # determine score
        score_iter = 0
        for j in range(number_of_points):
            point_pair = matchingPoints[j]
            pa = points1[int(point_pair[0])]
            pb = points1[int(point_pair[1])]

            pb_est = np.dot(R, pa) + d

            # again, use distance squared instead of distance
            dist = (pb[0] - pb_est[0]) * (pb[0] - pb_est[0]) + (pb[1] - pb_est[1]) * (pb[1] - pb_est[1])
            score_iter += dist

        # add to array
        costheta = R[0][0]
        sintheta = R[1][0]

        theta = np.arctan2(sintheta, costheta)

        scores[i][0] = score_iter
        scores[i][1] = theta
        scores[i][2] = d[0]
        scores[i][3] = d[1]

    # find best theta, d
    best_transform = min(scores, key=lambda x: x[0])

    theta = best_transform[1]
    d = np.array([best_transform[2], best_transform[3]])

    H = {"theta": theta, "d": d}

    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    R = np.array([[costheta, -sintheta], [sintheta, costheta]])

    # match every point, classify as inliers or outliers
    rsquared = r * r

    inliers = []
    outliers = []

    for i in range(number_of_points):
        point_pair = matchingPoints[i]
        pa = points1[int(point_pair[0])]
        pb = points2[int(point_pair[1])]

        pb_est = np.dot(R, pa) + d

        # again, use distance squared instead of distance
        dist = np.sum((pb - pb_est) * (pb - pb_est))

        if dist <= rsquared:
            inliers.append(i)
        else:
            outliers.append(i)

    return (H, inliers, outliers)