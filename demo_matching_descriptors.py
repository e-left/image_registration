import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

from corners import myDetectHarrisFeatures
from matching_descriptors import descriptorMatching, myRANSAC

image1 = Image.open("im1.png")
image_data1 = np.asarray(image1)
print(image_data1.shape)
grayscale_image_data1 = cv2.cvtColor(image_data1, cv2.COLOR_RGB2GRAY)
corners_a = myDetectHarrisFeatures(grayscale_image_data1)

image2 = Image.open("im2.png")
image_data2 = np.asarray(image2)
grayscale_image_data2 = cv2.cvtColor(image_data2, cv2.COLOR_RGB2GRAY)
corners_b = myDetectHarrisFeatures(grayscale_image_data2)

print(f"{corners_a.shape[0]} points identified in picture 1")
print(f"{corners_b.shape[0]} points identified in picture 2")

matchingPoints = descriptorMatching(corners_a, corners_b, 1)

print(f"In total {len(matchingPoints)} point pairs were kept")

r = 25
(H, inliers, outliers) = myRANSAC(corners_a, corners_b, matchingPoints, r, 100)

print(f"Optimal transform determined: {H}")
print(f"{len(inliers)} point pairs are inliers with radius {r}")
print(f"{len(outliers)} point pairs are outliers with radius {r}")

# find outlier points
outliers_a = []
outliers_b = []
for outlier in outliers:
    point_pair = matchingPoints[int(outlier)]
    point_a = corners_a[int(point_pair[0])]
    point_b = corners_b[int(point_pair[1])]
    outliers_a.append(point_a)
    outliers_b.append(point_b)
outliers_a = np.array(outliers_a).T
outliers_b = np.array(outliers_b).T

# find inlier points
inliers_a = []
inliers_b = []
for inlier in inliers:
    point_pair = matchingPoints[int(inlier)]
    point_a = corners_a[int(point_pair[0])]
    point_b = corners_b[int(point_pair[1])]
    inliers_a.append(point_a)
    inliers_b.append(point_b)
inliers_a = np.array(inliers_a).T
inliers_b = np.array(inliers_b).T

# plot outliers
plt.subplot(2, 1, 1)
plt.imshow(image_data1)
plt.scatter(outliers_a[1], outliers_a[0], marker="s", color="gray", alpha=0.8)
plt.subplot(2, 1, 2)
plt.imshow(image_data2)
plt.scatter(outliers_b[1], outliers_b[0], marker="s", color="gray", alpha=0.8)
plt.savefig("outliers.png")
plt.show()

# plot inliers
cs = matplotlib.cm.rainbow(np.linspace(0, 1, len(inliers)))
plt.subplot(2, 1, 1)
plt.imshow(image_data1)
plt.scatter(inliers_a[1], inliers_a[0], marker="s", alpha=0.8, c=cs)
plt.subplot(2, 1, 2)
plt.imshow(image_data2)
plt.scatter(inliers_b[1], inliers_b[0], marker="s", alpha=0.8, c=cs)
plt.savefig("inliers.png")
plt.show()