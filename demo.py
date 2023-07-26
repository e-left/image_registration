import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from corners import myDetectHarrisFeatures
from matching_descriptors import descriptorMatching, myRANSAC

def imStitch(im1, im2):
    # convert images to arrays and grayscales, keeping the original colored
    image_data1 = np.asarray(im1)
    grayscale_image_data1 = cv2.cvtColor(image_data1, cv2.COLOR_RGB2GRAY)
    image_data2 = np.asarray(im2)
    grayscale_image_data2 = cv2.cvtColor(image_data2, cv2.COLOR_RGB2GRAY)

    print("loaded images")

    # extract corners (salient points, important features)
    corners_a = myDetectHarrisFeatures(grayscale_image_data1)
    corners_b = myDetectHarrisFeatures(grayscale_image_data2)

    print("extracted points")

    # get top 1% pairs
    matchingPoints = descriptorMatching(corners_a, corners_b, 1)

    print("constructed pairs")

    # determine optimal transform for image 2
    # don't care about inliers and outliers
    r = 25
    iterations = 100
    (H, _, _) = myRANSAC(corners_a, corners_b, matchingPoints, r, iterations)

    print("determined transform")

    # apply transform to first image
    # and merge two images

    # calculate size of transformed im1
    # take values of four corners
    theta = H["theta"]
    d = H["d"]
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    R = np.array([[costheta, -sintheta], [sintheta, costheta]])

    width = image_data1.shape[0]
    height = image_data1.shape[1]

    p1 = np.array([0, 0])
    p2 = np.array([0, height])
    p3 = np.array([width, 0])
    p4 = np.array([width, height])
    

    p1 = np.dot(R, p1) + d
    p2 = np.dot(R, p2) + d
    p3 = np.dot(R, p3) + d
    p4 = np.dot(R, p4) + d

    p1 = np.round(p1)
    p2 = np.round(p2)
    p3 = np.round(p3)
    p4 = np.round(p4)

    points = np.vstack([p1, p2, p3, p4])


    max_point = np.max(points, axis=0)
    min_point = np.min(points, axis=0)

    im1_width = image_data1.shape[0]
    im1_height = image_data1.shape[1]
    im2_width = image_data2.shape[0]
    im2_height = image_data2.shape[1]

    # check also with im2 dimensions
    max_x = int(max(max_point[0], im2_width))
    max_y = int(max(max_point[1], im2_height))
    min_x = int(min(min_point[0], 0))
    min_y = int(min(min_point[1], 0))

    width = max_x - min_x
    height = max_y - min_y

    im = np.zeros((width, height, 3))

    # three cases
    # im1 pixel exists, im2 pixel does not exist => keep im1 pixel
    # im1 pixel does not exist, im2 pixel exists => keep im2 pixel
    # im1 pixel exists, im2 pixel exists => keep im2 pixel (untransformed)
    for x in range(width):
        for y in range(height):
            # check if im2 pixel exists
            if x + min_x < im2_width and y + min_y < im2_height and x + min_x >= 0 and y + min_y >= 0:
                # if it does, keep it
                im[x][y] = image_data2[x + min_x][y + min_y]
            else:
                # else check if im1 pixel exists
                # [x, y].T = R*[x_1, y_1].T + d
                # if it exists, keep it
                coords = np.array([x + min_x, y + min_y])
                original_coords = np.dot(R.T, (coords - d))
                # round them. in a more sophisticated version you would interpolate here
                x_1 = round(original_coords[0]) 
                y_1 = round(original_coords[1])
                if x_1 < im1_width and y_1 < im1_height and x_1 >= 0 and y_1 >= 0:
                    im[x][y] = image_data1[x_1][y_1]

    # return merged image
    return im

if __name__ == "__main__":
    im1 = Image.open("im1.png")
    im2 = Image.open("im2.png")

    im = imStitch(im1, im2)
    im = im / 255.0

    plt.imshow(im)
    plt.savefig("stiched.png")
    plt.show()

    im1 = Image.open("imForest1.png")
    im2 = Image.open("imForest2.png")

    im = imStitch(im1, im2)
    im = im / 255.0

    plt.imshow(im)
    plt.savefig("stiched_forest.png")
    plt.show()