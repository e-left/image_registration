import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from corners import myDetectHarrisFeatures

image = Image.open("im1.png")
image_data = np.asarray(image)
grayscale_image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)

corners = myDetectHarrisFeatures(grayscale_image_data)
plt.imshow(grayscale_image_data, "gray")
print(corners.shape)
cy = corners[:, 0]
cx = corners[:, 1]
plt.scatter(cx, cy, marker="v", color="red")
plt.savefig("corners.png")
plt.show()
