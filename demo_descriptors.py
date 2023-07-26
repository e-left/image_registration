import numpy as np
import cv2
from PIL import Image

from descriptors import myLocalDescriptor, myLocalDescriptorUpgrade

image = Image.open("im1.png")
image_data = np.asarray(image)
grayscale_image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)

p = [100, 100]
rhom = 5
rhoM = 20
rhostep = 1
N = 8

# snippet to rotate image if needed
# import numpy as np
# import cv2

# def rotate_image(image, angle):
#   image_center = tuple(np.array(image.shape[1::-1]) / 2)
#   rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#   result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
#   return result

print("For point p, simple descriptor")
res= myLocalDescriptor(grayscale_image_data, p, rhom, rhoM, rhostep, N)
print(res)
print("For point p, upgraded descriptor")
res_upgr = myLocalDescriptorUpgrade(grayscale_image_data, p, rhom, rhoM, rhostep, N)
print(res_upgr)

qs = [[200, 200], [202, 202]]
for i in range(len(qs)):
    q = qs[i]
    print(f"For point q{i}, simple descriptor")
    res= myLocalDescriptor(grayscale_image_data, q, rhom, rhoM, rhostep, N)
    print(res)
    print(f"For point q{i}, upgraded descriptor")
    res_upgr = myLocalDescriptorUpgrade(grayscale_image_data, q, rhom, rhoM, rhostep, N)
    print(res_upgr)