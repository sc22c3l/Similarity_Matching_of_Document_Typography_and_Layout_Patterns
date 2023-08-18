
from PIL import Image
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import stats
# Open the image file
group=[

       [
[1,1,0,0,0,0,0,0,0,0],
[0,1,1,0,0,0,0,0,0,0],
[0,0,1,1,0,0,0,0,0,0],
[0,0,0,1,1,0,0,0,0,0],
[0,0,0,0,1,1,0,0,0,0],
[0,0,0,0,0,1,1,0,0,0],
[0,0,0,0,0,0,1,1,0,0],
[0,0,0,0,0,0,0,1,1,0],
[0,0,0,0,0,0,0,0,1,1],
[0,0,0,0,0,0,0,0,1,1],
[0,0,0,0,0,0,0,1,1,0],
[0,0,0,0,0,0,1,1,0,0],
[0,0,0,0,0,1,1,0,0,0],
[0,0,0,0,1,1,0,0,0,0],
[0,0,0,1,1,0,0,0,0,0],
[0,0,1,1,0,0,0,0,0,0]
         ],
# Aligned
         [
 [1,1,1,1,1,1,1,0,0,0],
 [1,1,1,1,1,1,0,0,0,0],
 [1,1,1,1,0,0,0,0,0,0],
 [1,1,1,1,1,0,0,0,0,0],
 [1,1,1,1,1,1,0,0,0,0],
 [1,1,1,1,1,0,0,0,0,0],
 [1,1,1,1,1,1,1,0,0,0],
 [1,1,1,1,1,0,0,0,0,0],
 [1,1,1,1,1,1,1,1,0,0],
 [1,1,1,1,1,1,0,0,0,0],
 [1,1,1,1,1,1,0,0,0,0],
 [1,1,1,1,1,0,0,0,0,0],
 [1,1,1,1,1,1,1,0,0,0],
 [1,1,1,1,1,0,0,0,0,0],
 [1,1,1,1,1,1,1,0,0,0],
 [1,1,1,1,1,0,0,0,0,0]
         ],
# Centered
         [
[0,0,0,0,1,1,0,0,0,0],
[0,1,1,1,1,1,1,1,1,0],
[0,0,1,1,1,1,1,1,0,0],
[0,0,0,1,1,1,1,0,0,0],
[0,0,0,1,1,1,1,0,0,0],

[0,0,0,1,1,1,1,0,0,0],
[0,1,1,1,1,1,1,1,1,0],
[0,0,1,1,1,1,1,1,0,0],
[0,0,0,1,1,1,1,0,0,0],
[0,0,0,1,1,1,1,0,0,0],

[0,0,0,1,1,1,1,0,0,0],
[0,1,1,1,1,1,1,1,1,0],
[0,0,1,1,1,1,1,1,0,0],
[0,0,0,1,1,1,1,0,0,0],
[0,0,0,1,1,1,1,0,0,0],
[0,0,0,0,1,1,0,0,0,0]
         ],
# Meshed alternately 4
         [
[1,1,1,1,0,0,0,0,0,0],
[0,0,1,1,1,1,1,0,0,0],
[1,1,1,1,1,0,0,0,0,0],
[0,0,1,1,1,1,1,0,0,0],
[0,1,1,1,1,0,0,0,0,0],
[0,0,0,1,1,1,1,1,0,0],
[0,0,1,1,1,0,0,0,0,0],
[0,0,0,0,1,1,1,1,1,0],
[1,1,1,1,1,0,0,0,0,0],
[0,0,1,1,1,1,1,0,0,0],
[1,1,1,1,1,0,0,0,0,0],
[0,0,1,1,1,1,1,0,0,0],
[1,1,1,1,1,0,0,0,0,0],
[0,0,1,1,1,1,1,0,0,0],
[1,1,1,1,1,0,0,0,0,0],
[0,0,1,1,1,1,1,0,0,0]
         ],  
        #  Matching end-start5
         [
[1,1,1,1,1,1,0,0,0,0],
[0,0,0,0,0,0,1,1,1,1],
[1,1,1,1,1,1,0,0,0,0],
[0,0,0,0,0,0,1,1,1,1],
[1,1,1,1,1,1,0,0,0,0],
[0,0,0,0,0,0,1,1,1,1],
[1,1,1,1,1,1,0,0,0,0],
[0,0,0,0,0,0,1,1,1,1],
[1,1,1,1,1,1,0,0,0,0],
[0,0,0,0,0,0,1,1,1,1],
[1,1,1,1,1,1,0,0,0,0],
[0,0,0,0,0,0,1,1,1,1],
[1,1,1,1,1,1,0,0,0,0],
[0,0,0,0,0,0,1,1,1,1],
[1,1,1,1,1,1,0,0,0,0],
[0,0,0,0,0,0,1,1,1,1]
         ],
        #  Casced 6
         [
[1,1,1,0,0,0,0,0,0,0],
[0,1,1,1,1,0,0,0,0,0],
[0,0,0,1,1,1,0,0,0,0],
[0,0,0,0,1,1,1,1,0,0],
[0,0,0,0,0,0,1,1,1,1]
         ],
         
    #  Framing 7
         [
[1,1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,1,1],
[1,1,1,0,0,0,0,0,1,1],
[1,0,0,0,0,0,0,0,1,1],
[1,0,0,0,0,0,0,0,1,1],
[1,1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,1,1]

         ],
    
        # Central cross: 
[[0,0,0,0,0,1,0,0,0,0],
 [0,0,0,0,0,1,0,0,0,0],
 [0,0,0,0,0,1,0,0,0,0],
 [0,0,0,0,0,1,0,0,0,0],
 [1,1,1,1,1,1,1,1,1,1],
 [0,0,0,0,0,1,0,0,0,0],
 [0,0,0,0,0,1,0,0,0,0],
 [0,0,0,0,0,1,0,0,0,0],
 [0,0,0,0,0,1,0,0,0,0],
 [0,0,0,0,0,1,0,0,0,0]] ,


# Centered square:
[
 [0,0,0,1,1,1,1,0,0,0],
 [0,0,0,1,1,1,1,0,0,0],
 [0,0,0,1,1,1,1,0,0,0],
 [0,0,0,1,1,1,1,0,0,0],
 [0,0,0,1,1,1,1,0,0,0],
 [0,0,0,1,1,1,1,0,0,0]],

# Hourglass shape:10
[[1,1,1,1,1,1,1,1,1,1],
 [0,1,1,1,1,1,1,1,1,0],
 [0,0,1,1,1,1,1,1,0,0],
 [0,0,0,1,1,1,1,0,0,0],
 [0,0,0,0,1,1,0,0,0,0],
 [0,0,0,0,1,1,0,0,0,0],
 [0,0,0,1,1,1,1,0,0,0],
 [0,0,1,1,1,1,1,1,0,0],
 [0,1,1,1,1,1,1,1,1,0],
 [1,1,1,1,1,1,1,1,1,1]],


[[1,1,1,1,1,1,1,1,1],
 [0,0,0,0,0,0,0,1,1],
 [1,1,1,1,1,1,1,1,1],
 [1,1,1,1,1,1,0,0,0],
 [1,1,1,1,1,1,1,1,1],
 [0,0,0,0,0,0,0,1,1],
 [1,1,1,1,1,1,1,1,1],
 [1,1,1,1,0,0,0,0,0],
 [1,1,1,1,1,1,1,1,1]]

         
      ]

index_arr=[]
size=len(group)
input_group=             [
 [1,1,1,1,1,1,1,1,1],
 [0,0,0,0,0,0,0,1,1],
 [1,1,1,1,1,1,1,1,1],
 [1,1,1,1,0,0,0,0,0],
 [1,1,1,1,1,1,1,1,1]]

import numpy as np


# index_arr=[]
# for i in range(5):
#     array1=np.array(input_group)
#     array2=np.array(group[i])
#     dist = np.linalg.norm(array1 - array2)
#     index_arr.append(dist)
    
# print('Euclidean Distance:\n',index_arr)
# print('Bottom 5 elements index: ',np.argsort(index_arr)[:5])


# from sklearn.metrics.pairwise import cosine_similarity

# # similarity = cosine_similarity(array1.flatten().reshape(1, -1), array2.flatten().reshape(1, -1))
# index_arr=[]

# for i in range(5):
#     array1=np.array(input_group)
#     array2=np.array(group[i])
#     similarity = cosine_similarity(array1.flatten().reshape(1, -1), array2.flatten().reshape(1, -1))

#     index_arr.append(similarity[0])
    
# print('Cosine Similarity:\n',index_arr)
# print('Top 5 elements index: ',np.argsort(index_arr)[-5:][::-1])


from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import directed_hausdorff

index_arr=[]

# input_group=[
#     [1,1,1,1,0,0,0,0,0,0],
#     [0,0,1,1,1,1,1,1,0,0],
#     [1,1,1,1,1,1,0,0,0,0],
#     [0,0,1,1,1,1,1,1,1,0],
#     [0,1,1,0,0,0,0,0,0,0],
#     [0,0,0,1,1,1,1,1,1,0],
#     [1,1,1,1,1,0,0,0,0,0],
#     [0,0,1,1,1,1,1,0,0,0],
#     [1,1,1,1,1,0,0,0,0,0],
#     [0,0,1,1,1,1,1,0,0,0]]

# Sample binary 2D arrays (binary images)
# matrix1 = np.array([[1, 0,1], [1, 1,0]])
# matrix2 = np.array([[1, 1,0], [0, 1,1]])

# # Calculate SSIM with a smaller window size
# ssim_value, _ = ssim(matrix1, matrix2, win_size=1, full=True)
def binary_array_to_coords(arr):
    return np.argwhere(arr == 1)
# print(ssim_value)
for i in range(size):
    array1=np.array(input_group)
    array2=np.array(group[i])
    coords1 = binary_array_to_coords(array1)
    coords2 = binary_array_to_coords(array2)
    hd1 = directed_hausdorff(coords1, coords2)[0]
    hd2 = directed_hausdorff(coords2, coords1)[0]
    hausdorff_distance = max(hd1, hd2)
    # ssim_value, _ = ssim(array1, array2, full=True)
    # print(ssim_value)
    index_arr.append(hausdorff_distance)

print('Hausdorff Distance:\n',index_arr)
print('Bottom 5 elements index: ',np.argsort(index_arr)[:5])



index_arr=[]
# Sample binary 2D arrays (binary images)
# matrix1 = np.array([[1, 0,1], [1, 1,0]])
# matrix2 = np.array([[1, 1,0], [0, 1,1]])

# # Calculate SSIM with a smaller window size
# ssim_value, _ = ssim(matrix1, matrix2, win_size=1, full=True)
def compute_hu_moments(arr):
    moments = cv2.moments(arr.astype(np.uint8))
    hu_moments = cv2.HuMoments(moments)
    return hu_moments
# print(ssim_value)
for i in range(size):
    array1=np.array(input_group)
    array2=np.array(group[i])
    hu1 = compute_hu_moments(array1)
    hu2 = compute_hu_moments(array2)

    # Compare Hu Moments
    # One simple way is to compute the sum of squared differences between the two sets of Hu Moments
    distance = np.sum((hu1 - hu2) ** 2)
    index_arr.append(distance)

print('Hu Moments Distance:\n',index_arr)
print('Bottom 5 elements index: ',np.argsort(index_arr)[:5])

index_arr=[]
# Convert binary 2D array into contours
def binary_array_to_contours(arr):
    contours, _ = cv2.findContours(arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Return the longest contour, assuming the largest contour corresponds to the main shape
    return max(contours, key=cv2.contourArea)
# print(ssim_value)
for i in range(size):
    array1=np.array(input_group)
    array2=np.array(group[i])
    contour1 = binary_array_to_contours(array1)
    contour2 = binary_array_to_contours(array2)

    # Compare Hu Moments
    # One simple way is to compute the sum of squared differences between the two sets of Hu Moments
    similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0)

    index_arr.append(similarity)

print('Contour Matching:\n',index_arr)
print('Bottom 5 elements index: ',np.argsort(index_arr)[:5])

index_arr=[]
import matplotlib.pyplot as plt

def binary_array_to_contour(arr):
    contours, _ = cv2.findContours(arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours[0].squeeze()

def compute_fourier_descriptors(contour):
    # Convert contour coordinates into complex numbers
    complex_representation = contour[:, 0] + 1j * contour[:, 1]
    
    # Compute Fourier Descriptors using FFT
    descriptors = np.fft.fft(complex_representation)
    
    return descriptors

def compare_fourier_descriptors(fd1, fd2, num_descriptors=5):
    # Using the first few descriptors (ignoring DC component)
    # This avoids high-frequency components which might be noise
    fd1 = fd1[1:num_descriptors+1]
    fd2 = fd2[1:num_descriptors+1]
    
    return np.linalg.norm(fd1 - fd2)

# print(ssim_value)
for i in range(size):
    array1=np.array(input_group)
    array2=np.array(group[i])
    contour1 = binary_array_to_contour(array1)
    contour2 = binary_array_to_contour(array2)

    fd1 = compute_fourier_descriptors(contour1)
    fd2 = compute_fourier_descriptors(contour2)

    distance = compare_fourier_descriptors(fd1, fd2)
    index_arr.append(distance)

print('Fourier Descriptors:\n',index_arr)
print('Bottom 5 elements index: ',np.argsort(index_arr)[:5])

exit()
img = Image.open('n02486410_0.JPEG')

# # Resize the image to 64x64
# img = img.resize((64, 64))

# # Convert the image data to a NumPy array
# img_array = np.array(img)

# # Print the shape of the array to confirm it's 64x64x3
# print(img_array)


# 打开图片
img = Image.open('picture\\rotated_image.jpg')

# # 旋转图片，其中angle是你想要旋转的角度
# angle = 60  # 例如，我们这里旋转45度
# img_rotated = img.rotate(angle, expand=True)

# # 保存旋转后的图片
# img_rotated.save('picture\\rotated_image.jpg')
# exit()



# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)
# #像素取反，变成白字黑底
# # gray = cv.bitwise_not(gray)
# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv.THRESH_OTSU)
# cv2.imshow('thresh', thresh)

import cv2
import numpy as np
import pytesseract
def sauvola(image, window_size, k, R):
    # # # Calculate the mean and standard deviation of the pixel intensities in the local neighborhood
    # mean, std_dev = cv2.meanStdDev(image, mask=cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size)))

    # # Calculate the threshold for each pixel
    # threshold = mean * (1 + k * ((std_dev / R) - 1))

    # # Apply the threshold to the image
    # _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
        # Calculate the local mean of the pixel intensities
    # mean = cv2.boxFilter(image, ddepth=-1, ksize=(window_size, window_size))

    # # Calculate the local standard deviation of the pixel intensities
    # sqr_img = cv2.sqrBoxFilter(image, ddepth=-1, ksize=(window_size, window_size))
    # std_dev = np.sqrt(sqr_img - np.square(mean))

    # # Calculate the threshold for each pixel
    # threshold = mean * (1 + k * ((std_dev / R) - 1))

    # # Apply the threshold to the image
    # binary_image = np.where(image > threshold, 0, 255).astype(np.uint8)
    
    # Calculate the local mean of the pixel intensities
    mean = cv2.boxFilter(image, ddepth=-1, ksize=(window_size, window_size))

    # Calculate the local standard deviation of the pixel intensities
    sqr_img = cv2.sqrBoxFilter(image, ddepth=-1, ksize=(window_size, window_size))
    std_dev = np.sqrt(sqr_img - np.square(mean))

    # Calculate the threshold for each pixel
    threshold = mean * (1 + k * ((std_dev / R) - 1))

    # Apply the threshold to the image
    binary_image = np.where(image > threshold, 255, 0).astype(np.uint8)


    return binary_image

def niblack(image, window_size, k):
    # Calculate the local mean of the pixel intensities
    mean = cv2.boxFilter(image, ddepth=-1, ksize=(window_size, window_size))

    # Calculate the local standard deviation of the pixel intensities
    sqr_img = cv2.sqrBoxFilter(image, ddepth=-1, ksize=(window_size, window_size))
    std_dev = np.sqrt(sqr_img - np.square(mean))

    # Calculate the threshold for each pixel
    threshold = mean + k * std_dev

    # Apply the threshold to the image
    binary_image = np.where(image > threshold, 255, 0).astype(np.uint8)

    return binary_image

def binarization(img):
    blur_gauss = cv2.GaussianBlur(img, (5, 5), 0)
    # plt.imshow(image)
    # plt.show()
    # Apply Median Blur
    blur_median = cv2.medianBlur(img, 5)

    # Save the images
    cv2.imwrite('picture\gaussian_blur.jpg', blur_gauss)
    cv2.imwrite('picture\median_blur.jpg', blur_median)
    ret, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save the result
    cv2.imwrite('picture\\binary_image.jpg', img_bin)
    
    sauvola_image = sauvola(img, window_size=15, k=0.5, R=128)
    cv2.imwrite('picture\\sauvola_image.jpg', sauvola_image)
    niblack_image = niblack(img, window_size=15, k=-0.2)
    cv2.imwrite('picture\\niblack_image.jpg', niblack_image)
   
    


# Load the image from file
# 'picture\\dataset\\training_data\\images\\00040534.png'
img = cv2.imread('picture\Test3.jpg',0)  # The '0' flag means load in grayscale
# Apply Gaussian Blur
plt.imshow(img)
plt.show()
binarization(img)
exit()
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# 读取图片
img = cv2.imread('picture\\rotated_image.jpg',0)

# 将图片转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用pytesseract来检测文本的方向
rotate_info = pytesseract.image_to_osd(gray, output_type=Output.DICT)
rotate_angle = rotate_info['rotate']

# 根据检测到的文本方向来纠正图片的旋转
height, width = img.shape[:2]
center = (width // 2, height // 2)
M = cv2.getRotationMatrix2D(center, rotate_angle, 1)
rotated = cv2.warpAffine(img, M, (width, height))

# 保存纠正后的图片
cv2.imwrite('corrected_text.jpg', rotated)
