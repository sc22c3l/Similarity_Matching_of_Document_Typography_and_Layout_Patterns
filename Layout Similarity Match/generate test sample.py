import random


# import ast

# string_representation = '[1, 0, 1, 0, 1]'
# array = ast.literal_eval(string_representation)

# print(array)
# exit()

# import subprocess

# def run_prolog_query(query):
#     prolog_process = subprocess.Popen(['swipl', '-q', '-f', 'pl 717.pl'],
#                                       stdin=subprocess.PIPE,
#                                       stdout=subprocess.PIPE,
#                                       stderr=subprocess.PIPE,
#                                       text=True)
    
#     stdout, stderr = prolog_process.communicate(query)
#     return stdout.strip()

# array=[[1,1,1,1,0,0,0,0,0,0],
#        [0,0,1,1,1,1,1,1,0,0],
#        [1,1,1,1,1,1,0,0,0,0],
#        [0,0,1,1,1,1,1,1,1,0],
#        [0,1,1,0,0,0,0,0,0,0],
#        [0,0,0,1,1,1,1,1,1,0],
#        [0,0,1,1,0,0,0,0,0,0],
#        [0,0,0,1,1,1,1,1,1,0],
#        [1,1,1,1,0,0,0,0,0,0],
#        [0,0,1,1,1,1,1,0,0,0],
#        [1,1,1,0,0,0,0,0,0,0],
#        [0,0,1,1,1,1,1,0,0,0],
#        [1,1,1,1,1,0,0,0,0,0],
#        [0,0,1,1,1,1,1,0,0,0],
#        [1,1,1,1,1,0,0,0,0,0],
#        [0,0,1,1,1,1,1,0,0,0]         ]

# queries = [
#     "group1(X), groups_compare(X, "+ str(array)+", Y), groups_result(Y, X, Z),write(Y),  halt."
# ]
# print(1)
# for query in queries:
#     result = run_prolog_query(query)
#     print('I:',result)

# exit()

# print(random.randint(1, 10))

# # matrix = [random.randint(0, 1) for i in range(11) for j in range(11)]
# array_2d = []
# array_2d = [[random.randint(0, 1) 
#              for _ in range(60)] 
#             for _ in range(20)]

# # for i in range(11):
# #     for j in range(11):
# #         matrix[i][j]=random.randint(0, 1)
# print(array_2d)
arr=[]
for i in range(11):
    arr1=[]
    for j in range(11):
        arr1.append(0)
    arr.append(arr1)
# print(arr)
import numpy as np

# Example multi-class confusion matrix for 3 classes
# Each row: actual class
# Each column: predicted class
confusion_matrix = np.array([
[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0], 
[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
]   
)



import numpy as np

# Example multi-class confusion matrix for 3 classes
# Each row: actual class
# Each column: predicted class
confusion_matrix = np.array([
[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0], 
[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
]   )

# Calculate recall for each class
recalls = []
for i in range(confusion_matrix.shape[0]):
    TP = confusion_matrix[i, i]
    FN = sum(confusion_matrix[i, :]) - TP
    recall = TP / (TP + FN)
    recalls.append(recall)

# Macro-average recall
average_recall = np.mean(recalls)
print(average_recall)

# Micro-average recall
TP_total = sum([confusion_matrix[i, i] for i in range(confusion_matrix.shape[0])])
FN_total = sum([sum(confusion_matrix[i, :]) for i in range(confusion_matrix.shape[0])]) - TP_total
micro_avg_recall = TP_total / (TP_total + FN_total)
print(micro_avg_recall)

exit()
array=[
    [1,1],
    [1,1]
]

def extend_row(L1,size):
    while len(L1)<size:
        L1.append(0)
    return L1

def extend_col(L1,size):
    Length=len(L1[0])
    add_list=[]
    for i in range(Length):
        add_list.append(0)
    while len(L1)<size:
        L1.append(add_list)
    return L1
for i in range(len(array)):
    extend_row(array[i],5)
# extend_col(array,5)
# print(array)



import cv2
import numpy as np

def array_to_contour(binary_array):
    """Converts a binary 2D array to a contour."""
    # Convert the binary array to a format suitable for findContours
    binary_array = (binary_array * 255).astype(np.uint8)
    
    # Find the contours. This returns a list of contours, so we might need to process multiple contours.
    contours, _ = cv2.findContours(binary_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # In this case, we're just returning the first contour.
    return contours[0]

def are_shapes_similar(binary_array1, binary_array2):
    """Returns True if the shapes in binary_array1 and binary_array2 are similar, False otherwise."""
    # Convert the binary arrays to contours
    contour1 = array_to_contour(binary_array1)
    contour2 = array_to_contour(binary_array2)

    print(contour1) 
    
    
    print(contour2) 
    # Compute the similarity between the two contours. The return value is a metric showing the similarity.
    # Lower values mean higher similarity.
    return cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0) < 0.1  # You may need to adjust the threshold

# Two example binary arrays
binary_array1 = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

binary_array2 = np.array([
    [0, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0]
])

print(are_shapes_similar(binary_array1, binary_array2))  # Prints: True
exit()
import numpy as np

def max_pool(input_array, pool_size):
    # Compute the shape of the input array and the output array
    input_shape = input_array.shape
    output_shape = (input_shape[0]//pool_size[0], input_shape[1]//pool_size[1])
    
    # Create the output array
    output_array = np.zeros(output_shape, dtype=input_array.dtype)
    
    # For each window in the input array, take the max and assign it to the output array
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            output_array[i, j] = np.max(input_array[i*pool_size[0]:(i+1)*pool_size[0], j*pool_size[1]:(j+1)*pool_size[1]])
    
    return output_array

# Now create a random 100x50 array with 0s and 1s
input_array = np.random.choice([0, 1], size=(100, 50))

# Reduce it to a 10x5 array using max pooling
# output_array = max_pool(input_array, (5, 10))
# print(output_array)

import numpy as np

def squeeze_array(input_array, squeeze_size=(10,10), filter_num=0.5):
    # Compute the shape of the input array and the output array
    input_shape = np.array(input_array.shape)
    output_shape = input_shape // squeeze_size

    # Pad the input array with zeros if necessary
    pad_sizes = np.where(input_shape % squeeze_size != 0, squeeze_size - input_shape % squeeze_size, 0)
    padded_array = np.pad(input_array, ((0, pad_sizes[0]), (0, pad_sizes[1])))

    # Reshape the padded array into a 4D array where each 10x10 subarray is a separate dimension
    reshaped_array = padded_array.reshape((*output_shape, *squeeze_size))

    # Take the mean along the last two dimensions, compare with filter_num and convert to integer
    output_array = (reshaped_array.mean(axis=(-1,-2)) > filter_num).astype(int)

    return output_array

# Now create a random 100x50 array with 0s and 1s
input_array = np.random.choice([0, 1], size=(100, 50))

# Squeeze it to a 10x5 array with filter_num=0.5
output_array = squeeze_array(input_array, filter_num=0.5)

print(output_array)
