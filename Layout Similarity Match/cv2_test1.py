import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import stats


# lst1=outlier_del(lst)
# print(lst1)
# exit()
# Load the image
# img = mpimg.imread('picture\Aligned2.jpg')

# Display the image
# plt.imshow(img)
# plt.show()
# Load the image
# img = cv2.imread('picture\Test5.jpg',0)
# import grouping
def dfs(x, y, grid, visited, group, current_group):
    if x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]) or visited[x][y] or grid[x][y] == 0:
        return
    
    visited[x][y] = True
    group[x][y] = current_group  # Assign the current group number
    
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]  # 8 directions
    
    for dx, dy in directions:
        dfs(x + dx, y + dy, grid, visited, group, current_group)

def separate_groups(grid):
    if not grid:
        return []
    
    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    group = [[0 for _ in range(cols)] for _ in range(rows)]
    
    current_group = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1 and not visited[i][j]:
                current_group += 1
                dfs(i, j, grid, visited, group, current_group)

    # Separate each group into its own grid
    separated = []
    for g in range(1, current_group + 1):
        new_grid = [[1 if cell == g else 0 for cell in row] for row in group]
        new_grid1 =[]
        for row in new_grid:
            if 1 in row:
                new_grid1.append(row)
        
        separated.append(new_grid1)

    


    return separated
def line_mark(img_path):
    img = cv2.imread(img_path,0)
    # plt.imshow(img)
    # plt.show()
    ret, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = img_bin
    # plt.imshow(img)
    # plt.show()
    
    # Save the result
    # cv2.imwrite('picture\\binary_image.jpg', img_bin)

    # Apply adaptive threshold to get binary image
    # Set pixel to 255 (white) if pixel intensity > threshold and to 0 (black) otherwise
    # In cv2.adaptiveThreshold, the threshold value is a mean of the neighbourhood area minus C. 
    # C is a constant that is subtracted from the mean or weighted mean.
    _, binary_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    # for i in binary_img:
    #     for j in i:
    #         if j == 1:
    #             print(0)
    # print(binary_img[100])
    # print(len(binary_img[10]))
    # Convert 255s to 1s
    binary_img = binary_img/255

    # print(binary_img[100])


    # Define a structuring element
    # Note: the size of the structuring element may need to be tuned for your specific use case!
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))

    # Perform dilation
    dilated = cv2.dilate(binary_img, se)


    # # Find contours
    # contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scaled = cv2.convertScaleAbs(dilated)
    contours, _ = cv2.findContours(scaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    contours = np.flip(contours, axis=0)




    # 清除杂线
    # # Create a copy of the original image to draw bounding boxes on
    result = img.copy()

    black_bars = []
    _, last_y, _, _ = cv2.boundingRect(contours[0]) 
    # last_y=0
    contours1=[]
    del_list=[]
    for i in range(1,len(contours)):
        contour=contours[i]
        x, y, w, h = cv2.boundingRect(contour)
        if(h<10):
            del_list.append(i)
        last_y=y
    # print('11111:',len(contours))
    # print(del_list)
    _, last_y, _, _ = cv2.boundingRect(contours[0]) 
    contours1.append(contours[0])
    for i in range(1,len(contours)):
        if i not in del_list:
            contours1.append(contours[i])
            x, y, w, h = cv2.boundingRect(contours[i])
            # print('|Lasty-y|',abs(last_y-y),'H:',h)
        last_y=y
    # print((contours1))
    # print(len(contours1))
    # exit()
    contours=contours1
            
        
        
    #grouping
    # Loop over the contours
    groups_list=[]
    group_list=[]
    count=0
    flag=0
    _, last_y, _, _ = cv2.boundingRect(contours[0]) 

    for contour in contours:
        # Get the bounding box
    
        x, y, w, h = cv2.boundingRect(contour) 
        # print('|Lasty-y|',last_y-y,'H:',h)
        # if(last_y-y<=h*1.8):
        #     group_list.append(count)
        if(abs(last_y-y)>h*1.8):
            groups_list.append(group_list)
            
            group_list=[]
        group_list.append(count)
        if x>=10:
            x+=10
        
        # Draw the bounding box
        # 2是边框，-1是黑条
        cv2.rectangle(result, (x, y), (x+w-20, y+h-2), (0, 255, 0), -1)
        black_bars.append((y, y+h))
        last_y =y
        count=count+1
        flag=1
    groups_list.append(group_list)
    # print(groups_list)
    # print('H:',h)
    # print(len(contours))

    # cv2.imshow('Text lines', result)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # plt.imshow(result)
    # plt.show()
    cv2.imwrite('picture\\result111.jpg', result)
    # # Now let's convert this image to a 2D array with 1s and 0s
    # # First, we need to convert it to binary again
    # _, binary_result = cv2.threshold(result, 100, 255, cv2.THRESH_BINARY_INV)

    # # Convert 255s to 1s
    # binary_result = binary_result/255
    # # Now, let's convert the image such that each bar in the image is represented by a single row in the array
    # # Each row in the array will contain a single value that indicates whether that row contains any black pixels (1) or not (0)
    # binary_result = binary_result.any(axis=1).astype(int)

    # # Now binary_result is a 2D array with 1s for filled boxes (black) and 0s for other parts
    # print(binary_result)
    # print(len(binary_result))

    # Convert the result image to binary
    _, binary = cv2.threshold(result, 100, 255, cv2.THRESH_BINARY_INV)



    # Create an empty 2D array to store the output
    output = np.zeros((len(black_bars), binary.shape[1]))

    # Initialize output1 as a list of lists with the correct size
    output1 = [[0]*(int(len(output[0])/h)+1) for _ in range(len(black_bars))]
    # Loop over the start and end rows of each black bar
    for i, (start_row, end_row) in enumerate(black_bars):
        # Take the mean of the pixels in the black bar and store it in the output array
        output[i] = binary[start_row:end_row].mean(axis=0)
        j=0
        k=0
        while j < len(output[i]):
            if j+h<=len(output[i]):
                sum1=sum(output[i][j:j+h])/h
            elif j+h>len(output[i]):
                sum1=sum(output[i][j:len(output[i])])/(len(output[i])-j)
            if(sum1>100):
                output1[i][k]=1
            if(sum1<=100):
                output1[i][k]=0
            j=j+h
            k=k+1
    # print('Len:',len(output[0]))
    np.savetxt("output.txt", output1, fmt="%d")
    # print(output1)
    output2=[]
    output2_temp=[]
    for group_list in groups_list:
        output2_temp=[]
        for i in group_list:
            output2_temp.append(output1[i])
        output2.append(output2_temp)
    # print(output2)
    result_temp=[]
    print(output2)
    for row in output2:
        
        row1=separate_groups(row)
        for r in row1:
            result_temp.append(r)
    # result=separate_groups(output1)
    result=str(result_temp)
    result=result.replace('],', '],\n')
    result=result.replace('[[[', '[\n[[')
    result=result.replace(']]]', ']]\n]')
    result=result.replace('[[', '\nGroup:\n[[')
    
    
    return result


# # Example usage
# grid = [
#     [1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 1],
#     [1, 0, 1, 0, 1],
#     [1, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1]
# ]

# result = separate_groups(grid)
# for r in result:
#     # for row in r:
#     #     print(row)
#     print(r)
#     print("---")

   
# img = cv2.imread('picture\Test2.jpg',0)
# res=line_mark('uploads/Test2.JPG')
# print(res)







# if (len(output1)>len(output1[0])):
#     textlen=len(output1)
# elif (len(output1)<=len(output1[0])):
#     textlen=len(len(output1[0]))
# import numpy as np

# def squeeze_array(input_list, squeeze_size, filter_num=0.5):
#     # Compute the shape of the input array and the output array
#     input_array = np.array(input_list)
#     input_shape = input_array.shape
#     output_shape = input_shape // squeeze_size

#     # Pad the input array with zeros if necessary
#     pad_sizes = np.where(input_shape % squeeze_size != 0, squeeze_size - input_shape % squeeze_size, 0)
#     padded_array = np.pad(input_array, ((0, pad_sizes[0]), (0, pad_sizes[1])))

#     # Reshape the padded array into a 4D array where each 10x10 subarray is a separate dimension
#     reshaped_array = padded_array.reshape((*output_shape, *squeeze_size))

#     # Take the mean along the last two dimensions, compare with filter_num and convert to integer
#     output_array = (reshaped_array.mean(axis=(-1,-2)) > filter_num).astype(int)

#     return output_array

# # Now create a random 100x50 array with 0s and 1s
# input_array = np.random.choice([0, 1], size=(100, 50))

# # Squeeze it to a 10x5 array with filter_num=0.5
# grid=int(textlen/10)
# if(textlen%10>0):
#     grid=grid+1
# print(grid)
# output_array = squeeze_array(output1, (grid,grid), filter_num=0.5)

# print(output_array)


# reverse the array
# Routput = np.flip(output1, axis=0)

# Now 'output' is a 2D array where each row represents a black bar in the image
# print(output[0][120])
# print(len(output))
# print(len(output[0]))
# output1=[]
# for i in range(len(output)):
#     j=0
#     if j < range(len(output[0])):
#         sum(output[i][j:j+h])
# np.savetxt("output.txt", output1, fmt="%d")