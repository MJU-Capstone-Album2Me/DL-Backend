import os
import cv2
import pyheif
import numpy as np

from PIL import Image

def converter(dir_path):
    if not os.path.exists(dir_path+"_convert"):
        os.makedirs(dir_path+"_convert")
    file_names = os.listdir(dir_path)
    for file_name in file_names:
        if(file_name==".DS_Store"):
            continue
        file_path = os.path.join(dir_path, file_name)
        print(file_path)
        file_name_exceptformat, img_format  = file_name.split(".")
        if img_format in ['heic', 'HEIC']:
            heif_file = pyheif.read(file_path)
            # convert the heic file to a PIL image
            pil_image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
            )
            # convert the PIL image to a cv2 image
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            # read the image using cv2
            image = cv2.imread(file_path)
        new_file_path = os.path.join(dir_path+"_convert",f"{file_name_exceptformat}.png")
        print(new_file_path)
        cv2.imwrite(new_file_path, image)

def resizer(dir_path, new_height): # scale_factor, down=True
    '''not applied on .heic .HEIC format '''
    if not os.path.exists(dir_path+"_resize"):
        os.makedirs(dir_path+"_resize")
    file_names = os.listdir(dir_path)
    for file_name in file_names:
        file_path = os.path.join(dir_path, file_name)
        image = cv2.imread(file_path)
        height, width, channels = image.shape
        aspect_ratio = round(float(width / height),3)
        scale_factor = round(max((new_height / height), (height / new_height)),3)
        # print("aspect_ratio: ", aspect_ratio, "scale_factor: ", scale_factor)
        new_width = int(aspect_ratio * new_height)
        resized_image = cv2.resize(image, (new_width, new_height))
        new_file_path = os.path.join(dir_path+"_resize", f"resize_{file_name}")
        cv2.imwrite(new_file_path, resized_image)

def cropper(dir_path):
    if not os.path.exists(dir_path+"_crop"):
        os.makedirs(dir_path+"_crop")
    file_names = os.listdir(dir_path)
    for file_name in file_names:
        file_path = os.path.join(dir_path, file_name)
        image = cv2.imread(file_path)
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold image
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        recs = []
        for contour in contours:
            # Get bounding rectangle for contours
            x, y, w, h = cv2.boundingRect(contour)
            ### 이 좌표로 크롭할 예정
            if w+h > 400:
                # print(x,y,w,h)
                recs.append([x,y,w,h])
        # Sort the list of rectangles by area, in descending order
        sorted_recs = sorted(recs, key=lambda r: r[2]*r[3], reverse=True)
        # Extract the second largest rectangle
        second_largest = sorted_recs[1]
        # print('recs', recs)
        # print('second_largest',second_largest)

        x, y, w, h = second_largest[0], second_largest[1], second_largest[2], second_largest[3]
        # Draw rectangle on image
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        new_file_path = os.path.join(dir_path+"_crop", f"crop_{file_name}")
        cv2.imwrite(new_file_path, image)

