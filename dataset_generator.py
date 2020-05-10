import numpy as np
import cv2

import utils as u
import xml.etree.ElementTree as ET

'''
Rotate image and compute new bounding box
@param image - image to be rotated
@param angle - rotation angle
@param bounding_box - original bounding box
@return: the rotated image and the new bounding box
'''
def rotate_image( image, angle, bounding_box ):

    # get image dimension
    img_height, img_width = image.shape[:2]
    diff =abs(img_height - img_width )
    image = cv2.copyMakeBorder(image,100,100,0,0,cv2.BORDER_REPLICATE)

    img_height, img_width = image.shape[:2]
    # get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D( center = (img_width // 2, img_height // 2), angle = angle, scale = 1.0 )

    # apply transformation (ratate image)
    rotated_image = cv2.warpAffine( image, rotation_matrix, (img_width, img_height) )

    # --- compute new bounding box ---
    # Apply same transformation to the four bounding box corners
    rotated_point_A = np.matmul( rotation_matrix, np.array( [bounding_box[0], bounding_box[1]+100, 1] ).T )
    rotated_point_B = np.matmul( rotation_matrix, np.array( [bounding_box[2], bounding_box[1]+100, 1] ).T )
    rotated_point_C = np.matmul( rotation_matrix, np.array( [bounding_box[2 ], bounding_box[3]+100, 1] ).T )
    rotated_point_D = np.matmul( rotation_matrix, np.array( [bounding_box[0], bounding_box[3]+100, 1] ).T )
    # Compute new bounding box, that is, the bounding box for rotated object
    x = np.array( [ rotated_point_A[0], rotated_point_B[0], rotated_point_C[0], rotated_point_D[0] ] )
    y = np.array( [ rotated_point_A[1], rotated_point_B[1], rotated_point_C[1], rotated_point_D[1] ] )
    new_boundingbox = [np.min( x ).astype(int), np.min( y ).astype(int), np.max( x ).astype(int), np.max( y ).astype(int)]

    return rotated_image, new_boundingbox


def width_shift_image( image, width_shift_range, boundingbox ):

#     img_height, img_width = image.shape[:2]
#     diff =abs(img_height - img_width )
#     extd_image = cv2.copyMakeBorder(image,0,0,0,img_width,cv2.BORDER_REPLICATE)

    img_height, img_width = image.shape[:2]
    factor = img_width * width_shift_range

    M = np.float32([[1,0,factor],[0,1,0]])
    shifted_image = cv2.warpAffine( image, M, (img_width, img_height) )

    # compute new bounding box
    shifted_point_A = np.matmul( M, np.array( [boundingbox[0], boundingbox[1], 1] ).T )
    shifted_point_C = np.matmul( M, np.array( [boundingbox[2], boundingbox[3], 1] ).T )

    new_boundingbox = [ shifted_point_A[0].astype(int), shifted_point_A[1].astype(int),
                        shifted_point_C[0].astype(int), shifted_point_C[1].astype(int) ]

    return shifted_image, new_boundingbox,

def height_shift_image( image, height_shift_range, boundingbox ):

    img_height, img_width = image.shape[:2]
    factor = height_shift_range * img_height

    M = np.float32([[1,0,0],[0,1,factor]])
    shifted_image = cv2.warpAffine(image, M, (img_width, img_height) )

    # compute new bounding box
    shifted_point_A = np.matmul( M, np.array( [boundingbox[0], boundingbox[1], 1] ).T )
    shifted_point_C = np.matmul( M, np.array( [boundingbox[2], boundingbox[3], 1] ).T )

    new_boundingbox = [ shifted_point_A[0].astype(int), shifted_point_A[1].astype(int),
                        shifted_point_C[0].astype(int), shifted_point_C[1].astype(int) ]

    return shifted_image, new_boundingbox

def horizontal_skew( img, factor, boundingbox ):


    #, zoomed rotation minor
    #print([boundingbox[0],boundingbox[1]], [boundingbox[2],boundingbox[1]], [boundingbox[2],boundingbox[3]], [boundingbox[2],boundingbox[3]])
    rows,cols,ch = img.shape

    pts1 = np.float32([[boundingbox[0],boundingbox[3]], [boundingbox[0],boundingbox[1]], [boundingbox[2],boundingbox[1]]])
    pts2 = np.float32([[0,0],[0,int(rows)],[int(cols*factor),int(rows*factor)]])

    M = cv2.getAffineTransform(pts1,pts2)

    shifted_image = cv2.warpAffine(img,M,(cols,rows))

    dim2 = (shifted_image.shape[1], shifted_image.shape[0])
    dim = (img.shape[1], img.shape[0])
    # resize image
    shifted_image = cv2.resize(shifted_image,dim)
    #print(shifted_image.shape)
    rows,cols,ch = img.shape
    #                          right                            top
    new_boundingbox = [10,10,int(img.shape[1] *factor), int(img.shape[0])-10]


    print("BOUNDING BOX FOR                "+str(factor)+"; "+str(new_boundingbox),"   "+str(dim2))
    #u.show_image(shifted_image)


    # We use warpAffine to transform
    # the image using the matrix, T
    #img_translation = cv2.warpAffine(image, T, (width, height))

    return shifted_image, new_boundingbox

def scale_image( image, scale_factor, boundingbox ):

    img_height, img_width = original_image.shape[:2]

    width = (int)(scale_factor * img_width)
    height = (int)(scale_factor * img_height)

    scaled_img = cv2.resize( image, (width,height) )

    scaling_marix = np.array( [ [scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, scale_factor] ] )

    scaled_point_A = np.matmul( scaling_marix, np.array( [boundingbox[0], boundingbox[1], 1] ).T )
    scaled_point_C = np.matmul( scaling_marix, np.array( [boundingbox[2], boundingbox[3], 1] ).T )

    new_boundingbox = [ scaled_point_A[0].astype(int), scaled_point_A[1].astype(int),
                        scaled_point_C[0].astype(int), scaled_point_C[1].astype(int) ]

    return scaled_img, new_boundingbox


'''
Apply the specidied transdormation n times
return: a list with all transformated images, it bounding box and the value factor used.
'''
def apply_transformation( image, bounding_box, transformation, n ):

    import random

    t_images_list = []


    horizontal_skew(image,0.5,bounding_box)

    for i in range(0, n):
        interval = f_dic[transformation]
        factor = random.uniform(interval[0], interval[1])
        img, bb = t_dic[transformation]( image, factor, bounding_box )
        t_images_list.append( (img, bb, factor) )

    return t_images_list


t_dic = { "rotation":rotate_image, "width_shift":width_shift_image, "height_shift":height_shift_image, "scale": scale_image, "horizontal_skew":horizontal_skew}
f_dic = { "rotation":(0, 90), "width_shift":(0, 0.5), "height_shift":(0, 0.5), "scale": (0.5, 1.5),"horizontal_skew":(0.5,0.9)}






import os
import glob
import pandas as pd
import argparse
import xml.etree.ElementTree as ET
def xml_to_csv(path):
    """Iterates through all .xml files (generated by labelImg) in a given directory and combines them in a single Pandas datagrame.

    Parameters:
    ----------
    path : {str}
        The path containing the .xml files
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """

    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        original_boundingbox=[]
        for member in root.findall('object'):
            image_path = path + "/"+root.find('filename').text
            width = int(root.find('size')[0].text)
            height = int(root.find('size')[1].text)
            x_min = int(member[4][0].text)
            y_min = int(member[4][1].text)
            x_max = int(member[4][2].text)
            y_max = int(member[4][3].text)
            value = (
                    #root.find('filename').text,
                    image_path,
                    x_min,
                    y_min,
                    x_max,
                    y_max
                    )
            xml_list.append(value)
            original_boundingbox =[x_min, y_min, x_max, y_max]
        image_name = (root.find('filename').text).split(".")[0]
        #print (original_boundingbox)
        #print( path + root.find('filename').text)

        original_image = u.read_image( path + "/"+root.find('filename').text)
        #u.show_image(original_image)
        rotated_images = apply_transformation( original_image, original_boundingbox, "rotation", 16 )
        i = 0
        #u.plot_images( rotated_images )
        for image in rotated_images:
            img_height, img_width = image[0].shape[:2]
            readimage = cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)
            cv2.imwrite(path+ "/aug/" + image_name+ "_rotated_" + str(i) +".jpg", readimage)
            #tree = ET.parse("G:/image_proccessing/data_object_detection_sample/frame1.xml")
            #root = tree.getroot()
            #print(image[1])
            root.find('filename').text = str(image_name+ "_rotated_" + str(i) +".jpg")
            root[6][4][0].text = str(image[1][0]) if image[1][0]> 0 else 0
            root[6][4][1].text = str(image[1][1]) if image[1][1]> 0 else 0
            root[6][4][2].text = str(image[1][2]) if image[1][2]< img_width else str(img_width)
            root[6][4][3].text = str(image[1][3]) if image[1][3]< img_height else str(img_height)
            tree.write(path+"/aug/" + image_name+ "_rotated_" + str(i) +".xml")

            i+=1


        original_image = u.read_image( path + "/"+root.find('filename').text)
        #u.show_image(original_image)
        rotated_images = apply_transformation( original_image, original_boundingbox, "horizontal_skew", 16 )
        i = 0
        #u.plot_images( rotated_images )
        for image in rotated_images:
            img_height, img_width = image[0].shape[:2]
            readimage = cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)
            cv2.imwrite(path+ "/aug/" + image_name+ "_rotated_" + str(i) +".jpg", readimage)
            #tree = ET.parse("G:/image_proccessing/data_object_detection_sample/frame1.xml")
            #root = tree.getroot()
            #print(image[1])
            root.find('filename').text = str(image_name+ "_rotated_" + str(i) +".jpg")
            root[6][4][0].text = str(image[1][0]) if image[1][0]> 0 else 0
            root[6][4][1].text = str(image[1][1]) if image[1][1]> 0 else 0
            root[6][4][2].text = str(image[1][2]) if image[1][2]< img_width else str(img_width)
            root[6][4][3].text = str(image[1][3]) if image[1][3]< img_height else str(img_height)
            tree.write(path+"/aug/" + image_name+ "_rotated_" + str(i) +".xml")

            i+=1

        w_shifted_images = apply_transformation( original_image, original_boundingbox, "width_shift", 16 )
        #u.plot_images( w_shifted_images )
        i=0
        for image in w_shifted_images:
            img_height, img_width = image[0].shape[:2]
            convtimage = cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)
            cv2.imwrite(path+ "/aug/" + image_name+ "_w_shifted_" + str(i) +".jpg", convtimage)
            #tree = ET.parse("G:/image_proccessing/data_object_detection_sample/frame1.xml")
            #root = tree.getroot()
            #print(image[1])
            root.find('filename').text = str(image_name+ "_w_shifted_" + str(i) +".jpg")
            root[6][4][0].text = str(image[1][0]) if image[1][0]> 0 else 0
            root[6][4][1].text = str(image[1][1]) if image[1][1]> 0 else 0
            root[6][4][2].text = str(image[1][2]) if image[1][2]< img_width else str(img_width)
            root[6][4][3].text = str(image[1][3]) if image[1][3]< img_height else str(img_height)
            tree.write(path+"/aug/" + image_name+ "_w_shifted_" + str(i) +".xml")

            i+=1

        h_shifted_images = apply_transformation( original_image, original_boundingbox, "height_shift", 16 )

        i = 0
        #u.plot_images( h_shifted_images )
        for image in h_shifted_images:
            img_height, img_width = image[0].shape[:2]
            #print (img_height)
            #print(img_width)
            convtimage = cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)
            cv2.imwrite(path+ "/aug/" + image_name+ "_h_shifted_" + str(i) +".jpg", convtimage)
            #tree = ET.parse("G:/image_proccessing/data_object_detection_sample/frame1.xml")
            #root = tree.getroot()
            #print(image[1])
            root.find('filename').text = str(image_name+ "_h_shifted_" + str(i) +".jpg")
            root[6][4][0].text = str(image[1][0]) if image[1][0]> 0 else 0
            #print(root[6][4][0].text)
            root[6][4][1].text = str(image[1][1]) if image[1][1]> 0 else 0
            root[6][4][2].text = str(image[1][2]) if image[1][2]< img_width else str(img_width)
            root[6][4][3].text = str(image[1][3]) if image[1][3]< img_height else str(img_height)
            #print(root[6][4][3].text)
            tree.write(path+ "/aug/" + image_name+ "_h_shifted_" + str(i) +".xml")

            i+=1
