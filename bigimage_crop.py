# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 11:13:56 2018

@author: tang
"""
from xml.dom.minidom import Document
import os, sys
import random
import glob  
from PIL import Image, ImageDraw
import cv2  
import numpy as np 
import matplotlib
from math import floor, ceil
from scipy.spatial import distance as dist
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xml.dom.minidom


def writeXml(tmp, imgname, w, h, d, bboxes, hbb=False):
    doc = Document()
    # owner
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    # owner
    folder = doc.createElement('folder')
    annotation.appendChild(folder)
    folder_txt = doc.createTextNode("VOC2007")
    folder.appendChild(folder_txt)

    filename = doc.createElement('filename')
    annotation.appendChild(filename)
    filename_txt = doc.createTextNode(imgname)
    filename.appendChild(filename_txt)
    # ones#
    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    source.appendChild(database)
    database_txt = doc.createTextNode("My Database")
    database.appendChild(database_txt)

    annotation_new = doc.createElement('annotation')
    source.appendChild(annotation_new)
    annotation_new_txt = doc.createTextNode("VOC2007")
    annotation_new.appendChild(annotation_new_txt)

    image = doc.createElement('image')
    source.appendChild(image)
    image_txt = doc.createTextNode("flickr")
    image.appendChild(image_txt)
    # owner
    owner = doc.createElement('owner')
    annotation.appendChild(owner)

    flickrid = doc.createElement('flickrid')
    owner.appendChild(flickrid)
    flickrid_txt = doc.createTextNode("NULL")
    flickrid.appendChild(flickrid_txt)

    ow_name = doc.createElement('name')
    owner.appendChild(ow_name)
    ow_name_txt = doc.createTextNode("idannel")
    ow_name.appendChild(ow_name_txt)
    # onee#
    # twos#
    size = doc.createElement('size')
    annotation.appendChild(size)

    width = doc.createElement('width')
    size.appendChild(width)
    width_txt = doc.createTextNode(str(w))
    width.appendChild(width_txt)

    height = doc.createElement('height')
    size.appendChild(height)
    height_txt = doc.createTextNode(str(h))
    height.appendChild(height_txt)

    depth = doc.createElement('depth')
    size.appendChild(depth)
    depth_txt = doc.createTextNode(str(d))
    depth.appendChild(depth_txt)
    # twoe#
    segmented = doc.createElement('segmented')
    annotation.appendChild(segmented)
    segmented_txt = doc.createTextNode("0")
    segmented.appendChild(segmented_txt)

    for bbox in bboxes:
        # threes#
        object_new = doc.createElement("object")
        annotation.appendChild(object_new)

        name = doc.createElement('name')
        object_new.appendChild(name)
        name_txt = doc.createTextNode(str(bbox[-1]))
        name.appendChild(name_txt)

        pose = doc.createElement('pose')
        object_new.appendChild(pose)
        pose_txt = doc.createTextNode("Unspecified")
        pose.appendChild(pose_txt)

        truncated = doc.createElement('truncated')
        object_new.appendChild(truncated)
        truncated_txt = doc.createTextNode("0")
        truncated.appendChild(truncated_txt)

        difficult = doc.createElement('difficult')
        object_new.appendChild(difficult)
        difficult_txt = doc.createTextNode("0")
        difficult.appendChild(difficult_txt)
        # threes-1#
        bndbox = doc.createElement('bndbox')
        object_new.appendChild(bndbox)
        hbb = True
        if hbb:
            xmin = doc.createElement('xmin')
            bndbox.appendChild(xmin)
            xmin_txt = doc.createTextNode(str(bbox[0]))
            xmin.appendChild(xmin_txt)

            ymin = doc.createElement('ymin')
            bndbox.appendChild(ymin)
            ymin_txt = doc.createTextNode(str(bbox[1]))
            ymin.appendChild(ymin_txt)

            xmax = doc.createElement('xmax')
            bndbox.appendChild(xmax)
            xmax_txt = doc.createTextNode(str(bbox[2]))
            xmax.appendChild(xmax_txt)

            ymax = doc.createElement('ymax')
            bndbox.appendChild(ymax)
            ymax_txt = doc.createTextNode(str(bbox[3]))
            ymax.appendChild(ymax_txt)
        else:
            x0 = doc.createElement('x1')
            bndbox.appendChild(x0)
            x0_txt = doc.createTextNode(str(bbox[0]))
            x0.appendChild(x0_txt)

            y0 = doc.createElement('y1')
            bndbox.appendChild(y0)
            y0_txt = doc.createTextNode(str(bbox[1]))
            y0.appendChild(y0_txt)

            x1 = doc.createElement('x2')
            bndbox.appendChild(x1)
            x1_txt = doc.createTextNode(str(bbox[2]))
            x1.appendChild(x1_txt)

            y1 = doc.createElement('y2')
            bndbox.appendChild(y1)
            y1_txt = doc.createTextNode(str(bbox[3]))
            y1.appendChild(y1_txt)

            x2 = doc.createElement('x3')
            bndbox.appendChild(x2)
            x2_txt = doc.createTextNode(str(bbox[4]))
            x2.appendChild(x2_txt)

            y2 = doc.createElement('y3')
            bndbox.appendChild(y2)
            y2_txt = doc.createTextNode(str(bbox[5]))
            y2.appendChild(y2_txt)

            x3 = doc.createElement('x4')
            bndbox.appendChild(x3)
            x3_txt = doc.createTextNode(str(bbox[6]))
            x3.appendChild(x3_txt)

            y3 = doc.createElement('y4')
            bndbox.appendChild(y3)
            y3_txt = doc.createTextNode(str(bbox[7]))
            y3.appendChild(y3_txt)

    xmlname = os.path.splitext(imgname)[0]
    print(os.path.join(tmp, xmlname))
    tempfile = os.path.join(tmp , xmlname + '.xml')
    with open(tempfile, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    return


def rectangle_box(x1,y1,x2,y2,x3,y3,x4,y4):
	quad = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
	rect = cv2.minAreaRect(quad)
	box1 = cv2.boxPoints(rect)
	box1 = np.int0(box1)


def order_points(pts): #4x2
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
 
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
 
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
 
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0] 
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
 
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order

    coor_tmp = np.array([tl, tr, br, bl], dtype="float32") #4x2
  
    #将距离原点最近的点设为第一个顶点
    [mm, _] = coor_tmp.shape
    dist_min = 0
    min_id = 0  #最小是哪个点
    for jj in range(mm):
        xx, yy = coor_tmp[jj][0], coor_tmp[jj][1]
        distt = xx**2 + yy**2
        if jj == 0 or distt < dist_min:
            dist_min = distt
            min_id = jj
    # print('min_id', min_id)

    coor_last = np.zeros(coor_tmp.shape)
    id_ = 0
    #将距原点最近的那个顶点放在第一个
    for j in range(min_id, mm):
        coor_last[id_, :] = coor_tmp[j, :]
        id_ += 1

    for j in range(min_id):
        coor_last[id_, :] = coor_tmp[j, :]
        id_ += 1


    return coor_last #np.array([tl, tr, br, bl], dtype="float32")



if __name__ == '__main__':
    #分割图片的输出文件夹
    crop_dir = r"D:\实验室项目\红外车辆检测\红外大车辆目标数据\img_crop"
    #分割图片可视化结果
    vision_crop_dir = r"D:\实验室项目\红外车辆检测\红外大车辆目标数据\vision_crop"
    #分割的xml
    crop_xml_dir = r"D:\实验室项目\红外车辆检测\红外大车辆目标数据\img_xml"
    # source train dir
    #原图
    train_img_dir = r"D:\实验室项目\红外车辆检测\红外大车辆目标数据\使用的图片"
    #原图的xml
    train_xml_dir = r"D:\实验室项目\红外车辆检测\红外大车辆目标数据\使用的xmlz"

    # xml_list = []
    # for file_name in os.listdir(train_img_dir):
    #     xml_list.append(file_name)

    f = glob.glob(train_xml_dir + '\*.xml' )
    #每张原图的每个目标随机裁剪几回
    repeat = 10
    #分割的照片以数字命名，起始命名为name_num
    name_num = 1
    for n in range(0,repeat):
        for file in f :
            #print(file)
            idx_P=file.rfind('z',1,len(file)-1)
            idx_D=file.rfind('.',1,len(file)-1)
            name=file[idx_P+1:idx_D]
            print('name',name)
            #img_name=name+'.xml'
            #img2_name=name+'.tif'
            # img_name  = train_img_dir+name+'.tif'
            # img2_name = target_img_dir+name+'.tif'
            img_name  = train_img_dir+name+'.jpg'

        #    print (img_name)
        #     img = cv2.imread(img_name)

            im = Image.open(img_name)
            imgwidth, imgheight = im.size
            # plt.clf()
            # plt.imshow(img)
            # currentAxis = plt.gca()

            DOMTree = xml.dom.minidom.parse(file)
            annotation = DOMTree.documentElement

            # filename = annotation.getElementsByTagName("filename")[0]
            # # print(filename)
            # imgname = filename.childNodes[0].data + '.tif'
            # print(imgname)

            objects = annotation.getElementsByTagName("object")
            if(not objects):
                continue
            print(file)
            i = 0
            for object in objects:
                i = i+1
                # crop_name = crop_dir + name + '_crop' + str(i) + '_' + str(n) + '.jpg'
                crop_name = crop_dir + '/' + str(name_num) + '.jpg'
                # xml_name = name + '_crop' + str(i) + '_' + str(n)
                xml_name = str(name_num)

                name_num = name_num + 1

                crop_vision_name = vision_crop_dir + name + '_crop' + str(i) + '_' + str(n) + '.jpg'
                bbox = object.getElementsByTagName("bndbox")[0]

                xmin = bbox.getElementsByTagName("xmin")[0]
                xmin = xmin.childNodes[0].data
                xmin = int(float(xmin))
                #        print(x0)

                ymin = bbox.getElementsByTagName("ymin")[0]
                ymin = ymin.childNodes[0].data
                ymin = int(float(ymin))
                #        print(y0)

                xmax = bbox.getElementsByTagName("xmax")[0]
                xmax = xmax.childNodes[0].data
                xmax = int(float(xmax))
                #        print(x1)

                ymax = bbox.getElementsByTagName("ymax")[0]
                ymax = ymax.childNodes[0].data
                ymax = int(float(ymax))
                category = object.getElementsByTagName("name")[0]
                category = category.childNodes[0].data
        #        print(category)

                w = xmax - xmin
                h = ymax - ymin

                if w <= 128 and h <= 128:
                    delta_x_max = floor((128 - w)/2)
                    delta_x_min = ceil(-(128 - w) / 2)
                    delta_y_max = floor((128 - h)/2)
                    delta_y_min = ceil(-(128 - h) / 2)

                    range_x_max = imgwidth - (ceil((xmax + xmin)/ 2) + 64)
                    range_x_min = 0 - (floor((xmax + xmin)/ 2) - 64)
                    range_y_max = imgheight - (ceil((ymax + ymin)/ 2) + 64)
                    range_y_min = 0 - (floor((ymax + ymin)/ 2) - 64)

                    delta_x_max = min([delta_x_max, range_x_max])
                    delta_x_min = max([delta_x_min, range_x_min])
                    delta_y_max = min([delta_y_max, range_y_max])
                    delta_y_min = max([delta_y_min, range_y_min])

                    if delta_x_min != delta_x_max:
                        move_x = random.randint(delta_x_min, delta_x_max+1)
                    else:
                        move_x = delta_x_min
                    if delta_y_min != delta_y_max:
                        move_y = random.randint(delta_y_min, delta_y_max+1)
                    else:
                        move_y = delta_y_min

                    x_crop = floor((xmin + xmax)/ 2) - 64 + move_x
                    y_crop = floor((ymin + ymax)/ 2) - 64 + move_y

                    crop = (x_crop, y_crop, x_crop + 128, y_crop + 128)
                    img_crop = im.crop(crop)
                    xmin_crop = 64 - floor(w/2) - move_x
                    xmax_crop = 64 + floor(w/2) - move_x
                    ymin_crop = 64 - floor(h/2) - move_y
                    ymax_crop = 64 + floor(h/2) - move_y
                    w_crop = 128
                    h_crop = 128



                elif w >= 256 or h >= 256:
                    continue
                else:
                    delta_x_max = floor((256 - w) / 2)
                    delta_x_min = ceil(-(256 - w) / 2)
                    delta_y_max = floor((256 - h) / 2)
                    delta_y_min = ceil(-(256 - h) / 2)

                    range_x_max = imgwidth - (ceil((xmax + xmin) / 2) + 128)
                    range_x_min = 0 - (floor((xmax + xmin) / 2) - 128)
                    range_y_max = imgheight - (ceil((ymax + ymin) / 2) + 128)
                    range_y_min = 0 - (floor((ymax + ymin) / 2) - 128)

                    delta_x_max = min([delta_x_max, range_x_max])
                    delta_x_min = max([delta_x_min, range_x_min])
                    delta_y_max = min([delta_y_max, range_y_max])
                    delta_y_min = max([delta_y_min, range_y_min])

                    if delta_x_min != delta_x_max:
                        move_x = random.randint(delta_x_min, delta_x_max+1)
                    else:
                        move_x = delta_x_min
                    if delta_y_min != delta_y_max:
                        move_y = random.randint(delta_y_min, delta_y_max+1)
                    else:
                        move_y = delta_y_min

                    x_crop = floor((xmin + xmax) / 2) - 128 + move_x
                    y_crop = floor((ymin + ymax) / 2) - 128 + move_y

                    crop = (x_crop, y_crop, x_crop + 256, y_crop + 256)
                    img_crop = im.crop(crop)
                    xmin_crop = 128 - floor(w / 2) - move_x
                    xmax_crop = 128 + floor(w / 2) - move_x
                    ymin_crop = 128 - floor(h / 2) - move_y
                    ymax_crop = 128 + floor(h / 2) - move_y
                    w_crop = 256
                    h_crop = 256
                box = []
                box.append([xmin_crop, ymin_crop, xmax_crop, ymax_crop, 'car'])
                writeXml(crop_xml_dir, xml_name, w_crop, h_crop, 1,box , hbb=True)

                # img_crop.show()
                img_crop.save(crop_name)

                draw = ImageDraw.Draw(img_crop)
                draw.line(
                    [(xmin_crop, ymin_crop), (xmin_crop, ymax_crop), (xmax_crop, ymax_crop), (xmax_crop, ymin_crop),
                     (xmin_crop, ymin_crop)], width=1, fill='red')
                # img_crop.show()
                img_crop.save(crop_vision_name)




