import cv2
import numpy as np
# red green
color_lst = [(178, 34, 34), (0, 255, 0), (0, 255, 255), (0, 0, 255)]

def contour_and_draw(image, label_map, n_class=2, shape=(512, 512)):
    #image should be (512,512,3), label_map should be (512, 512)
    all_contours=[]
    for c_id in range(1, n_class):
        one_channel = np.zeros(shape, dtype=np.uint8)
        one_channel[label_map == c_id] = 1
        _, contours, hierarchy = cv2.findContours(one_channel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.append(contours)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    return image, all_contours

def contour_and_draw_rainbow(image,label_map, color_num, color_range, n_class=2, shape=(512, 512)):
    #image should be (512,512,3), label_map should be (512,512)
    all_contours=[]
    for c_id in range(1, n_class):
        one_channel = np.zeros(shape, dtype=np.uint8)
        one_channel[label_map == c_id] = 1
        _, contours, hierarchy = cv2.findContours(one_channel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.append(contours)
        color_step = (255 * 4) / color_range
        assert color_num < color_range, 'color_num should be less than or equal to color_range'
        if color_num < color_range / 4:
            cv2.drawContours(image, contours, -1, (255, color_step * color_num, 0), 2)
        elif color_num < color_range / 2:
            cv2.drawContours(image, contours, -1, (255 - color_step * (color_num - color_range/4) , 255, 0), 2)
        elif color_num < (color_range * 3 / 4):
            cv2.drawContours(image, contours, -1, (0, 255, color_step * (color_num - color_range/2)), 2)
        else:
            cv2.drawContours(image, contours, -1, (0, (255 - color_step * (color_num - (color_range*3/4))), 255), 2)
    return image, all_contours

def contour_and_draw_brainRegion(image,label_map, n_class=4, shape=(512, 512)):
    #image should be (512,512,3), label_map should be (cls_num,512,512)
    all_contours=[]
    for c_id in range(1, n_class):
        one_channel = np.zeros(shape, dtype=np.uint8)
        one_channel[label_map[c_id] == 1] = 1
        contours, hierarchy = cv2.findContours(one_channel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.append(contours)
        cv2.drawContours(image, contours, -1, color_lst[c_id], 2)
    return image, all_contours

def contour_and_draw_doc(image,label_maps, n_class=2, shape=(512, 512)):
    #the first label map belongs to AI,and then 3 other person
    all_contours=[]
    for c_id in range(len(label_maps)):
        for cls in range(1, n_class):
            label_map = label_maps[c_id]
            one_channel = np.zeros(shape, dtype=np.uint8)
            one_channel[label_map == cls] = 1
            _, contours, hierarchy = cv2.findContours(one_channel, cv2.RETR_TREE,
                                                      cv2.CHAIN_APPROX_SIMPLE)
            all_contours.append(contours)
            image = cv2.drawContours(image, contours, -1, color_lst[c_id], (2, 1)[c_id > 0])

    return image, all_contours