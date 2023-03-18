import numpy as np
import random
from skimage import transform as sk_transform


class Mask(object):
    def __init__(self, w=64, h=64, resize=64,sub_w_num=8, sub_h_num=8, dense_rate=0):
        self.w = w
        self.h = h
        self.sub_w_num = sub_w_num
        self.sub_h_num = sub_h_num
        self.target_num_max = sub_w_num*sub_h_num
        self.target_num_range = [1, 32]
        self.dense_rate = dense_rate
        self.resize = resize

    def single_square_shape(self, diameter):
        point_list = []
        move = diameter//2
        for x in range(0, diameter):
            for y in range(0, diameter):
                point_list = point_list + [(x-move, y-move)]
        return point_list

    def judge_adjacent(self, img):
        adj_point_list =[]
        move_list = [(-1,0),(1,0) , (0, -1), (0, 1)]
        m, n = img.shape
        for i in range(m):
            for j in range(n):
                if img[i,j]>0:
                    for move in move_list:
                        if 0 <= i+move[0] <= m-1 and 0 <= j + move[1] <= n-1:
                            if img[i+move[0],j + move[1]] ==0:
                                if (i+move[0],j + move[1]) not in adj_point_list:
                                    adj_point_list = adj_point_list + [(i+move[0],j + move[1])]
        return adj_point_list


    def single_random_shape(self, area):
        img =np.zeros([19,19])
        point_num =1
        img[9,9]=1
        point_list = [(0,0)]
        while point_num<area:
            adj_point_list = self.judge_adjacent(img)
            if len(adj_point_list)>0:
                    for  point in adj_point_list:
                        if random.random() < 0.5 and point_num <area:
                            img[point] = 1
                            point_list = point_list + [(point[0]-9,point[1]-9)]
                            point_num += 1
        return point_list

    def single_mask(self, target_num=None):
        if target_num is None:
            # self.target_num = random.randint(1, self.target_num_max)
            self.target_num = random.randint(self.target_num_range[0], self.target_num_range[1])
        else:
            self.target_num = target_num
        self.dense = True if random.random() < self.dense_rate else False

        pos_list = []

        pos_id_list = list(range(self.sub_w_num * self.sub_h_num))
        random.shuffle(pos_id_list)
        for i in range(self.target_num):
            w_id, h_id = divmod(pos_id_list[i], self.sub_w_num)
            x_pos = random.randint(0, self.w/self.sub_w_num-1) + w_id*self.w/self.sub_w_num
            y_pos = random.randint(0, self.h/self.sub_h_num-1) + h_id*self.h/self.sub_h_num
            pos_list = pos_list + [(x_pos, y_pos)]
        self.pos_list = np.array(pos_list)

        mask_image = np.ones([self.w, self.h])
        max_area = 20
        min_area = 3
        for i in range(self.target_num):
            area = random.randint(min_area, max_area)
            single_target_shape = self.single_random_shape(area)
            single_target_shape = np.array(single_target_shape)
            single_target_shape[:, 0] = single_target_shape[:, 0] + self.pos_list[i, 0]
            single_target_shape[:, 1] = single_target_shape[:, 1] + self.pos_list[i, 1]
            for j in range(single_target_shape.shape[0]):
                if -1 < single_target_shape[j, 0] < self.w and -1 < single_target_shape[j, 1] < self.h:
                    mask_image[single_target_shape[j, 0], single_target_shape[j, 1]] =0
        mask_image = sk_transform.resize(mask_image,(self.resize,self.resize),order=0)
        return mask_image

    def __call__(self, n,target_num=None, dense=None):
        Ms = []
        for i in range(n):
            mask = self.single_mask(target_num=target_num)
            mask=mask.astype("int64")
            Ms.append(mask)
        return  Ms