import pyrender
import torch
from os.path import join
import os

import trimesh
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from math import cos, sin
import cv2
import math
import imageio
import json
import time
from multiprocessing import Queue, Process
from scipy import ndimage
from car_models import car_id2name

if os.name == 'nt':
    root_dir = 'D:/pku-autonomous-driving'
else:
    root_dir = os.path.join(os.path.expanduser("~"), 'data', 'pku-autonomous-driving')


class PUB_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', debug=False):
        self.root_dir = root_dir
        self.split = split

        self.test_file_map = dict()
        self.test_file_mirror_map = dict()

        self.data = list()
        self.indices = None
        self.load_data()

        self.scale_rate = 3384 / 640     # 长宽都缩放3倍

        self.w = int(3384 / self.scale_rate)
        self.h = int(2710 / self.scale_rate)
        self.cx = 1686.2379 / self.scale_rate
        self.cy = 1354.9849 / self.scale_rate
        self.fx = 2304.5479 / self.scale_rate
        self.fy = 2305.8757 / self.scale_rate

        self.k = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0, 0, 1]], dtype=np.float32)
        self.debug = debug
        self.mask_save_dir = "instance_mask"
        self.depth_save_dir = "instance_depth"

        if not os.path.exists(self.mask_save_dir):
            os.makedirs(self.mask_save_dir)

        if not os.path.exists(self.depth_save_dir):
            os.makedirs(self.depth_save_dir)


    def load_data(self):
        if self.split in ['train', 'val']:
            data = pd.read_csv(join(self.root_dir, 'train.csv'))
            for (ImageId, PredictionString) in data.values:
                self.data.append({'ImageId': ImageId,
                                  'PredictionString': PredictionString})

        else:
            data = pd.read_csv(join(self.root_dir, 'sample_submission.csv'))
            for (ImageId, PredictionString) in data.values:
                self.data.append({'ImageId': ImageId,
                                  'PredictionString': PredictionString})
        sample_count = len(self.data)  # 训练集中样本的数量
        indices = np.arange(sample_count)
        # np.random.seed(0)  # 固定随机种子
        # np.random.shuffle(indices)

        self.indices = indices

    def __getitem__(self, index):

        return self.get_train_item(index)

    def get_train_item(self, index):
        index_ = self.indices[index]
        sample_info = self.data[index_]
        ImageId, PredictionString = sample_info['ImageId'], sample_info['PredictionString']
        # if ImageId != "ID_cd09cc695":
        #     return

        items = PredictionString.split(' ')
        model_types, yaws, pitches, rolls, xs, ys, zs = [items[i::7] for i in range(7)]
        rgb_path = join(self.root_dir, 'train_images', f'{ImageId}.jpg')
        image = Image.open(rgb_path)
        image_array = np.array(image)
        image_array = cv2.resize(image_array, (self.w, self.h))[:, :, ::-1]

        boxes = []

        for model_type, yaw, pitch, roll, x, y, z in zip(model_types, yaws, pitches, rolls, xs, ys, zs):
            yaw, pitch, roll, xw, yw, zw = [float(x) for x in [yaw, pitch, roll, x, y, z]]
            yaw, pitch, roll = -pitch, -yaw, -roll  # 好像要变换一下

            if os.path.exists(f"{self.depth_save_dir}/{ImageId}_{xw}_{zw}.png") and \
                    os.path.exists(f"{self.mask_save_dir}/{ImageId}_{xw}_{zw}.png"):
                print("跳过:", f"{self.depth_save_dir}/{ImageId}_{xw}_{zw}.png")
                continue

            with open(os.path.join(self.root_dir, 'car_models_json',
                                   car_id2name[int(model_type)].name + '.json')) as json_file:
                data = json.load(json_file)
                vertices = np.array(data['vertices'])
                vertices[:, 1] = -vertices[:, 1]
                triangles = np.array(data['faces']) - 1

            vertices, triangles = upsample_vertices_v1(vertices, triangles, depth=1)

            Rt = np.eye(4)
            t = np.array([xw, yw, zw])
            Rt[:3, 3] = t
            Rt[:3, :3] = euler_to_Rot(yaw, -pitch, roll).T
            Rt = Rt[:3, :]
            P = np.ones((vertices.shape[0], vertices.shape[1] + 1))
            P[:, :-1] = vertices
            # P[-1] = np.array([0, 0, 0, 1])
            P = P.T

            world_cor_points = np.dot(Rt, P)
            # point_cloud.append(world_cor_points.T)
            img_cor_points = np.dot(self.k, world_cor_points)
            img_cor_points = img_cor_points.T
            img_cor_points[:, 0] /= img_cor_points[:, 2]
            img_cor_points[:, 1] /= img_cor_points[:, 2]
            img_cor_points = img_cor_points.astype(int)

            xmin, ymin, xmax, ymax = self.cal_bbox(img_cor_points)

            boxes.append([xmin, ymin, xmax, ymax])
            # cv2.imshow('', cv2.resize(image_array[ymin: ymax, xmin: xmax], (300, 300)))
            # cv2.imshow('bbox', self.draw_bbox(image_array, xmin, ymin, xmax, ymax))
            # cv2.waitKey(500)
            # print(vertices.shape, triangles.shape)

            vertices[:, 1] = -vertices[:, 1]
            # icosahedron = trimesh.creation.capsule()
            # icosahedron = pyrender.Mesh.from_trimesh(icosahedron)
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, smooth=False)
            camera = pyrender.IntrinsicsCamera(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)
            mesh = pyrender.Mesh.from_trimesh(mesh)
            scene = pyrender.Scene()

            Rt = np.eye(4)
            t = np.array([xw, yw, zw])
            Rt[:3, 3] = t

            def rotx(t):
                ''' 3D Rotation about the x-axis. '''
                c = np.cos(t)
                s = np.sin(t)
                return np.array([[1, 0, 0],
                                 [0, c, -s],
                                 [0, s, c]])

            def roty(t):
                ''' Rotation about the y-axis. '''
                c = np.cos(t)
                s = np.sin(t)
                return np.array([[c, 0, s],
                                 [0, 1, 0],
                                 [-s, 0, c]])

            def rotz(t):
                ''' Rotation about the z-axis. '''
                c = np.cos(t)
                s = np.sin(t)
                return np.array([[c, -s, 0],
                                 [s, c, 0],
                                 [0, 0, 1]])

            R = np.dot(rotx(pitch), roty(yaw))
            Rt[:3, :3] = R

            scene.add(mesh, pose=Rt)
            # Y = np.array([[cos(yaw), 0, sin(yaw)],
            #               [0, 1, 0],
            #               [-sin(yaw), 0, cos(yaw)]])
            cam_pose = np.eye(4)
            cam_pose[:3, :3] = np.dot(rotz(math.pi), roty(math.pi))
            scene.add(camera, pose=cam_pose)
            # scene.add(icosahedron, pose=np.eye(4))
            # scene.add(camera, pose=np.array([[1, 0, 0, 0],
            #                                  [0, 1, 0, 0],
            #                                  [0, 0, 1, 0],
            #                                  [0, 0, 0, 1]], dtype=np.float32))
            # pyrender.Viewer(scene, viewport_size=(self.w, self.h), use_raymond_lighting=True)
            r = pyrender.OffscreenRenderer(self.w, self.h)
            color, depth = r.render(scene)
            color = np.array(color)
            weighted_image = np.copy(image_array)
            cv2.addWeighted(weighted_image, 0.5, color, 0.5, 0, weighted_image)

            cv2.imwrite(f'{self.mask_save_dir}/{ImageId}_{xw}_{zw}.png', color)
            depth *= 256.0
            imageio.imwrite(f'{self.depth_save_dir}/{ImageId}_{xw}_{zw}.png', depth.astype(np.uint16))
            # cv2.imshow('', weighted_image)
            # cv2.waitKey(500)
            r.delete()    # Free all OpenGL resources.

    def __len__(self):
        # if os.name == 'nt' and not self.debug:
        #     return 2
        return len(self.indices)

    def world_2_image(self, model_type, xw, yw, zw, yaw, pitch, roll):
        x_l, y_l, z_l = self.model_size_dict[model_type]
        Rt = np.eye(4)
        t = np.array([xw, yw, zw])
        Rt[:3, 3] = t
        rot_mat = euler_to_Rot(yaw, pitch, roll).T
        #
        Rt[:3, :3] = rot_mat
        Rt = Rt[:3, :]
        rotation_vec, _ = cv2.Rodrigues(Rt[:3, :3])
        # print(yaw, pitch, roll, rotation_vec, zw/10)

        P = np.array([[0, 0, 0, 1],
                      [x_l, y_l, -z_l, 1],
                      [x_l, y_l, z_l, 1],
                      [-x_l, y_l, z_l, 1],
                      [-x_l, y_l, -z_l, 1],
                      [x_l, -y_l, -z_l, 1],
                      [x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1],

                      [0, 0, z_l, 1],
                      [0, 0, -z_l, 1],
                      ]).T
        img_cor_points = np.dot(self.k, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        return img_cor_points

    def image_2_world(self, img_cor):
        u, v, z = img_cor
        xw = (u - self.cx) / self.fx * z
        yw = (v - self.cy) / self.fy * z
        return xw, yw, z

    def cal_bbox(self, points):
        xmin, ymin, zmin = np.min(points, axis=0)
        xmax, ymax, zmax = np.max(points, axis=0)
        # xmin = np.clip(xmin, 0, self.w)
        # ymin = np.clip(ymin, 0, self.h)
        # xmax = np.clip(xmax, 0, self.w)
        # ymax = np.clip(ymax, 0, self.h)
        return xmin, ymin, xmax, ymax

    def draw_bbox(self, image, xmin, ymin, xmax, ymax):
        image = np.array(image)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=1)

        return image

    def add_point_cloud_v1(self, point_cloud, img_cor_points, world_cor_points, split=1):
        """

        :param point_cloud: shape is (2710, 3384, 3)
        :param img_cor_points: shape is (N, 3), 最后一个元素存储图像坐标系上点的深度, 类型为float
        :param world_cor_points: shape is (N, 3), 存储世界坐标系中的坐标值, 类型为float
        :return:
        """
        # 先去掉img_cor_points中超出范围的
        # 深度从大到小
        sorted_index = np.argsort(-world_cor_points[:, 2], axis=0)  # 先根据深度进行sort
        img_cor_points = img_cor_points[sorted_index]
        world_cor_points = world_cor_points[sorted_index]

        us, vs = img_cor_points[:, 0].astype(int), img_cor_points[:, 1].astype(int)
        valid_idx_0 = np.logical_and((us < self.w), (vs < self.h))
        valid_idx_1 = np.logical_and((us > 0), (vs > 0))
        valid_idx = np.logical_and(valid_idx_0, valid_idx_1)

        num_elements = us.shape[0]  #
        interval_length = num_elements // split
        # 分批次进行深度计算
        for i in range(split):
            us_ = us[valid_idx][i * interval_length:(i + 1) * interval_length]
            vs_ = vs[valid_idx][i * interval_length:(i + 1) * interval_length]

            # 取出距离
            world_cor_points_ = world_cor_points[valid_idx][i * interval_length:(i + 1) * interval_length]
            previous_zs = point_cloud[vs_, us_, 2]
            current_zs = world_cor_points_[:, 2]

            updated_idx = current_zs < previous_zs  # 现在的深度比之前小的，就更新它
            point_cloud[vs_[updated_idx], us_[updated_idx]] = world_cor_points_[updated_idx]

        # 在这里把落在相同位置的深度给整出来
        # vus = np.concatenate((vs.reshape(-1, 1), us.reshape(-1, 1)), axis=1)
        # uniq_vus = np.unique(vus, axis=0)
        # 先把uniq_uvs扩展到与uvs的长度相同
        # padded = np.zeros_like(uvs)
        # padded[:uniq_uvs.shape[0], :uniq_uvs.shape[1]] = uniq_uvs

        # for uniq_vu in uniq_vus:
        #     # 先取出所有这个值的下标
        #     same_idx = np.all(vus == uniq_vu, axis=1)
        #     min_z_idx = np.argmin(world_cor_points[same_idx][:, 2])
        #
        #     point_cloud[uniq_vu[0], uniq_vu[1]] = world_cor_points[same_idx][min_z_idx]
        #     print(world_cor_points[same_idx][min_z_idx])
        # print(vus.shape, uniq_vus.shape)

        # for k, img_cor_point in enumerate(img_cor_points):
        #     i_x, i_y, z = img_cor_point
        #     i_x, i_y = int(i_x), int(i_y)
        #
        #     try:
        #         # if (point_cloud[i_x, i_y] == np.array([0, 0, 0])).all():
        #         #     point_cloud[i_x, i_y] = world_cor_points[k]
        #         # else:
        #         previous_z = point_cloud[i_x, i_y, 2]
        #         if z < previous_z:  # 如果这个点之前有过点了，比较深度，取深度小的
        #             point_cloud[i_x, i_y] = world_cor_points[k]
        #     except IndexError:
        #         pass


def draw_obj(image, vertices, triangles):
    for t in triangles:
        coord = np.array([vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]], dtype=np.int32)
        #         cv2.fillConvexPoly(image, coord, (0,0,255))
        cv2.polylines(image, np.int32([coord]), 1, (0, 0, 255))


def get_depth(z):
    z = float(z)
    interval = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80])
    depth = [5, 4, 4, 4, 4, 4, 4, 4, 3]
    idx = np.where(interval < z)[0].max()
    print("迭代次数:", depth[idx])
    return depth[idx]


def check_valid(mask, x, y):
    # print(mask.size, x, y)
    try:
        r, g, b = mask.getpixel((int(x), int(y)))
        if (r, g, b) == (255, 255, 255):
            return False
    except:
        pass

    return True


def draw_line(image, points):
    image = np.array(image)
    color = (255, 0, 0)
    lineTpye = cv2.LINE_4
    # cv2.line(image, tuple(points[0][:2]), tup        le(points[9][:2]), color, lineTpye)
    # cv2.line(image, tuple(points[0][:2]), tuple(points[10][:2]), color, lineTpye)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, lineTpye)
    cv2.line(image, tuple(points[1][:2]), tuple(points[4][:2]), color, lineTpye)

    cv2.line(image, tuple(points[1][:2]), tuple(points[5][:2]), color, lineTpye)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, lineTpye)
    cv2.line(image, tuple(points[2][:2]), tuple(points[6][:2]), color, lineTpye)
    cv2.line(image, tuple(points[3][:2]), tuple(points[4][:2]), color, lineTpye)
    cv2.line(image, tuple(points[3][:2]), tuple(points[7][:2]), color, lineTpye)

    cv2.line(image, tuple(points[4][:2]), tuple(points[8][:2]), color, lineTpye)
    cv2.line(image, tuple(points[5][:2]), tuple(points[8][:2]), color, lineTpye)

    cv2.line(image, tuple(points[5][:2]), tuple(points[6][:2]), color, lineTpye)
    cv2.line(image, tuple(points[6][:2]), tuple(points[7][:2]), color, lineTpye)
    cv2.line(image, tuple(points[7][:2]), tuple(points[8][:2]), color, lineTpye)
    return image


def draw_points(image, points):
    image = np.array(image)
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), 15, (255, 0, 0), -1)
    return image


def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))


def upsample_vertices_v1(vertices, triangles, depth=5):
    # 所有顶点的坐标
    if depth == 0:
        return vertices, triangles
    else:
        point_as_cor, point_bs_cor, point_cs_cor = vertices[triangles][:, 0], vertices[triangles][:, 1], vertices[
                                                                                                             triangles][
                                                                                                         :, 2]
        # 所有顶点的下标
        point_as_idx, point_bs_idx, point_cs_idx = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        #
        center_point_1_cor = (point_as_cor + point_bs_cor) / 2
        center_point_2_cor = (point_as_cor + point_cs_cor) / 2
        center_point_3_cor = (point_bs_cor + point_cs_cor) / 2

        center_point_1_idx = np.arange(vertices.shape[0], vertices.shape[0] + center_point_1_cor.shape[0])
        vertices = np.concatenate((vertices, center_point_1_cor), axis=0)

        center_point_2_idx = np.arange(vertices.shape[0], vertices.shape[0] + center_point_2_cor.shape[0])
        vertices = np.concatenate((vertices, center_point_2_cor), axis=0)

        center_point_3_idx = np.arange(vertices.shape[0], vertices.shape[0] + center_point_3_cor.shape[0])
        vertices = np.concatenate((vertices, center_point_3_cor), axis=0)

        triangle_1s = np.concatenate(
            (center_point_1_idx.reshape((-1, 1)), center_point_2_idx.reshape((-1, 1)), point_as_idx.reshape(-1, 1)),
            axis=1)
        triangle_2s = np.concatenate(
            (center_point_1_idx.reshape((-1, 1)), center_point_3_idx.reshape((-1, 1)), point_bs_idx.reshape(-1, 1)),
            axis=1)
        triangle_3s = np.concatenate(
            (center_point_2_idx.reshape((-1, 1)), center_point_3_idx.reshape((-1, 1)), point_cs_idx.reshape(-1, 1)),
            axis=1)
        triangle_4s = np.concatenate((center_point_1_idx.reshape((-1, 1)), center_point_2_idx.reshape((-1, 1)),
                                      center_point_3_idx.reshape(-1, 1)), axis=1)

        triangles = np.concatenate((triangles, triangle_1s, triangle_2s, triangle_3s, triangle_4s), axis=0)

        return upsample_vertices_v1(vertices, triangles, depth - 1)


def upsample_vertices(vertices, triangles, depth=5):
    # 用concatenate
    t1 = time.time()
    for triangle in np.copy(triangles):
        # 顶点下标
        point_a_idx, point_b_idx, point_c_idx = triangle
        point_a, point_b, point_c = vertices[triangle]  # 获取三角形的三个点
        # 求三个顶点的三个中点
        center_point1 = np.array((point_a + point_b) / 2)
        center_point2 = np.array((point_a + point_c) / 2)
        center_point3 = np.array((point_b + point_c) / 2)

        # 把点加到vertices中
        vertices = np.concatenate((vertices, center_point1[None]))
        center_point1_idx = vertices.shape[0] - 1
        vertices = np.concatenate((vertices, center_point2[None]))
        center_point2_idx = vertices.shape[0] - 1
        vertices = np.concatenate((vertices, center_point3[None]))
        center_point3_idx = vertices.shape[0] - 1

        triangles = np.concatenate((triangles, np.array([point_a_idx, center_point1_idx, center_point2_idx])[None]))
        triangles = np.concatenate((triangles, np.array([point_b_idx, center_point1_idx, center_point3_idx])[None]))
        triangles = np.concatenate((triangles, np.array([point_c_idx, center_point2_idx, center_point3_idx])[None]))
        triangles = np.concatenate(
            (triangles, np.array([center_point1_idx, center_point2_idx, center_point3_idx])[None]))

    print("耗时：", depth, time.time() - t1)
    if depth == 0:
        return vertices, triangles
    else:
        return upsample_vertices(vertices, triangles, depth - 1)


    def add_point_cloud(self, point_cloud, img_cor_points, world_cor_points):
        for k, img_cor_point in enumerate(img_cor_points):
            i_x, i_y, z = img_cor_point
            i_x, i_y = int(i_x), int(i_y)

            try:
                previous_z = point_cloud[i_x, i_y, 2]
                if z < previous_z:  # 如果这个点之前有过点了，比较深度，取深度小的
                    # print("替换")
                    point_cloud[i_x, i_y] = world_cor_points[k]
            except IndexError:
                pass




# def interpolate(pc):
#     h, w, _ = pc.shape
#     radius = 5
#     for i in range(radius, h):
#         for j in range(radius, w):
#             if (pc[i, j] == np.array([0, 0, 0])).all():
#                 for k in range(3):
#                     try:
#                         mean_value = np.mean(pc[i - radius:i + radius, j - radius:j + radius, k])
#                         pc[i, j, k] = mean_value
#                     except IndexError:
#                         pass
#     return pc

def mul_process_get_item(queue, dataset):
    while not queue.empty():
        t = queue.get()
        try:
            dataset.__getitem__(t)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    max_z = 0
    dataset = PUB_Dataset(root_dir, 'train', debug=True)

    q = Queue()
    for i in range(len(dataset)):
        q.put(i)

    p_list = []
    for i in range(1):
        p = Process(target=mul_process_get_item, args=(q, dataset))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()




        # print(f"第{i}张图片")
        # dataset.__getitem__(i)
