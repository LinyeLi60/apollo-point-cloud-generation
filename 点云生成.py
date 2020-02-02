import torch
from os.path import join
import os
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from math import cos, sin
import cv2
import imageio
import json
import time
from scipy import ndimage
from car_models import car_id2name

if os.name == 'nt':
    root_dir = 'D:/pku-autonomous-driving'
else:
    root_dir = os.path.join(os.path.expanduser("~"), 'data', 'pku-autonomous-driving')
    # root_dir = "/home/ps/NewDisk/pku-autonomous-driving"
if os.name == 'nt':
    map_root_dir = 'D:/3d-car-understanding-test/test/images'
else:
    map_root_dir = os.path.join(os.path.expanduser("~"), 'data', '3d-car-understanding-test/test/images')


class PUB_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', debug=False):
        self.root_dir = root_dir
        self.split = split

        self.test_file_map = dict()
        self.test_file_mirror_map = dict()

        self.data = list()
        self.indices = None
        self.load_data()
        self.w = 3384
        self.h = 2710
        # 1686.2379、1354.9849为主点坐标（相对于成像平面）
        # 摄像机分辨率 3384*2710
        self.cx = 1686.2379
        self.cy = 1354.9849
        self.fx = 2304.5479
        self.fy = 2305.8757
        self.k = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0, 0, 1]], dtype=np.float32)
        self.debug = debug
        self.save_dir = "depth_map"
        self.debug_dir = "depth_debug"
        self.contours_dir = "contour_images"  # 轮廓图

        if not os.path.exists(self.contours_dir):
            os.makedirs(self.contours_dir)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

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

        if self.split != 'test':
            return self.get_train_item(index)
        else:
            return self.get_test_item(index)

    def get_train_item(self, index):
        index_ = self.indices[index]
        sample_info = self.data[index_]
        ImageId, PredictionString = sample_info['ImageId'], sample_info['PredictionString']
        if os.path.exists(os.path.join(self.save_dir, ImageId + '.png')) and os.path.exists(
                os.path.join(self.debug_dir, ImageId + '.png')):
            print("跳过", ImageId)

        items = PredictionString.split(' ')
        model_types, yaws, pitches, rolls, xs, ys, zs = [items[i::7] for i in range(7)]
        rgb_path = join(self.root_dir, 'train_images', f'{ImageId}.jpg')
        image = Image.open(rgb_path)
        image_array = np.array(image)

        boxes = []
        point_cloud = np.ones_like(image_array, dtype=np.float) * 999  # 点云的shape为(w, h, 3)

        final_depth_map = np.ones((image_array.shape[0], image_array.shape[1])) * 999

        for model_type, yaw, pitch, roll, x, y, z in zip(model_types, yaws, pitches, rolls, xs, ys, zs):
            yaw, pitch, roll, xw, yw, zw = [float(x) for x in [yaw, pitch, roll, x, y, z]]
            yaw, pitch, roll = -pitch, -yaw, -roll  # 好像要变换一下

            with open(os.path.join(self.root_dir, 'car_models_json',
                                   car_id2name[int(model_type)].name + '.json')) as json_file:
                data = json.load(json_file)
                vertices = np.array(data['vertices'])
                vertices[:, 1] = -vertices[:, 1]
                triangles = np.array(data['faces']) - 1

            # 顶点上采样，原来顶点只有四千多个

            # 单辆车的点云
            obj_point_cloud = np.ones_like(image_array, dtype=np.float) * 999
            ts = time.time()

            vertices, triangles = upsample_vertices_v1(vertices, triangles, depth=get_depth(z))

            Rt = np.eye(4)
            t = np.array([xw, yw, zw])
            Rt[:3, 3] = t
            Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
            Rt = Rt[:3, :]
            P = np.ones((vertices.shape[0], vertices.shape[1] + 1))
            P[:, :-1] = vertices
            P[-1] = np.array([0, 0, 0, 1])
            P = P.T

            world_cor_points = np.dot(Rt, P)
            # point_cloud.append(world_cor_points.T)
            img_cor_points = np.dot(self.k, world_cor_points)
            img_cor_points = img_cor_points.T
            img_cor_points[:, 0] /= img_cor_points[:, 2]
            img_cor_points[:, 1] /= img_cor_points[:, 2]

            # 先转置以下
            world_cor_points = world_cor_points.T
            # 先算出所有的点，然后用min
            # 取出每个点
            add_point_cloud_v1(obj_point_cloud, np.copy(img_cor_points), np.copy(world_cor_points))
            add_point_cloud_v1(point_cloud, np.copy(img_cor_points), np.copy(world_cor_points))
            # print("顶点数量", len(vertices), "耗时:", time.time() - ts)

            obj_point_cloud[obj_point_cloud == 999.0] = 0
            obj_point_cloud = obj_point_cloud[:, :, 2]

            # 二值化
            struct = ndimage.generate_binary_structure(2, 1)
            """
            array([[False,  True, False],
                   [ True,  True,  True],
                   [False,  True, False]], dtype=bool)
            """
            binary_map = (obj_point_cloud > 0).astype(int)
            print(f"距离{z}进行binary_dilation之前顶点的个数:", np.sum(binary_map))
            # np.where(binary_map==1)
            # 插值5次，基本去掉空洞
            # for _ in range(get_depth(z)):
            #     binary_map = ndimage.binary_dilation(binary_map, structure=struct)

            while True:
                contours, hierarchy = cv2.findContours(binary_map.astype(np.uint8), cv2.RETR_TREE,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                # 只绘制最外边那个轮廓

                edge_contours = [contours[c] for c in range(len(contours)) if hierarchy[0, c, 3] < 0]
                print(f"距离:{z}, 轮廓个数:{len(contours)}, 外边缘轮廓个数:{len(edge_contours)}")

                if len(edge_contours) == 1:
                    binary_map = cv2.drawContours(np.zeros_like(binary_map, dtype=np.uint8), edge_contours, -1, (255, 255, 255),
                                                  thickness=-1)  # thickness表示填充

                    contours_image = cv2.drawContours(np.zeros_like(image_array, dtype=np.uint8), edge_contours, -1, (0, 255, 0),
                                                      thickness=-1)  # thickness表示填充

                    break
                else:
                    # 重新进行上填充
                    binary_map = ndimage.binary_dilation(binary_map, structure=struct)
                    print(f"距离{z}进行binary_dilation之后顶点的个数:", np.sum(binary_map))

            previous_map = obj_point_cloud != 0  # 之前是0的位置
            updated_map = np.logical_xor(previous_map, binary_map)  # 只更新之前是0这次是1的位置
            cv2.imwrite(f'update/{ImageId}_{z}a.jpg', previous_map.astype(np.uint8)*255)
            cv2.imwrite(f'update/{ImageId}_{z}b.jpg', updated_map.astype(np.uint8)*255)
            cv2.imwrite(f'update/{ImageId}_{z}c.jpg', binary_map.astype(np.uint8))

            obj_point_cloud[updated_map] = obj_point_cloud[obj_point_cloud > 0].mean()  # 这里是不是太草率的？

            xs, ys = np.where(obj_point_cloud > 0)
            bigger_xs, bigger_ys = np.where(obj_point_cloud > final_depth_map)  # 先记住深度叠加不对的点
            final_depth_map_bak = np.copy(final_depth_map)
            final_depth_map[xs, ys] = obj_point_cloud[xs, ys]
            final_depth_map[bigger_xs, bigger_ys] = final_depth_map_bak[bigger_xs, bigger_ys]

            img_cor_points = img_cor_points.astype(int)

            xmin, ymin, xmax, ymax = self.cal_bbox(img_cor_points)

            boxes.append([xmin, ymin, xmax, ymax])

        point_cloud[point_cloud == 999] = 0  # 把999变成0

        # 填补空洞
        # depth_map = point_cloud[:, :, 2]
        final_depth_map[final_depth_map == 999] = 0
        final_depth_map *= 256.0
        # cv2.imwrite(f'depth_image/{ImageId}.png', final_depth_map)
        imageio.imwrite(f'{self.save_dir}/{ImageId}.png', final_depth_map.astype(np.uint16))

        final_depth_map /= 256.0
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            final_depth_map = self.draw_bbox(final_depth_map, xmin, ymin, xmax, ymax)

        imageio.imwrite(f'{self.debug_dir}/{ImageId}.png', final_depth_map.astype(np.uint16))

        # point_cloud = interpolate(point_cloud)

        # print(np.where((point_cloud != np.array([0, 0, 0]).all())))
        # point_cloud_reshaped = np.reshape(point_cloud, (-1, 3))
        # point_cloud_reshaped = point_cloud_reshaped[~np.all(point_cloud_reshaped == np.array([0, 0, 0]),
        #                                                     axis=1)]  # remove all zero rows
        # np.save(F'np_files/{ImageId}.npy', point_cloud_reshaped)
        # np.save(F'depth_image/{ImageId}.npy', final_depth_map)


    def get_test_item(self, index):
        # try:
        print(index)

        index = self.indices[index]
        sample_info = self.data[index]
        image_id, _ = sample_info['ImageId'], sample_info['PredictionString']
        mirror = False
        image_filename = f'{image_id}.jpg'
        rgb_path = join(self.root_dir, 'test_images', f'{image_id}.jpg')
        mask_path = rgb_path.replace('images', 'masks')
        if image_filename in self.test_file_map and os.path.exists(
                join(map_root_dir, self.test_file_map[image_filename])):
            print('redirect', image_id)
            rgb_path = join(map_root_dir, self.test_file_map[image_filename])

        elif image_filename in self.test_file_mirror_map and os.path.exists(
                join(map_root_dir, self.test_file_mirror_map[image_filename])):
            print('mirror', image_id)
            rgb_path = join(map_root_dir, self.test_file_mirror_map[image_filename])
            mirror = True
        else:
            print('不重定向', image_id)
            with open('a.txt', 'a') as f:
                print(image_id, '有问题')
                f.write(image_id + '\n')
            rgb_path = join(self.root_dir, 'test_images', f'{image_id}.jpg')

        image = Image.open(rgb_path)
        mask = Image.new('RGB', image.size)
        try:
            mask = Image.open(mask_path)
            if mirror:
                mask = ImageOps.mirror(mask)
        except Exception as e:
            pass

        if self.debug:
            cv2.imshow('', cv2.resize(np.array(image)[:, :, ::-1], (640, 480)))
            cv2.imshow('original',
                       cv2.resize(cv2.imread(join(self.root_dir, 'test_images', f'{image_id}.jpg')), (640, 480)))
            key = cv2.waitKey(0)
            if key != 32:
                with open('a.txt', 'a') as f:
                    print(image_id, '有问题')
                    f.write(image_id + '\n')
            # if mirror:
            #     cv2.destroyAllWindows()

        image_tensor, _ = self.transforms[self.split](image, _)
        return image_id, Image.open(join(self.root_dir, 'test_images', f'{image_id}.jpg')), image_tensor, mask, mirror
        # except Exception as e:
        #     print(e)

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
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), thickness=3)

        return image


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


def add_point_cloud(point_cloud, img_cor_points, world_cor_points):
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


def add_point_cloud_v1(point_cloud, img_cor_points, world_cor_points, split=1):
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
    valid_idx_0 = np.logical_and((us < 3384), (vs < 2710))
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


if __name__ == '__main__':
    max_z = 0
    dataset = PUB_Dataset(root_dir, 'train', debug=True)
    for i in range(len(dataset)):
        print(f"第{i}张图片")
        dataset.__getitem__(i)
        # mask.show()
