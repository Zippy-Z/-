import math
from skimage import feature as ft
import matplotlib.pyplot as plt
from sklearn import svm
import cv2
from sklearn import metrics
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import os
import re
import torch
import random
import glob
import numpy as np
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
import time

# =============================================================================
# 使用说明：
# 1、使用了path = os.getcwd()函数获取文件路径，故请将本文件置于train与test数据集所在路径运行
# 2、我们使用的训练数据集中手动删除了study82样本所对应的txt文件与jpg图片（此样本只有10个label），并修改了study168样本中其中一个label的错误'identification'值（将'T11-T12'改为'T12-L1'）
# 3、由于我们剔除了部分不符规范的样本，运行前请手动修改/删除上述两个样本，确保train数据集样本数为149，test数据集样本数为51
# =============================================================================

# 去除随机性
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministric = True
torch.backends.cudnn.benchamark = False
torch.set_default_tensor_type(torch.DoubleTensor)


# =============================================================================
# 将图片的大小统一修改为256*256
# 统一图片大小方便后面输入网络中
# =============================================================================
def resize_image(split, path):
    """修改图片尺寸
    :param infile: 图片源文件
    :param outfile: 重设尺寸文件保存地址
    :param x_s: 设置的宽度
    :return:
    """
    filename = glob.glob(path + '/' + split + '/data/*.jpg')
    for ind, file in enumerate(filename):
        dd = filename[ind].split('\\')
        im = Image.open(file)
        x, y = im.size
        y_s = 256
        x_s = 256
        out = im.resize((x_s, y_s), Image.ANTIALIAS)
        outfile = path + '/' + split + '1/data/' + dd[-1]
        out.save(outfile)


# =============================================================================
#          把每个图片的g各个部位的位置数据存到Y_1中
# =============================================================================
def load_Y_data(path, split, num):
    Y_1 = np.zeros((11, 2, num), np.int64)
    for ind, file in enumerate(glob.glob(path + '/' + split + '/label/*.txt')):
        df = open(file, 'r')
        lines = df.readlines()
        j = 0
        for i in lines:
            d = [m.start() for m in re.finditer(',', i)]
            Y_1[j, 0, ind] = eval(i[0:d[0]])
            Y_1[j, 1, ind] = eval(i[d[0] + 1:d[1]])
            j = j + 1
        df.close()
    return Y_1


# =============================================================================
# 由于图片缩小之后，脊椎对应的位置坐标也应该调整
# =============================================================================
def rechange_position(split, path, Y_position, num):
    # Y_position为各个图片中的脊椎位置的三维数组，大小为11*2，有149层
    Y_position_new = np.zeros((11, 2, num), np.int64)
    for ind, file in enumerate(glob.glob(path + '/' + split + '/data/*.jpg')):
        im = Image.open(file)
        x, y = im.size
        for j in range(11):
            Y_position_new[j, 0, ind] = int(Y_position[j, 0, ind] * 256 / x)
            Y_position_new[j, 1, ind] = int(Y_position[j, 1, ind] * 256 / y)
    return Y_position_new


# =============================================================================
# 将位置坐标从y坐标值从大到小排序
# =============================================================================
def sort_position(Y_1, num):
    Y_position_sort = Y_1
    for i in range(num):
        data = Y_position_sort[:, :, i]
        data = data[np.argsort(-data[:, 1])]
        Y_position_sort[:, :, i] = data
    return Y_position_sort


# =============================================================================
# 将{'identification': 'L5', 'vertebra': 'v2'}等信息以字典形式存入Y_definition中
# =============================================================================
def load_definition(path, split, num):
    Y_definition = []
    for ind, file in enumerate(glob.glob(path + '/' + split + '/label/*.txt')):
        Y_definition_each = []
        df = open(file, 'r')
        lines = df.readlines()
        for i in lines:
            each = {}
            d1 = [m.start() for m in re.finditer('{', i)]
            d2 = [m.start() for m in re.finditer('}', i)]
            text_temp = i[d1[0] + 1:d2[0]]
            d3 = [m.start() for m in re.finditer("'", text_temp)]
            text_4 = text_temp[d3[-2] + 1:d3[-1]]
            text_3 = text_temp[d3[-4] + 1:d3[-3]]
            text_2 = text_temp[d3[-6] + 1:d3[-5]]
            text_1 = text_temp[d3[-8] + 1:d3[-7]]
            each[text_1] = text_2
            each[text_3] = text_4
            Y_definition_each.append(each)
        Y_definition.append(Y_definition_each)
    return Y_definition


# =============================================================================
# 以下是截取局部图片的过程
# 包括高斯热图，图片截取函数
# =============================================================================
def show_cut(image, y0, y1, x0, x1):
    """
        原图与所截区域相比较
    :param path: 图片路径
    :param left: 区块左上角位置的像素点离图片左边界的距离
    :param upper：区块左上角位置的像素点离图片上边界的距离
    :param right：区块右下角位置的像素点离图片左边界的距离
    :param lower：区块右下角位置的像素点离图片上边界的距离
     故需满足：lower > upper、right > left
    """
    img = Image.open(image)
    left = x0
    upper = y0
    right = x1
    lower = y1
    box = (left, upper, right, lower)
    roi = img.crop(box)
    plt.imshow(roi)


# =============================================================================
#   通过局部图片制作热图
# =============================================================================
def makeHeatmap_1(img, vert_posi, disc_posi):
    """
    :param img:
    :param vert_posi:
    :param disc_posi:
    :return: 用输入的img制作热图
    """
    keyposi = vert_posi + disc_posi
    kpsoi_ia = ia.KeypointsOnImage.from_xy_array(keyposi, shape=(img.shape[0], img.shape[1], 1))

    aug = iaa.Sequential([iaa.Resize(size={'height': 256, 'width': 256})])
    aug_det = aug.to_deterministic()

    img_data = img[:, :, np.newaxis]
    image_aug = aug_det.augment_image(img_data)
    kpsoi_aug = aug_det.augment_keypoints(kpsoi_ia)
    image_aug_2D = np.squeeze(image_aug, 2)

    distance_maps = kpsoi_aug.to_distance_maps()
    height, width = kpsoi_aug.shape[0:2]
    max_distance = np.linalg.norm(np.float32([0, 0]) - np.float32([height, width]))
    distance_maps_normalized = distance_maps / max_distance
    distance_maps_normalized = 1.0 - distance_maps_normalized
    keyposi_list = kpsoi_aug.to_xy_array()
    return image_aug_2D, keyposi_list, distance_maps_normalized


# =============================================================================
# 局部高斯热图
# =============================================================================
def makeGaussianMap(path, center, sigma=20, half_sz=50):
    """
    Parameters
    -heatmap: 热图（heatmap）
    - plane_idx：关键点列表中第几个关键点（决定了在热图中通道）
    - center： 关键点的位置
    - sigma: 生成高斯分布概率时的一个参数
    Returns
    - heatmap: 热图
    """
    img = Image.open(path)
    center_x, center_y = center  # mou发
    height = img.height
    width = img.width
    th = 4.6052
    delta = np.sqrt(th * 2)
    x0 = int(max(0, center_x - half_sz + 0.5))
    y0 = int(max(0, center_y - half_sz + 0.5))
    x1 = int(min(width - 1, center_x + half_sz + 0.5))
    y1 = int(min(height - 1, center_y + half_sz + 0.5))
    exp_factor = 1 / 2.0 / sigma / sigma
    heatmap = np.zeros(img.size)
    y_vec = (np.arange(y0, y1 + 1) - center_y) ** 2  # y1 included
    x_vec = (np.arange(x0, x1 + 1) - center_x) ** 2
    xv, yv = np.meshgrid(x_vec, y_vec)
    arr_sum = exp_factor * (xv + yv)
    arr_exp = np.exp(-arr_sum)
    heatmap[y0:y1 + 1, x0:x1 + 1] = arr_exp
    show_img = img * heatmap
    x0 = int(max(0, center_x - half_sz + 0.5))
    y0 = int(max(0, center_y - half_sz + 5 + 0.5))
    x1 = int(min(width - 1, center_x + half_sz + 0.5))
    y1 = int(min(height - 1, center_y + half_sz - 5 + 0.5))
    heatcut = show_img[y0:y1 + 1, x0:x1 + 1]
    return show_img, heatcut  # ,cut


# =============================================================================
# 全局高斯热图
# =============================================================================
def makeGaussianMap_all(img, center, sigma=20, half_sz=50):
    """
    Parameters
    -heatmap: 热图（heatmap）
    - plane_idx：关键点列表中第几个关键点（决定了在热图中通道）
    - center： 关键点的位置
    - sigma: 生成高斯分布概率时的一个参数
    Returns
    - heatmap: 热图
    """
    oneHeatmap = np.zeros(img.shape)
    heatmap = np.zeros(img.shape)
    for i in center:
        center_x, center_y = i
        height, width = img.shape
        th = 4.6052
        delta = np.sqrt(th * 2)
        x0 = int(max(0, center_x - half_sz + 0.5))
        y0 = int(max(0, center_y - half_sz + 0.5))
        x1 = int(min(width - 1, center_x + half_sz + 0.5))
        y1 = int(min(height - 1, center_y + half_sz + 0.5))
        exp_factor = 1 / 2.0 / sigma / sigma
        y_vec = (np.arange(y0, y1 + 1) - center_y) ** 2
        x_vec = (np.arange(x0, x1 + 1) - center_x) ** 2
        xv, yv = np.meshgrid(x_vec, y_vec)
        arr_sum = exp_factor * (xv + yv)
        arr_exp = np.exp(-arr_sum)
        oneHeatmap[y0:y1 + 1, x0:x1 + 1] = arr_exp
        heatmap = heatmap + oneHeatmap
    show_img = img * heatmap
    return heatmap, show_img


# =============================================================================
# 加载局部图片的图片数据
# =============================================================================
def load_data_svm(path, split, num, Y_position, position_name):
    X = np.zeros((num, 2904), np.float64)  # 局部图片像素信息行向量
    indx = Identification[position_name]
    # 局部热图转换成行向量保存
    for ind, file in enumerate(glob.glob(path + '/' + split + '1/data/*.jpg')):
        center = []
        center = Y_position[indx, :, ind]
        show_img, heatcut = makeGaussianMap(file, center, sigma=20, half_sz=25)
        Heatcut = cv2.resize(heatcut, (100, 100), interpolation=cv2.INTER_CUBIC)
        heat_fea = ft.hog(Heatcut, orientations=6, pixels_per_cell=[8, 8], cells_per_block=[2, 2], block_norm='L1')
        X[ind, :] = heat_fea
    X_heat = torch.from_numpy(X)
    return X_heat


# =============================================================================
# 加载局部图片的label
# =============================================================================
def load_state(Y_definition, num, position_name):
    # print('start loading ' + position_name + ' state data...')
    Y = []
    key = 0
    if (position_name == 'L5-S1' or position_name == 'L4-L5' or position_name == 'L3-L4' or position_name == 'L2-L3'
            or position_name == 'L1-L2' or position_name == 'T12-L1'):
        key = 1
    if (position_name == 'L5' or position_name == 'L4' or position_name == 'L3' or position_name == 'L2'
            or position_name == 'L1'):
        key = 2
    for i in range(num):
        for j in range(11):
            if (Y_definition[i][j]['identification'] == position_name and key == 2):
                state = Y_definition[i][j]['vertebra']
                d = [m.start() for m in re.finditer(",", state)]  # 处理存在两种状态的情况
                if d:
                    state = state[:d[0]]
                Y.append(state)
                break
            elif (Y_definition[i][j]['identification'] == position_name and key == 1):
                state = Y_definition[i][j]['disc']
                d = [m.start() for m in re.finditer(",", state)]  # 处理存在两种状态的情况
                if d:
                    state = state[:d[0]]
                Y.append(state)
                break
            else:
                if j == 10:
                    print(i)
                    print(j)
                continue
    return Y


# =============================================================================
# 加载局部图片的label
# =============================================================================
def load_state_disc(Y_defibition, num, position_name):
    key = 0
    Y = []
    if (position_name == 'L5-S1' or position_name == 'L4-L5' or position_name == 'L3-L4' or position_name == 'L2-L3'
            or position_name == 'L1-L2' or position_name == 'T12-L1'):
        key = 1
    if (key == 0):
        for arr in Y_defibition:
            for dic in arr:
                if dic['identification'] == position_name:
                    if dic['vertebra']:
                        Y.append(dic['vertebra'])
                    else:
                        dic['vertebra'] = 'v1'
                        Y.append(dic['vertebra'])
    if (key == 1):
        for arr in Y_defibition:
            for dic in arr:
                if dic['identification'] == position_name:
                    if 'disc' in dic:
                        if ',' in dic['disc']:
                            a = dic['disc'].split(',')
                            Y.append(a[1])
                        else:
                            Y.append(dic['disc'])
                    else:
                        dic['disc'] = 'v1'
                        Y.append(dic['disc'])
    return Y


# =============================================================================
# 二分类网络构建
# =============================================================================
def define_model(D_in, H_1, H_2, D_out):
    net = torch.nn.Sequential(
        torch.nn.Linear(D_in, H_1),
        torch.nn.ReLU(),
        torch.nn.Linear(H_1, H_2),
        torch.nn.ReLU(),
        torch.nn.Linear(H_2, D_out)
    )

    class Net(torch.nn.Module):
        def __init__(self, D_in, H_1, H_2, D_out):
            super(Net, self).__init__()
            self.linear1 = torch.nn.Linear(D_in, H_1)
            self.relu1 = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(H_1, H_2)
            self.relu2 = torch.nn.ReLU()
            self.linear3 = torch.nn.Linear(H_2, D_out)

        def forward(self, x):
            y1 = self.linear1(x)
            y2 = self.relu1(y1)
            y3 = self.linear2(y2)
            y4 = self.relu2(y3)
            y5 = self.linear3(y4)
            return y5

    net = Net(D_in, H_1, H_2, D_out)
    return net


# 定义损失函数
def define_loss():
    Loss = torch.nn.CrossEntropyLoss()
    return Loss


# 定义优化器
def define_optimizer():
    learning_rate = 1e-2
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    return optimizer


# 模型训练
def train(x, y, net, Loss, optimizer):
    # print("start training...")
    loss_train = 0
    key = 0
    for t in range(20):
        loss = 0
        for i in range(x.shape[0]):
            y_pred = net(x[i, :])  # 前向传播：通过像模型输入x计算预测的y
            loss_i = (y_pred - y[i]) * (y_pred - y[i])  # 计算loss
            loss = loss + loss_i
        if (loss < 1):
            break
        else:
            optimizer.zero_grad()  # 在反向传播之前，使用optimizer将它要更新的所有张量的梯度清零
            loss.backward()  # 反向传播：根据模型的参数计算loss的梯度
            optimizer.step()  # 调用Optimizer的step函数使它所有参数更新
    for i in range(x.shape[0]):
        y_pred = net(x[i, :])  # 前向传播：通过像模型输入x计算预测的y
        loss_i = (y_pred - y[i]) * (y_pred - y[i])
        if (math.fabs(loss_i) < 0.01):
            key = key + 1
        loss_train = loss_train + loss_i
    loss_train = loss_train / x.shape[0]
    print("训练完成, 损失函数为 {}".format(loss_train.item()))
    print("分类正确个数为{}".format(key))
    print("正确率为{}%".format(100 * key / x.shape[0]))

    # 输出网络参数
    net_path = 'net.pth'
    torch.save(net.state_dict(), net_path)
    return net_path


# 模型评估
def test(x, y, net_path, Loss):
    # print("start testing...")
    torch.load(net_path)
    Loss = 0
    key = 0
    for i in range(x.shape[0]):
        y_pred = net(x[i, :])
        loss_i = (y_pred - y[i]) * (y_pred - y[i])  # 计算loss
        Loss = Loss + loss_i
        if (loss_i < 0.01):
            key = key + 1
    Loss = Loss / x.shape[0]
    print("测试完成, 损失函数为 {}".format(Loss.item()))
    print("分类正确个数为{}".format(key))
    print("正确率为{}%".format(100 * key / x.shape[0]))
    return key / x.shape[0]


# =============================================================================
# 提取二分类预测结果，以列表形式存储
# =============================================================================
def out_state_2class(x, y, net_path):
    torch.load(net_path)
    Y_pred = []
    for i in range(x.shape[0]):
        y_pred = net(x[i, :])
        y_pred = y_pred.detach().numpy()
        y_int = round(y_pred[0])
        Y_pred.append(y_int)
    return Y_pred


# =============================================================================
# 数据整合函数：vertebra五合一
# =============================================================================
def from_5_to_1(L5, L4, L3, L2, L1, num):
    L_all = []
    for i in range(num):
        L5[i] = 'v' + str(L5[i])
        L4[i] = 'v' + str(L4[i])
        L3[i] = 'v' + str(L3[i])
        L2[i] = 'v' + str(L2[i])
        L1[i] = 'v' + str(L1[i])
        L_each = [L5[i], L4[i], L3[i], L2[i], L1[i]]
        L_all.append(L_each)
    return L_all


# =============================================================================
#  将五分类结果转换为对应字符串label
# =============================================================================
def Get_name(Y_predict, name):
    Y_label = []
    for label in Y_predict:
        Y_label.append(name[str(label)])
    return Y_label


# =============================================================================
# 数据整合函数：disc六合一
# =============================================================================
def from_6_to_1(L6, L5, L4, L3, L2, L1, num):
    L_all = []
    for i in range(num):
        L_each = []
        L_each.append(L6[i])
        L_each.append(L5[i])
        L_each.append(L4[i])
        L_each.append(L3[i])
        L_each.append(L2[i])
        L_each.append(L1[i])
        L_all.append(L_each)
    return L_all


# =============================================================================
# 数据清洗，修改不符合规范的存储label的txt文件
# =============================================================================
def data_wash(path, split):
    # print('start washing' + split + 'data...')
    filename = glob.glob(path + '/' + split + '/data/*.txt')
    for ind, file in enumerate(filename):
        dd = filename[ind].split('\\')
        Y_identification_each = []
        cf = open(file, 'r')
        lines = cf.readlines()
        for info in lines:
            each = {}
            c1 = [m.start() for m in re.finditer(',', info)]
            text_cord = info[c1[0] - 3:c1[1] + 1]
            d1 = [m.start() for m in re.finditer('{', info)]
            d2 = [m.start() for m in re.finditer('}', info)]
            text_content = info[d1[0] + 1:d2[0]]
            d3 = [m.start() for m in re.finditer("'", text_content)]
            if (len(d3) == 12):
                for i in range(12):
                    if d3[i + 1] - d3[i] == 1:
                        del d3[i - 2:i + 2]
                        break
            text_4 = text_content[d3[-2] + 1:d3[-1]]
            text_3 = text_content[d3[-4] + 1:d3[-3]]
            text_2 = text_content[d3[-6] + 1:d3[-5]]
            text_1 = text_content[d3[-8] + 1:d3[-7]]
            each[text_1] = text_2
            each[text_3] = text_4
            text_label = str(each)
            text = text_cord + text_label
            Y_identification_each.append(text)
        f_2 = open(path + '/' + split + '_washed' + '/label/' + dd[-1], 'wb')
        for t in Y_identification_each:
            t = str(t) + '\n'
            t = t.encode("utf-8")
            f_2.write(t)
        f_2.close()
        cf.close()
    # print('finish washing' + split + 'data...')


# =============================================================================
# 多分类网络构建
# =============================================================================
def get_data(XTRAIN, YTRAIN, split, num, name):
    Y = np.zeros((num,), np.int)
    for ind, file in enumerate(XTRAIN):
        label = YTRAIN[ind]
        if label == '':
            Y[ind] = 1
        else:
            Y[ind] = name[label]
    return Y


# 模型训练
def classifier_train(X_train, Y_train):
    clf = svm.SVC()
    clf.fit(X_train, Y_train)
    return clf


# 模型评估
def classifier_test(clf, X_test, Y_test):
    Y_predicted = clf.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(Y_test, Y_predicted)
    acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum() * 100
    return confusion_matrix, acc, Y_predicted


# =============================================================================
# 使用pytorch搭建卷积神经网络
# =============================================================================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(  # 创建时序容器
            nn.Conv2d(1, 4, kernel_size=3),
            nn.BatchNorm2d(4),  # 归一化
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=5)
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * 25 * 25, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 22),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)  # 将上一次的卷积结果拉伸为一行
        x = self.fc(x)
        return x


# =============================================================================
# 将图片的大小统一修改为256*256并存储在train1或test1中的data文件夹，统一图片大小方便后面输入网络中
# =============================================================================
def Resize_Image(split, path):
    filename = glob.glob(path + '/' + split + '/data/*.jpg')
    for ind, file in enumerate(filename):
        dd = filename[ind].split('\\')
        im = Image.open(file)
        y_s = 256
        x_s = 256
        out = im.resize((x_s, y_s), Image.ANTIALIAS)
        outfile = path + '/' + split + '2/data/' + dd[-1]
        out.save(outfile)


# =============================================================================
# 读取经过大小调整的图片
# =============================================================================
def Load_Image(split, path, num):
    ii = np.zeros((num, 256, 256), np.int64)
    count = 0
    filename = glob.glob(path + '/' + split + '2/data/*.jpg')
    for ind, file in enumerate(filename):
        im = Image.open(file)
        matrix = np.array(im)
        ii[count, :, :] = matrix.copy()
        count += 1
    ii = np.expand_dims(ii, 1)
    ii = torch.from_numpy(ii)
    ii = ii.double()
    return ii


# =============================================================================
#  将最终结果以txt格式存储在test2中的final_pred文件夹
# =============================================================================
def Output_Final(split, path, xy, ver, disc):
    xy = xy.tolist()
    count = 0
    filename = glob.glob(path + '/' + split + '2/label/*.txt')
    ID = ['T12-L1', 'L1', 'L1-L2', 'L2', 'L2-L3', 'L3', 'L3-L4', 'L4', 'L4-L5', 'L5', 'L5-S1']
    ver_disc = []
    for i in range(len(ver)):
        v_d = [disc[i][0], ver[i][0], disc[i][1], ver[i][1], disc[i][2], ver[i][2], disc[i][3], ver[i][3], disc[i][4],
               ver[i][4], disc[i][5]]
        ver_disc.append(v_d)
    for ind, file in enumerate(filename):
        dd = filename[ind].split('\\')
        f = open(path + '/' + split + '2/final_pred/' + dd[-1], 'wb')
        for i in range(11):
            line1 = str(xy[count][i + 11]) + ',' + str(xy[count][i]) + ','
            if i % 2 == 0:
                line2 = "{'identification': '" + ID[i] + "', 'disc': '" + ver_disc[ind][i] + "'}\n"
            else:
                line2 = "{'identification': '" + ID[i] + "', 'vertebra': '" + ver_disc[ind][i] + "'}\n"
            line = line1 + line2
            line = line.encode("utf-8")
            f.write(line)
        f.close()
        count += 1


# =============================================================================
#  将预测坐标输出并以txt格式存储在train1或test1中的pred文件夹
# =============================================================================
def Output_XY(split, path, xy):
    xy = xy.tolist()
    count = 0
    filename = glob.glob(path + '/' + split + '2/label/*.txt')
    for ind, file in enumerate(filename):
        dd = filename[ind].split('\\')
        f = open(path + '/' + split + '2/pred/' + dd[-1], 'wb')
        for i in range(11):
            line = str(xy[count][i + 11]) + ',' + str(xy[count][i]) + '\n'
            line = line.encode("utf-8")
            f.write(line)
        f.close()
        count += 1


# =============================================================================
#  读取标记好的label并储存在一个ndarray中
# =============================================================================
def Load_Y_Data(path, split, num):
    Y_1 = np.zeros((num, 11, 2), np.int64)
    for ind, file in enumerate(glob.glob(path + '/' + split + '/data/*.txt')):
        df = open(file, 'r')
        lines = df.readlines()
        j = 0
        for i in lines:
            d = [m.start() for m in re.finditer(',', i)]
            Y_1[ind, j, 0] = eval(i[0:d[0]])
            Y_1[ind, j, 1] = eval(i[d[0] + 1:d[1]])
            j = j + 1
        df.close()
    return Y_1


# =============================================================================
# 变换坐标信息，使之对应调整至256x256大小的图片，存储至新的ndarray中并返回
# =============================================================================
def Match_Position(split, path, Y_position, num):
    # Y_position为各个图片中的脊椎位置的三维数组，大小为11*2，有150层,输出YY为22的列向量，前11项为Y坐标从大到小排列，后11项为对应X坐标
    Y_position_new = np.zeros((num, 11, 2), np.double)
    YY = np.zeros((num, 22), np.double)
    for ind, file in enumerate(glob.glob(path + '/' + split + '/data/*.jpg')):
        im = Image.open(file)
        x, y = im.size
        for j in range(11):
            Y_position_new[ind, j, 0] = (Y_position[ind, j, 0] * 256 / x)
            Y_position_new[ind, j, 1] = (Y_position[ind, j, 1] * 256 / y)
    for num in range(num):
        temp = Y_position_new[num, :, 1].copy()
        index = np.argsort(-temp)
        YY[num, 0:11] = Y_position_new[num, index, 1]
        YY[num, 11:22] = Y_position_new[num, index, 0]
        # 坐标归一化处理，提高训练效果
        for k in range(22):
            YY[num, k] /= 256
    YY = torch.from_numpy(YY)
    YY = YY.double()
    return YY


# =============================================================================
# 将256x256下预测坐标变换成原图坐标系中的坐标,用ndarray存储
# =============================================================================
def Match_Original_Position(split, path, xy, num):
    original_pos = np.zeros((num, 22), np.int64)
    for ind, file in enumerate(glob.glob(path + '/' + split + '/data/*.jpg')):
        im = Image.open(file)
        x, y = im.size
        for j in range(11):
            original_pos[ind, j] = int(round(xy[ind, j] * y))
        for j in range(11, 22):
            original_pos[ind, j] = int(round(xy[ind, j] * x))
    return original_pos


# =============================================================================
#  按照预测点y值从大到小的顺序重新排序label数据，并存储在新的txt文件中，存储路径为test1中的label文件夹
# =============================================================================
def Restore_Label(split, path):
    filename = glob.glob(path + '/' + split + '/data/*.txt')
    for ind, file in enumerate(filename):
        dd = filename[ind].split('\\')
        f = open(file, 'r')
        Text = f.readlines()
        text = []
        ntext = []
        f_2 = open(path + '/' + split + '2/label/' + dd[-1], 'wb')
        y = np.zeros((11, 1), np.int64)
        l = 0
        for t in Text:
            t = str(t)
            d = [m.start() for m in re.finditer(',', t)]
            y[l] = int(t[d[0] + 1:d[1]])
            l += 1
            text.append(t)
        index = np.argsort(-y, axis=0)
        for i in range(11):
            ntext.append(text[index[i, 0]])
        for i in range(11):
            nline = ntext[i].encode("utf-8")
            f_2.write(nline)
        f_2.close()
        f.close()


# =============================================================================
#  测试集正确率计算（规定坐标误差的绝对值在15以内为正确）
# =============================================================================
def Get_Acc(split, path, num):
    accs = np.zeros((num, 1), np.double)
    p = np.zeros((num, 11, 2), np.int64)
    l = np.zeros((num, 11, 2), np.int64)
    filename = glob.glob(path + '/' + split + '2/pred/*.txt')
    for ind, file in enumerate(filename):
        dd = filename[ind].split('\\')
        count = 0
        f = open(file, 'r')
        p_cord = f.readlines()
        for t in p_cord:
            d = [m.start() for m in re.finditer(',', t)]
            p[ind, count, 0] = int(t[0:d[0]])
            p[ind, count, 1] = int(t[d[0] + 1:len(t)])
            count += 1
        count = 0
        f_2 = open(path + '/' + split + '2/label/' + dd[-1], 'r')
        l_cord = f_2.readlines()
        for t in l_cord:
            d = [m.start() for m in re.finditer(',', t)]
            l[ind, count, 0] = int(t[0:d[0]])
            l[ind, count, 1] = int(t[d[0] + 1:d[1]])
            count += 1
        f_2.close()
        f.close()
    for i in range(num):
        count = 0
        for j in range(11):
            if abs(p[i, j, 0] - l[i, j, 0]) <= 15:
                count += 1
            if abs(p[i, j, 1] - l[i, j, 1]) <= 15:
                count += 1
        accs[i] = count / 22
    acc = accs.sum() / num
    return acc


# =============================================================================
#  导出训练好的卷积神经网络到指定路径
# =============================================================================
def Save_Whole_Net(Net, path):
    torch.save(Net(), path + '/Pos_Pred_CNN.pkl')


# =============================================================================
#  导出训练好的卷积神经网络的参数到指定路径
# =============================================================================
def Save_Net_Params(Net, path):
    torch.save(Net().state_dict(), path + '/Pos_Pred_CNN_Params.pkl')


# =============================================================================
#                              主程序
# =============================================================================
if __name__ == '__main__':

    # 程序计时开始
    start = time.time()

    # 定义一些超参数
    batch_size = 10
    learning_rate = 0.000038  # adam
    num_epoches = 20

    #  环境准备，创建若干空文件夹
    path = os.getcwd()  # 获取当前文件的目录，故需将测试集及训练集与此文件放在一个文件夹中
    split = '/train2'
    if not os.path.exists(path + split):
        os.makedirs(path + split + '/data')
    split = '/test2'
    if not os.path.exists(path + split):
        os.makedirs(path + split + '/data')
        os.makedirs(path + split + '/label')
        os.makedirs(path + split + '/pred')
        os.makedirs(path + split + '/final_pred')
    split = '/train1'
    if not os.path.exists(path + split):
        os.makedirs(path + split + '/data')
    split = '/test1'
    if not os.path.exists(path + split):
        os.makedirs(path + split + '/data')
    split = '/train_washed'
    if not os.path.exists(path + split):
        os.makedirs(path + split + '/label')
    split = '/test_washed'
    if not os.path.exists(path + split):
        os.makedirs(path + split + '/label')

    # =============================================================================
    #      一、坐标预测
    # =============================================================================
    # 训练集数据预处理
    split = 'train'
    Train_num = 149
    Test_num = 51
    YY_1 = Load_Y_Data(path, split, Train_num)
    Resize_Image(split, path)
    Train_y = Match_Position(split, path, YY_1, Train_num)
    Train_x = Load_Image(split, path, Train_num)

    # 测试集数据预处理
    split = 'test'
    YY_2 = Load_Y_Data(path, split, Test_num)
    Resize_Image(split, path)
    Test_y = Match_Position(split, path, YY_2, Test_num)
    Restore_Label(split, path)  # 用于后续计算预测结果正确率以评价模型
    Test_x = Load_Image(split, path, Test_num)

    # 数据集封装
    Train_data = TensorDataset(Train_x, Train_y)
    Train_loader = DataLoader(Train_data, batch_size=batch_size, shuffle=False)
    Test_data = TensorDataset(Test_x, Test_y)
    Test_loader = DataLoader(Test_data, batch_size=batch_size, shuffle=False)

    # 选择模型
    Model = CNN()
    if torch.cuda.is_available():
        Model = Model.cuda()

    # 定义损失函数和优化器
    Criterion = nn.MSELoss()
    Optimizer = optim.Adam(Model.parameters(), lr=learning_rate,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=0,
                           amsgrad=False)

    # 训练模型
    Epoch = 0
    for data in Train_loader:
        img, label = data
        img = Variable(img)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = Model(img)
        loss = Criterion(out, label)
        print_loss = loss.data.item()
        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()
        Epoch += 1

    # 模型评估
    Pred = np.zeros([51, 22])
    count = 0
    Model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in Test_loader:
        img, label = data
        img = Variable(img)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        out = Model(img)
        for xy in range(list(out.size())[0]):
            for i in range(22):
                Pred[count, i] = out[xy, i]
            count += 1
        loss = Criterion(out, label)
        eval_loss += loss.data.item() * label.size(0)
        num_correct = (out == label).sum()
    split = 'test'
    Pred = Match_Original_Position(split, path, Pred, Test_num)
    Output_XY(split, path, Pred)
    Acc = Get_Acc(split, path, Test_num)
    print('坐标预测部分:\n')
    print('正确率: ' + str(100 * Acc) + '%')
    print('22个坐标值的MSELoss: {:.6f}'.format(eval_loss / (len(Test_data))))
    print('11个预测点的MSELoss: {:.6f}'.format(2 * eval_loss / (len(Test_data))))

    # # 模型/模型参数导出，可选代码段
    # Save_Whole_Net(CNN, path)
    # Save_Net_Params(CNN, path)

    # =============================================================================
    #      二、分类任务
    # =============================================================================
    print('=====正在进行数据预处理......=====')
    # 去除随机性
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministric = True
    torch.backends.cudnn.benchamark = False
    torch.set_default_tensor_type(torch.DoubleTensor)

    # 数据预处理
    path = os.getcwd()  # 获取当前文件的目录，故需将测试集及训练集与此文件放在一个文件夹中
    State_name = {'v1': 1, 'v2': 2, 'v3': 3, 'v4': 4, 'v5': 5}
    Down_name = {'1': 'v1', '2': 'v2', '3': 'v3', '4': 'v4', '5': 'v5'}
    Identification = {'L5-S1': 0, 'L5': 1, 'L4-L5': 2, 'L4': 3, 'L3-L4': 4, 'L3': 5,
                      'L2-L3': 6, 'L2': 7, 'L1-L2': 8, 'L1': 9, 'T12-L1': 10}
    split_1w = 'train_washed'
    splitw = 'test_washed'
    split_1 = 'train'
    split = 'test'
    data_wash(path, split_1)
    data_wash(path, split)
    all_acc = []

    # =============================================================================
    #      （1）训练集的X_train的最终结果———tensor形式
    # =============================================================================
    Y_1 = load_Y_data(path, split_1w, num=149)
    resize_image(split_1, path)
    Y_position = rechange_position(split_1, path, Y_1, num=149)
    Y_definition = load_definition(path, split_1w, num=149)
    Y_1_sort = sort_position(Y_position, num=149)
    X_L5_S1 = load_data_svm(path, split_1, 149, Y_1_sort, 'L5-S1')
    X_L5 = load_data_svm(path, split_1, 149, Y_1_sort, 'L5')
    X_L4_L5 = load_data_svm(path, split_1, 149, Y_1_sort, 'L4-L5')
    X_L4 = load_data_svm(path, split_1, 149, Y_1_sort, 'L4')
    X_L3_L4 = load_data_svm(path, split_1, 149, Y_1_sort, 'L3-L4')
    X_L3 = load_data_svm(path, split_1, 149, Y_1_sort, 'L3')
    X_L2_L3 = load_data_svm(path, split_1, 149, Y_1_sort, 'L2-L3')
    X_L2 = load_data_svm(path, split_1, 149, Y_1_sort, 'L2')
    X_L1_L2 = load_data_svm(path, split_1, 149, Y_1_sort, 'L1-L2')
    X_L1 = load_data_svm(path, split_1, 149, Y_1_sort, 'L1')
    X_T12_L1 = load_data_svm(path, split_1, 149, Y_1_sort, 'T12-L1')

    # =============================================================================
    #      （2）测试集的输入数据———tensor形式
    # =============================================================================
    Y_test = load_Y_data(path, splitw, num=51)
    resize_image(split, path)
    Y_position_test = rechange_position(split, path, Y_test, num=51)
    Y_test_sort = sort_position(Y_position_test, num=51)
    Y_definition_test = load_definition(path, splitw, num=51)
    #     每个位置的小图片的图片信息数据集
    #     五分类的输入数据
    X_L5_S1_test = load_data_svm(path, split, 51, Y_test_sort, 'L5-S1')
    X_L4_L5_test = load_data_svm(path, split, 51, Y_test_sort, 'L4-L5')
    X_L3_L4_test = load_data_svm(path, split, 51, Y_test_sort, 'L3-L4')
    X_L2_L3_test = load_data_svm(path, split, 51, Y_test_sort, 'L2-L3')
    X_L1_L2_test = load_data_svm(path, split, 51, Y_test_sort, 'L1-L2')
    X_T12_L1_test = load_data_svm(path, split, 51, Y_test_sort, 'T12-L1')
    #     二分类的输入数据
    X_L5_test = load_data_svm(path, split, 51, Y_test_sort, 'L5')
    X_L4_test = load_data_svm(path, split, 51, Y_test_sort, 'L4')
    X_L3_test = load_data_svm(path, split, 51, Y_test_sort, 'L3')
    X_L2_test = load_data_svm(path, split, 51, Y_test_sort, 'L2')
    X_L1_test = load_data_svm(path, split, 51, Y_test_sort, 'L1')

    # =============================================================================
    #      （3）二分类的输出数据
    # =============================================================================
    #     训练数据
    Y_L5 = load_state(Y_definition, 149, 'L5')
    Y_L4 = load_state(Y_definition, 149, 'L4')
    Y_L3 = load_state(Y_definition, 149, 'L3')
    Y_L2 = load_state(Y_definition, 149, 'L2')
    Y_L1 = load_state(Y_definition, 149, 'L1')
    Y_L5_value = []
    for i in range(149):
        Y_L5_value.append(State_name[Y_L5[i]])
    Y_L4_value = []
    for i in range(149):
        Y_L4_value.append(State_name[Y_L4[i]])
    Y_L3_value = []
    for i in range(149):
        Y_L3_value.append(State_name[Y_L3[i]])
    Y_L2_value = []
    for i in range(149):
        Y_L2_value.append(State_name[Y_L2[i]])
    Y_L1_value = []
    for i in range(149):
        Y_L1_value.append(State_name[Y_L1[i]])
    L5_valuetensor = torch.tensor(Y_L5_value)
    L4_valuetensor = torch.tensor(Y_L4_value)
    L3_valuetensor = torch.tensor(Y_L3_value)
    L2_valuetensor = torch.tensor(Y_L2_value)
    L1_valuetensor = torch.tensor(Y_L1_value)
    print('二分类的训练数据加载完成！')
    #     测试数据
    Y_L5_test = load_state(Y_definition_test, 51, 'L5')
    Y_L4_test = load_state(Y_definition_test, 51, 'L4')
    Y_L3_test = load_state(Y_definition_test, 51, 'L3')
    Y_L2_test = load_state(Y_definition_test, 51, 'L2')
    Y_L1_test = load_state(Y_definition_test, 51, 'L1')
    Y_L5_value_test = []
    for i in range(51):
        Y_L5_value_test.append(State_name[Y_L5_test[i]])
    Y_L4_value_test = []
    for i in range(51):
        Y_L4_value_test.append(State_name[Y_L4_test[i]])
    Y_L3_value_test = []
    for i in range(51):
        Y_L3_value_test.append(State_name[Y_L3_test[i]])
    Y_L2_value_test = []
    for i in range(51):
        Y_L2_value_test.append(State_name[Y_L2_test[i]])
    Y_L1_value_test = []
    for i in range(51):
        Y_L1_value_test.append(State_name[Y_L1_test[i]])
    L5_valuetensor_test = torch.tensor(Y_L5_value_test)
    L4_valuetensor_test = torch.tensor(Y_L4_value_test)
    L3_valuetensor_test = torch.tensor(Y_L3_value_test)
    L2_valuetensor_test = torch.tensor(Y_L2_value_test)
    L1_valuetensor_test = torch.tensor(Y_L1_value_test)
    print('二分类的测试数据加载完成！')

    # =============================================================================
    #      （4）五分类的输出数据
    # =============================================================================
    Y_L5_S1_disc = load_state_disc(Y_definition, 149, 'L5-S1')
    Y_L4_L5_disc = load_state_disc(Y_definition, 149, 'L4-L5')
    Y_L3_L4_disc = load_state_disc(Y_definition, 149, 'L3-L4')
    Y_L2_L3_disc = load_state_disc(Y_definition, 149, 'L2-L3')
    Y_L1_L2_disc = load_state_disc(Y_definition, 149, 'L1-L2')
    Y_T12_L1_disc = load_state_disc(Y_definition, 149, 'T12-L1')
    Y_L5_disc = load_state_disc(Y_definition, 149, 'L5')
    print('五分类的训练数据加载完成！')
    Y_L5_S1_test = load_state_disc(Y_definition_test, 51, 'L5-S1')
    Y_L4_L5_test = load_state_disc(Y_definition_test, 51, 'L4-L5')
    Y_L3_L4_test = load_state_disc(Y_definition_test, 51, 'L3-L4')
    Y_L2_L3_test = load_state_disc(Y_definition_test, 51, 'L2-L3')
    Y_L1_L2_test = load_state_disc(Y_definition_test, 51, 'L1-L2')
    Y_T12_L1_test = load_state_disc(Y_definition_test, 51, 'T12-L1')
    Y_L5_test = load_state_disc(Y_definition_test, 51, 'L5')
    print('五分类的测试数据加载完成！')
    print('=====数据预处理完成!=====')

    # =============================================================================
    #      训练pytorch二分类模型
    # =============================================================================
    print('=====正在进行二分类模型训练......======')
    D_in, H_1, H_2, D_out = 2904, 10, 10, 1
    net = define_model(D_in, H_1, H_2, D_out)
    Loss = define_loss()
    optimizer = define_optimizer()
    print("开始训练L5二分类模型")
    net_path_L5 = train(X_L5, L5_valuetensor, net, Loss, optimizer)
    print("开始训练L4二分类模型")
    net_path_L4 = train(X_L4, L4_valuetensor, net, Loss, optimizer)
    print("开始训练L3二分类模型")
    net_path_L3 = train(X_L3, L3_valuetensor, net, Loss, optimizer)
    D_in, H_1, H_2, D_out = 2904, 20, 10, 1
    net = define_model(D_in, H_1, H_2, D_out)
    Loss = define_loss()
    optimizer = define_optimizer()
    print("开始训练L2二分类模型")
    net_path_L2 = train(X_L2, L2_valuetensor, net, Loss, optimizer)
    D_in, H_1, H_2, D_out = 2904, 16, 10, 1
    net = define_model(D_in, H_1, H_2, D_out)
    Loss = define_loss()
    optimizer = define_optimizer()
    print("开始训练L1二分类模型")
    net_path_L1 = train(X_L1, L1_valuetensor, net, Loss, optimizer)
    print('=====5组二分类模型训练完成！=====')

    # =============================================================================
    #      测试训练好的pytorch二分类模型的效果
    # =============================================================================
    print('=====正在进行二分类模型效果测试......=====')
    print("开始测试L5二分类模型")
    L5_acc = test(X_L5_test, L5_valuetensor_test, net_path_L5, Loss)
    all_acc.append(L5_acc*100)
    print("开始测试L4二分类模型")
    L4_acc = test(X_L4_test, L4_valuetensor_test, net_path_L4, Loss)
    all_acc.append(L4_acc*100)
    print("开始测试L3二分类模型")
    L3_acc = test(X_L3_test, L3_valuetensor_test, net_path_L3, Loss)
    all_acc.append(L3_acc*100)
    print("开始测试L2二分类模型")
    L2_acc = test(X_L2_test, L2_valuetensor_test, net_path_L2, Loss)
    all_acc.append(L2_acc*100)
    print("开始测试L1二分类模型")
    L1_acc = test(X_L1_test, L1_valuetensor_test, net_path_L1, Loss)
    all_acc.append(L1_acc*100)

    # # =============================================================================
    # #     输出预测结果的列表
    # # =============================================================================
    L5_class = out_state_2class(X_L5_test, L5_valuetensor_test, net_path_L5)
    L4_class = out_state_2class(X_L4_test, L4_valuetensor_test, net_path_L4)
    L3_class = out_state_2class(X_L3_test, L3_valuetensor_test, net_path_L3)
    L2_class = out_state_2class(X_L2_test, L2_valuetensor_test, net_path_L2)
    L1_class = out_state_2class(X_L1_test, L1_valuetensor_test, net_path_L1)
    L5_TO_L1_CLASS = from_5_to_1(L1_class, L2_class, L3_class, L4_class, L5_class, 51)

    # =============================================================================
    #      训练五分类模型
    # =============================================================================
    # =============================================================================
    #      五分类输入数据格式转换，tensor转成numpy
    # =============================================================================
    X_cifar_train_1 = X_L5_S1.numpy()
    X_cifar_train_2 = X_L4_L5.numpy()
    X_cifar_train_3 = X_L3_L4.numpy()
    X_cifar_train_4 = X_L2_L3.numpy()
    X_cifar_train_5 = X_L1_L2.numpy()
    X_cifar_train_6 = X_T12_L1.numpy()

    X_cifar_test_1 = X_L5_S1_test.numpy()
    X_cifar_test_2 = X_L4_L5_test.numpy()
    X_cifar_test_3 = X_L3_L4_test.numpy()
    X_cifar_test_4 = X_L2_L3_test.numpy()
    X_cifar_test_5 = X_L1_L2_test.numpy()
    X_cifar_test_6 = X_T12_L1_test.numpy()

    # =============================================================================
    #      读取训练集和测试集的LABEL
    # =============================================================================
    Y_train_1 = get_data(X_cifar_train_1, Y_L5_S1_disc, split_1w, num=149, name=State_name)
    Y_train_2 = get_data(X_cifar_train_2, Y_L4_L5_disc, split_1w, num=149, name=State_name)
    Y_train_3 = get_data(X_cifar_train_3, Y_L3_L4_disc, split_1w, num=149, name=State_name)
    Y_train_4 = get_data(X_cifar_train_4, Y_L2_L3_disc, split_1w, num=149, name=State_name)
    Y_train_5 = get_data(X_cifar_train_5, Y_L1_L2_disc, split_1w, num=149, name=State_name)
    Y_train_6 = get_data(X_cifar_train_6, Y_T12_L1_disc, split_1w, num=149, name=State_name)

    Y_test_1 = get_data(X_cifar_test_1, Y_L5_S1_test, splitw, num=51, name=State_name)
    Y_test_2 = get_data(X_cifar_test_2, Y_L4_L5_test, splitw, num=51, name=State_name)
    Y_test_3 = get_data(X_cifar_test_3, Y_L3_L4_test, splitw, num=51, name=State_name)
    Y_test_4 = get_data(X_cifar_test_4, Y_L2_L3_test, splitw, num=51, name=State_name)
    Y_test_5 = get_data(X_cifar_test_5, Y_L1_L2_test, splitw, num=51, name=State_name)
    Y_test_6 = get_data(X_cifar_test_6, Y_T12_L1_test, splitw, num=51, name=State_name)

    # =============================================================================
    #      多分类器训练及输出混淆矩阵及准确率
    # =============================================================================
    clf_1 = classifier_train(X_cifar_train_1, Y_train_1)
    confusion_matrix, acc, Y_predicted_1 = classifier_test(clf_1, X_cifar_test_1, Y_test_1)
    print("五分类模型测试集效果评估：")
    print("L5_S1的五分类模型混淆矩阵：")
    print(confusion_matrix)
    print('L5_S1 正确率: ' + str(acc) + '%\n')
    all_acc.append(acc)
    clf_2 = classifier_train(X_cifar_train_2, Y_train_2)
    confusion_matrix, acc, Y_predicted_2 = classifier_test(clf_2, X_cifar_test_2, Y_test_2)
    print("L4_L5的五分类模型混淆矩阵：")
    print(confusion_matrix)
    print('L4_L5 正确率: ' + str(acc) + '%\n')
    all_acc.append(acc)
    clf_3 = classifier_train(X_cifar_train_3, Y_train_3)
    confusion_matrix, acc, Y_predicted_3 = classifier_test(clf_3, X_cifar_test_3, Y_test_3)
    print("L3_L4的五分类模型混淆矩阵：")
    print(confusion_matrix)
    print('L3_L4 正确率: ' + str(acc) + '%\n')
    all_acc.append(acc)
    clf_4 = classifier_train(X_cifar_train_4, Y_train_4)
    confusion_matrix, acc, Y_predicted_4 = classifier_test(clf_4, X_cifar_test_4, Y_test_4)
    print("L2_L3的五分类模型混淆矩阵：")
    print(confusion_matrix)
    print('L2_L3 正确率: ' + str(acc) + '%\n')
    all_acc.append(acc)
    clf_5 = classifier_train(X_cifar_train_5, Y_train_5)
    confusion_matrix, acc, Y_predicted_5 = classifier_test(clf_5, X_cifar_test_5, Y_test_5)
    print("L1_L2的五分类模型混淆矩阵：")
    print(confusion_matrix)
    print('L1_L2 正确率: ' + str(acc) + '%\n')
    all_acc.append(acc)
    clf_6 = classifier_train(X_cifar_train_6, Y_train_6)
    confusion_matrix, acc, Y_predicted_6 = classifier_test(clf_6, X_cifar_test_6, Y_test_6)
    print("T12_L1的五分类模型混淆矩阵：")
    print(confusion_matrix)
    print('T12_L1 正确率: ' + str(acc) + '%\n')
    all_acc.append(acc)

    Label_L5_S1 = Get_name(Y_predicted_1, Down_name)
    Label_L4_L5 = Get_name(Y_predicted_2, Down_name)
    Label_L3_l4 = Get_name(Y_predicted_3, Down_name)
    Label_L2_L3 = Get_name(Y_predicted_4, Down_name)
    Label_L1_L2 = Get_name(Y_predicted_5, Down_name)
    Label_T12_L1 = Get_name(Y_predicted_6, Down_name)
    DISC_CLASS = from_6_to_1(Label_T12_L1, Label_L1_L2, Label_L2_L3, Label_L3_l4, Label_L4_L5, Label_L5_S1, 51)

    # 11个分类任务测试集平均正确率
    print('11个分类任务测试集平均正确率:' + str(sum(all_acc)/11) + '%\n')

    # =============================================================================
    #      坐标定位与分类任务预测结果输出(txt形式)
    # =============================================================================
    Output_Final(split, path, Pred, L5_TO_L1_CLASS, DISC_CLASS)

    # 程序计时结束
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
