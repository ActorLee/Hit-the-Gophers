import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import screeninfo
import os
import random


block_row=0
block_col=0
list=[[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]

screen = screeninfo.get_monitors()[0]
width, height = screen.width, screen.height


# path="C:/Users/lyw/dataapp/data/train_3x3/"
# use_gpu = torch.cuda.is_available()
#
# def to_tensor(img):
#     img = torch.from_numpy(img.transpose((2, 0, 1)))
#     return img.float().div(255)
#
#
#
# class net_full_img(nn.Module):
#     def __init__(self):
#         super(net_full_img, self).__init__()
#
#         self.conv1 = nn.Conv2d(3, 30, kernel_size=5, stride=1, padding=0)
#         self.conv2 = nn.Conv2d(30, 50, kernel_size=5, stride=1, padding=0)
#         self.fc1 = nn.Linear(140450, 500)  # 2809*n
#         self.fc2 = nn.Linear(500, 9)
#
#         self._initialize_weight()
#
#
#     def forward(self, x):
#         x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
#         x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
#         x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True)
#         x=F.dropout(x,0.6)
#         # x = torch.cat([x, y], dim=1)
#         x = self.fc2(x)
#         return x
#
#
#
# class test_dataset():
#     def __init__(self, path=""):
#         self.path = path
#         self.data = open(path + "label_test_new.txt").readlines()
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         img = cv2.imread(self.path +"face_data/"+self.data[index][:15])
#         img = cv2.resize(img, (224, 224), 0, 0, cv2.INTER_LINEAR)
#         img2 = to_tensor(img)
#         label = int(self.data[index][16:])
#         return img2, label
#
#
# test_data=test_dataset(path=path)
# test_loader=DataLoader(test_data, batch_size=1)
# model=net_full_img()
# criterion = nn.CrossEntropyLoss(size_average=False)
#
# def test():
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:
#         if use_gpu:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data, volatile=True), Variable(target)
#         output = model(data)
#         test_loss += criterion(output, target).item()
#         pred = output.data.max(1, keepdim=True)[1]
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#
#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
#        test_loss, correct, len(test_loader.dataset),
#        100. * float(correct) / float(len(test_loader.dataset))))
#
#
#
# if use_gpu:
#     model = model.cuda()
#     print('USE GPU')
# else:
#     print('USE CPU')
#
# model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))  # multi gpu
# model.load_state_dict(torch.load("C:/Users/lyw/dataapp/model/91.25_30x50x500_face_SGD_dropout.pkl"))

dishu_width=280
dishu_height=180
img_dishu=cv2.imread("C:/Users/lyw/dataapp/dishu.jpg")
img_dishu=cv2.resize(img_dishu,(dishu_width, dishu_height), 0, 0, cv2.INTER_LINEAR)
img_bg = np.ones((height, width, 3), np.uint8)
h,w,c=img_dishu.shape
def draw_dishu():
    global block_col, block_row,img_bg,h,w,c
    img_bg = np.ones((height, width, 3), np.uint8)
    loc = random.sample(list, 1)
    block_row = loc[0][0]
    block_col = loc[0][1]
    for i in range(0, h):
        for j in range(0, w):
            for k in range(0, 3):
                img_bg[i+int(height / 3 * block_row)+30][j+int(width / 3 * block_col+70)][k] = img_dishu[i][j][k]
    cv2.imshow("test",img_bg)
    cv2.waitKey(1000)


while(1):
    draw_dishu()





