import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt

path="/home/liyouwei/gaze/train_3x3/"
model_num=0
base_lr=0.000001
epoch_num=1000
Loss_list = []
Accuracy_list = []
batch_size=128
num_workers=10
use_gpu = torch.cuda.is_available()

def to_tensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0)

class net_full_img(nn.Module):
    def __init__(self):
        super(net_full_img, self).__init__()

        self.conv1 = nn.Conv2d(3, 30, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(30, 50, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(140450, 500)  # 2809*n
        self.fc2 = nn.Linear(500, 9)

        self._initialize_weight()

    def _initialize_weight(self):
        nn.init.normal_(self.conv1.weight, mean=0, std=0.1)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        self.apply(initialize_weights)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True)
        x=F.dropout(x,0.6)
        # x = torch.cat([x, y], dim=1)
        x = self.fc2(x)
        return x

model=net_full_img()


if use_gpu:
    model = model.cuda()
    print('USE GPU')
else:
    print('USE CPU')

model._initialize_weight()

model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))  # multi gpu
model.load_state_dict(torch.load("/home/liyouwei/gaze/train_3x3/new_example/model/500_30x50x500_face_SGD_dropout.pkl"))

class train_dataset():
    def __init__(self, path=""):
        self.path = path
        self.data = open(path + "label_train_new.txt").readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = cv2.imread(self.path +"face_data/"+ self.data[index][:15])
        img = cv2.resize(img, (224, 224), 0, 0, cv2.INTER_LINEAR)
        img2 = to_tensor(img)
        label = int(self.data[index][16:])
        return img2, label

class test_dataset():
    def __init__(self, path=""):
        self.path = path
        self.data = open(path + "label_test_new.txt").readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = cv2.imread(self.path +"face_data/"+self.data[index][:15])
        img = cv2.resize(img, (224, 224), 0, 0, cv2.INTER_LINEAR)
        img2 = to_tensor(img)
        label = int(self.data[index][16:])
        return img2, label

train_data = train_dataset(path=path)
test_data=test_dataset(path=path)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=num_workers)
test_loader=DataLoader(test_data, batch_size=batch_size,num_workers=num_workers)


# def adjust_learning_rate(optimizer, epoch):
#     lr = base_lr * (0.1 ** (epoch // 100))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % batch_size == 0:
                print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch,loss.item()))
                Loss_list.append(loss.item())

    correct2 = 0
    if(epoch%20==0):
        for data,target in train_loader:
                if use_gpu:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data, volatile=True), Variable(target)
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1]
                correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()
        print("train_set Accuracy: {}/{} ({:.2f}%)\n".format(correct2,len(train_loader.dataset),100. * float(correct2) / float(len(train_loader.dataset))))



def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
       test_loss, correct, len(test_loader.dataset),
       100. * float(correct) / float(len(test_loader.dataset))))
    #scheduler.step(test_loss)

    Accuracy_list.append(100. * float(correct) / float(len(test_loader.dataset)))





#print(model)
optimizer = torch.optim.SGD(model.parameters(),lr=base_lr)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
criterion = nn.CrossEntropyLoss(size_average=False)


for epoch in range(1, epoch_num+1):
    # adjust_learning_rate(optimizer,epoch)
    print('----------------start train---------------')
    train(epoch)
    print('----------------end train-----------------')

    print('----------------start test----------------')
    test()

    print('----------------end test----------------')
    # if(epoch%30==0):
    #     base_lr=base_lr*0.5



    print("lr=%s" % base_lr)
    if(epoch%20==0):
        plt.figure(figsize=(30, 30), dpi=200)
        x1=range(0,epoch)
        y1=Loss_list
        y2=Accuracy_list
        plt.subplot(211)
        plt.plot(x1, y1, label='Frist line', linewidth=1, color='r', marker='o',
                 markerfacecolor='blue', markersize=5)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.subplot(212)
        plt.plot(x1, y2, label='Frist line', linewidth=1, color='r', marker='o',
                 markerfacecolor='blue', markersize=5)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.savefig("/home/liyouwei/gaze/train_3x3/new_example/loss_accu.jpg")
        plt.close()

    if(epoch%200==0):
        torch.save(model.state_dict(),"/home/liyouwei/gaze/train_3x3/new_example/model/%s_30x50x500_face_SGD_dropout.pkl"%epoch)
    if(Accuracy_list[len(Accuracy_list)-1]>91.0):
        torch.save(model.state_dict(),
                   "/home/liyouwei/gaze/train_3x3/new_example/model/%s_30x50x500_face_SGD_dropout.pkl" %Accuracy_list[len(Accuracy_list)-1])





