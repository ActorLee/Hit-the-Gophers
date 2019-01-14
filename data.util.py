import json
import os
import cv2
import shutil
import screeninfo
import random
import _thread
from torch.utils.data import TensorDataset, DataLoader,dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
txtfile="C:/Users/lyw/dataapp/data.json"
label_file="C:/Users/lyw/dataapp/label.txt"
label_file_3x3="C:/Users/lyw/dataapp/label_3x3.txt"
filepath="C:/Users/lyw/dataapp/data/train_3x3/test/"
label_path_test = "C:/Users/lyw/dataapp/data/train_3x3/label_test.txt"
label_path_train = "C:/Users/lyw/dataapp/data/train_3x3/label_train.txt"
filepath_train = "C:/Users/lyw/dataapp/data/train_3x3/train/"
filepath_test = "C:/Users/lyw/dataapp/data/train_3x3/test/"
def data_label_4x4():
    f=open(txtfile,"r")
    f2=open(label_file,"w")
    data=f.readlines()
    for i in data:
        hang = json.loads(i)
        if(hang["img"][23]=="0"):
            shutil.copy("C:/Users/lyw/dataapp/data/"+hang["img"],"C:/Users/lyw/dataapp/data/all/"+hang["img"][17:])
            f2.write(hang["img"][17:])
            f2.write(" ")
            f2.write(str(int(hang["block"]["row"])*4+int(hang["block"]["col"])))
            f2.write("\n")

def data_label():
    label = []
    test=[]
    for i in range(0,9):
        label.append([])

    f=open(txtfile,"r")
    f2=open(label_path_train,"w")
    f3=open(label_path_test,"w")
    data=f.readlines()
    for i in data:
        hang = json.loads(i)
        for j in range(0,9):
            if(int(hang["block"]["row"])*3+int(hang["block"]["col"])==j):
                label[j].append(i)
    for i in range(0,9):
        test+=random.sample(label[i],66)

    print(len(data))
    print(len(test))
    for j in test:
        data.remove(j)
    print(len(data))
    for i in data:
        hang = json.loads(i)
        #shutil.copy("C:/Users/lyw/dataapp/data/source_data_2/"+hang["img"],"C:/Users/lyw/dataapp/data/train_3x3/all_data/"+hang["img"][12:16]+hang["img"][17:])
        f2.write(hang["img"][12:16]+hang["img"][17:])
        f2.write(" ")
        f2.write(str(int(hang["block"]["row"])*3+int(hang["block"]["col"])))
        f2.write("\n")
    for k in test:
        hang = json.loads(k)
        #shutil.copy("C:/Users/lyw/dataapp/data/source_data_2/"+hang["img"],"C:/Users/lyw/dataapp/data/train_3x3/test/"+hang["img"][12:16]+hang["img"][17:])
        f3.write(hang["img"][12:16]+hang["img"][17:])
        f3.write(" ")
        f3.write(str(int(hang["block"]["row"])*3+int(hang["block"]["col"])))
        f3.write("\n")

    f.close()
    f2.close()




def data_resize():
    e=os.listdir(filepath)
    for i in e:
        img_source=cv2.imread(filepath+i)
        img=cv2.resize(img_source,(320,240),0,0,cv2.INTER_LINEAR)
        cv2.imwrite(filepath+i,img)



def label_get_train_test():
    e_train=os.listdir(filepath_train)
    e_test=os.listdir(filepath_test)
    f2 = open(label_file_3x3, "r")
    f_train=open(label_path_train,"w")
    f_test = open(label_path_test, "w")
    data_ori=f2.readlines()
    for i in data_ori:
        for j in e_train:
            if(i[:11]==j):
                f_train.write(i)
        for k in e_test:
            if(i[:11]==k):
                f_test.write(i)


def get_data_point():
    screen = screeninfo.get_monitors()[0]
    width, height = screen.width, screen.height
    img = np.ones((height, width, 3), np.uint8)
    f = open(txtfile, "r")
    data = f.readlines()
    for i in data:
        hang = json.loads(i)
        x=int(hang["point"]["c_x"])
        y=int(hang["point"]["c_y"])
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
    cv2.imshow("1",img)
    cv2.waitKey()



def data_label_3x3():
    f = open(txtfile, "r")
    f2 = open(label_file_3x3, "w")
    data = f.readlines()
    for i in data:
        hang = json.loads(i)
        row=99
        col=99
        if (hang["img"][23] == "0"):
            shutil.copy("C:/Users/lyw/dataapp/data/" + hang["img"], "C:/Users/lyw/dataapp/data/data_3x3/" + hang["img"][17:])
            f2.write(hang["img"][17:])
            f2.write(" ")
            x = float(hang["point"]["c_x"])
            y = float(hang["point"]["c_y"])
            if(x==0):
                col=0
            else:
                if(1366.0/x>=3):
                    col=0
                elif(1.5<=1366.0/x<3):
                    col=1
                elif(1366.0/x<1.5):
                    col=2

            if (y == 0):
                row = 0
            else:
                if (768.0 / y >= 3):
                    row = 0
                elif (1.5 <= 768.0 / y < 3):
                    row = 1
                elif (768.0 / y < 1.5):
                    row = 2

            f2.write(str(row*3+col))
            f2.write("\n")

def draw_data_3x3():
    screen = screeninfo.get_monitors()[0]
    width, height = screen.width, screen.height
    img = np.ones((height, width, 3), np.uint8)
    f = open(txtfile, "r")
    data = f.readlines()
    cv2.line(img,(0,int(768/3)),(1366,int(768/3)),(255,0,0),1)
    cv2.line(img, (0, int(768 / 3*2)), (1366, int(768/ 3*2)), (255, 0, 0), 1)
    cv2.line(img, (int(1366/3), 0), (int(1366/3), 768), (255, 0, 0), 1)
    cv2.line(img, (int(1366 / 3*2), 0), (int(1366 / 3*2), 768), (255, 0, 0), 1)
    f2 = open("C:/Users/lyw/dataapp/data/train_3x3/label_test_new.txt")
    hang2 = f2.readlines()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in data:
        hang = json.loads(i)
        x = int(hang["point"]["c_x"])
        y = int(hang["point"]["c_y"])
        for j in hang2:
            if(hang["img"][12:16]+hang["img"][17:]==j[:15]):
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
                #cv2.putText(img, j[12:13], (x, y), font, 0.5, (255, 255, 255), 1)
    cv2.imshow("1", img)
    cv2.imwrite("C:/Users/lyw/dataapp/data/train_3x3/data_test.jpg",img)
    cv2.waitKey()

draw_data_3x3()

def draw_data_4x4():
    screen = screeninfo.get_monitors()[0]
    width, height = screen.width, screen.height
    img = np.ones((height, width, 3), np.uint8)
    f = open(txtfile, "r")
    data = f.readlines()
    cv2.line(img,(0,int(768/4)),(1366,int(768/4)),(255,0,0),1)
    cv2.line(img, (0, int(768 / 4*2)), (1366, int(768/ 4*2)), (255, 0, 0), 1)
    cv2.line(img, (0, int(768 / 4 * 3)), (1366, int(768 / 4 * 3)), (255, 0, 0), 1)
    cv2.line(img, (int(1366/4), 0), (int(1366/4), 768), (255, 0, 0), 1)
    cv2.line(img, (int(1366 / 4*2), 0), (int(1366 / 4*2), 768), (255, 0, 0), 1)
    cv2.line(img, (int(1366 / 4 * 3), 0), (int(1366 / 4 * 3), 768), (255, 0, 0), 1)
    f2 = open("C:/Users/lyw/dataapp/data/train_2019.1.2/label.txt")
    hang2 = f2.readlines()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in data:
        hang = json.loads(i)
        x = int(hang["point"]["c_x"])
        y = int(hang["point"]["c_y"])
        for j in hang2:
            if(hang["img"][17:]==j[:11]):
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
                cv2.putText(img, j[12:len(j)-1], (x, y), font, 0.5, (255, 255, 255), 1)
    cv2.imshow("1", img)
    cv2.imwrite("C:/Users/lyw/dataapp/data/train_2019.1.2/data.jpg",img)
    cv2.waitKey()


def example_draw():
    row=[0,1,2]
    col=[0,1,2]
    screen = screeninfo.get_monitors()[0]
    width, height = screen.width, screen.height
    img = np.ones((height, width, 3), np.uint8)
    cv2.line(img, (0, int(768 / 3)), (1366, int(768 / 3)), (255, 0, 0), 1)
    cv2.line(img, (0, int(768 / 3 * 2)), (1366, int(768 / 3 * 2)), (255, 0, 0), 1)
    cv2.line(img, (int(1366 / 3), 0), (int(1366 / 3), 768), (255, 0, 0), 1)
    cv2.line(img, (int(1366 / 3 * 2), 0), (int(1366 / 3 * 2), 768), (255, 0, 0), 1)
    sigma=40
    instant=70
    for j in row:
        for k in col:
            for i in range(1000):
                x=0
                y=0
                while (1):
                    x = random.gauss((width / 3 * (k + 1) + width / 3 * k) / 2, sigma)
                    x=int(x)
                    if (width / 3 * k+instant < x < width / 3 * (k+ 1)-instant):
                        break
                while (1):
                    y = random.gauss((height / 3 * j + height / 3 * (j + 1)) / 2, sigma * height / width)
                    y=int(y)
                    if (height / 3 * j+instant < y < height / 3 * (j + 1)-instant):
                        break
                # y=int(random.uniform(height / 3 * j+instant,height / 3 * (j + 1)-instant))
                # x=int(random.uniform(width / 3 * k+instant,width / 3 * (k+ 1)-instant))
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
    cv2.imshow("1",img)
    cv2.waitKey()




