import cv2
import numpy as np
import screeninfo
import random
import os
import math
import sys
import datetime
import json

count=0
c_x=0
c_y=0
c_color=0
circle_index=0
block_row=0
block_col=0
txtfile="C:/Users/lyw/dataapp/data.json"
file_path="C:/Users/lyw/dataapp/data/source_data_2/"
sigma=40
instance=70
#list=[[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]]
list=[[0,0,0],[0,1,0],[0,2,0],[1,0,0],[1,1,0],[1,2,0],[2,0,0],[2,1,0],[2,2,0]]


circle_pre_block=10   #每个block采样点的数目
circle_pre_point=1   #每个采样点的验证次数
frame_num=1            #每次验证所采的帧数
begin_flag=0
precision=8        #生成圆的大小

def OnMouseAction(event, x, y, flags, param):
    global count,c_y,c_y,circle_index,circle_index_pre_block,begin_flag

    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        if(math.sqrt((x1-c_x)**2+(y1-c_y)**2)<precision and c_color==0):
            if(begin_flag==1):
                get_img()
                circle_index = circle_index + 1
                if(circle_index>circle_pre_point-1):

                    row_col()
                    random_circle_location(block_row,block_col)
                    circle_index=0
                #random_circle_color()
                draw_circle(c_x,c_y,c_color)
            else:
                begin_flag = 1
                row_col()
                random_circle_location(block_row,block_col)
                draw_circle(c_x,c_y,c_color)
        # else:
        #     sys.exit()

    elif event == cv2.EVENT_RBUTTONDOWN:
        x1, y1 = x, y
        if (math.sqrt((x1 - c_x) ** 2 + (y1 - c_y) ** 2) < precision and c_color == 1):
            if (begin_flag == 1):
                get_img()
                begin_flag = 1
                circle_index = circle_index + 1
                if (circle_index > circle_pre_point-1):

                    row_col()
                    random_circle_location(block_row,block_col)
                    circle_index=0
                #random_circle_color()
                draw_circle(c_x, c_y, c_color)
            else:
                begin_flag = 1
                row_col()
                random_circle_location(block_row, block_col)
                draw_circle(c_x, c_y, c_color)
        else:
            sys.exit()

def get_img():
    random_file_name = random.randint(100000, 999999)
    global cap,count,nowTime,f,c_x,c_y,block_row,block_col,circle_index,circle_pre_point
    #if (circle_index != circle_pre_point - 1):
    while(1):
        if(count<frame_num):
            ret, frame = cap.read()
            save_path = os.path.join(file_path, str(random_path_name))
            check_dir(save_path)
            cv2.imwrite(os.path.join(file_path, str(random_path_name)+"/"+str(random_file_name)+str(count)+ ".jpg"), frame)
            new_dict={"img":str(random_path_name)+"/"+str(random_file_name)+str(count)+".jpg","point":{"c_x":c_x,"c_y":c_y},
                      "block":{"row":block_row,"col":block_col},"circle_pre_block":circle_pre_block,"circle_pre_point":circle_pre_point,
                      "instance":instance,"precision":precision}
            json.dump(new_dict,f)
            f.write("\n")
            count += 1
        else:
            count=0
            break

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def draw_circle(x,y,color):

    img = np.ones((height, width, 3), np.uint8)
    cv2.imshow("demo", img)
    cv2.waitKey(100)
    img = np.zeros((height, width, 3),np.uint8)
    if(color==0):
        cv2.circle(img, (x, y), precision, (0, 255, 0), -1)
    elif(color==1):
        cv2.circle(img, (x, y), precision, (0, 0, 255), -1)


    cv2.imshow("demo", img)
    cv2.waitKey(0)

def random_circle_location(row,col):
    global c_x,c_y
    while(1):
        x = int(random.gauss((width / 3 * (col + 1) + width / 3 * col) / 2, sigma))
        if(width/3*col+instance<x<width/3*(col+1)-instance):
            break
    while(1):
        y = int(random.gauss((height / 3 * row + height / 3 * (row + 1)) / 2, sigma * height / width))
        if(height/3*row+instance<y<height/3*(row+1)-instance):
            break
    c_x=x
    c_y=y

def row_col():
    global list,block_col,block_row
    if len(list):
        loc = random.sample(list, 1)
        block_row = loc[0][0]
        block_col = loc[0][1]
        loc[0][2] += 1
        for i in list:
            if(i[2]>=circle_pre_block):
                list.remove(i)
    else:
        f.close()
        sys.exit()



def random_circle_color():
    global c_color
    color = random.randint(0, 1)
    c_color = color

nowTime=datetime.datetime.now().strftime('%Y%m%d%H%M%S')
random_path_name=nowTime+str(random.randint(10,99))

screen = screeninfo.get_monitors()[0]
width, height = screen.width, screen.height

f=open(txtfile,"a")
cv2.namedWindow("demo", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("demo",cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("demo", OnMouseAction)
cap = cv2.VideoCapture(0)
random_circle_location(2,2)
draw_circle(c_x,c_y,c_color)








