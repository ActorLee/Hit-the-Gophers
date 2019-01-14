import cv2
import numpy as np
import screeninfo
import random
import os
import math
import sys
import datetime
import json


screen = screeninfo.get_monitors()[0]
width, height = screen.width, screen.height
count=0
c_x=int(width/2)
c_y=int(height/2)
c_color=0
circle_index=0
txtfile="C:/Users/lyw/dataapp/data_reg.json"
file_path="C:/Users/lyw/dataapp/data/source_data_reg/"

point_num=5
point_index=0
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

                    random_circle_location()
                    circle_index=0
                #random_circle_color()
                draw_circle(c_x,c_y,c_color)
            else:
                begin_flag = 1
                random_circle_location()
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

                    random_circle_location()
                    circle_index=0
                #random_circle_color()
                draw_circle(c_x, c_y, c_color)
            else:
                begin_flag = 1
                random_circle_location()
                draw_circle(c_x, c_y, c_color)
        else:
            sys.exit()

def get_img():
    random_file_name = random.randint(100000, 999999)
    global cap,count,nowTime,f,c_x,c_y,circle_index,circle_pre_point
    #if (circle_index != circle_pre_point - 1):
    while(1):
        if(count<frame_num):
            ret, frame = cap.read()
            save_path = os.path.join(file_path, str(random_path_name))
            check_dir(save_path)
            cv2.imwrite(os.path.join(file_path, str(random_path_name)+"/"+str(random_file_name)+str(count)+ ".jpg"), frame)
            new_dict={"img":str(random_path_name)+"/"+str(random_file_name)+str(count)+".jpg","point":{"c_x":c_x,"c_y":c_y},"circle_pre_point":circle_pre_point,"precision":precision}
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
    cv2.waitKey(50)
    img = np.zeros((height, width, 3),np.uint8)
    if(color==0):
        cv2.circle(img, (x, y), precision, (0, 255, 0), -1)
    elif(color==1):
        cv2.circle(img, (x, y), precision, (0, 0, 255), -1)


    cv2.imshow("demo", img)
    cv2.waitKey(0)

def random_circle_location():
    global c_x,c_y,point_num,point_index
    if(point_index>point_num-1):
        sys.exit()
    else:
        point_index+=1
        y=random.randint(0,height)
        x=random.randint(0,width)
        c_x=x
        c_y=y



def random_circle_color():
    global c_color
    color = random.randint(0, 1)
    c_color = color

nowTime=datetime.datetime.now().strftime('%Y%m%d%H%M%S')
random_path_name=nowTime+str(random.randint(10,99))



f=open(txtfile,"a")
cv2.namedWindow("demo", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("demo",cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("demo", OnMouseAction)
cap = cv2.VideoCapture(0)
draw_circle(c_x,c_y,c_color)








