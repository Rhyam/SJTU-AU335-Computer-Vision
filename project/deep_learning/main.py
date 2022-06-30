import cv2
import numpy as np
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from tensorflow import keras
from perspectiveTransform import locate_and_correct
from Unet import unet_predict
from CNN import cnn_predict
import sys


class Window:
    def __init__(self, win, ww, wh):
        self.win = win
        self.ww = ww
        self.wh = wh
        self.win.geometry("%dx%d+%d+%d" % (ww, wh, 200, 50))  # 界面启动时的初始位置
        self.win.title("Car License Plate Detection and Recognization")
        self.img_src_path = None

        self.label_src = Label(self.win, text='原图:', font=('微软雅黑', 13)).place(x=0, y=0)
        self.label_lic = Label(self.win, text='车牌区域:', font=('微软雅黑', 13)).place(x=125, y=520)
        self.label_pred = Label(self.win, text='识别结果:', font=('微软雅黑', 13)).place(x=400, y=520)

        self.can_src = Canvas(self.win, width=512, height=512, bg='white', relief='solid', borderwidth=1)  # 原图画布
        self.can_src.place(x=50, y=0)
        self.can_lic = Canvas(self.win, width=245, height=85, bg='white', relief='solid', borderwidth=1)  # 车牌区域1画布
        self.can_lic.place(x=50, y=550)
        self.can_pred = Canvas(self.win, width=245, height=85, bg='white', relief='solid', borderwidth=1)  # 车牌识别1画布
        self.can_pred.place(x=320, y=550)

        self.button1 = Button(self.win, text='选择文件', width=10, height=1, command=self.load_show_img)  # 选择文件按钮
        self.button1.place(x=ww-330, y=wh - 50)
        self.button2 = Button(self.win, text='识别车牌', width=10, height=1, command=self.display)  # 识别车牌按钮
        self.button2.place(x=ww-230, y=wh - 50)
        self.button3 = Button(self.win, text='清空所有', width=10, height=1, command=self.clear)  # 清空所有按钮
        self.button3.place(x=ww-130, y=wh - 50)
        self.unet = keras.models.load_model('unet.h5')
        self.unet_green = keras.models.load_model('unet_green.h5')
        self.unet_green2 = keras.models.load_model('unet_green2.h5')
        self.cnn = keras.models.load_model('cnn.h5')
        print('lauching...')
        cnn_predict(self.cnn, [np.zeros((80, 240, 3))])
        print("already prepared!")


    def load_show_img(self):
        self.clear()
        sv = StringVar()
        sv.set(askopenfilename())
        self.img_src_path = Entry(self.win, state='readonly', text=sv).get()  # 获取到所打开的图片
        img_open = Image.open(self.img_src_path)
        if img_open.size[0] * img_open.size[1] > 240 * 80:
            img_open = img_open.resize((512, 512), Image.ANTIALIAS)
        self.img_Tk = ImageTk.PhotoImage(img_open)
        self.can_src.create_image(258, 258, image=self.img_Tk, anchor='center')

    def display(self):
        if self.img_src_path == None:  # 还没选择图片就进行预测
            self.can_pred.create_text(32, 15, text='请选择图片', anchor='nw', font=('黑体', 28))
        else:
            img_src = cv2.imdecode(np.fromfile(self.img_src_path, dtype=np.uint8), -1)  # 从中文路径读取时用
            h, w = img_src.shape[0], img_src.shape[1]
            if h * w <= 240 * 80 and 2 <= w / h <= 5:  # 满足该条件说明可能整个图片就是一张车牌,无需定位,直接识别即可
                lic = cv2.resize(img_src, dsize=(240, 80), interpolation=cv2.INTER_AREA)[:, :, :3]  # 直接resize为(240,80)
                img_src_copy, Lic_img = img_src, [lic]
                Lic_pred = cnn_predict(self.cnn, Lic_img)
            else:  # 否则就需通过unet对img_src原图预测,得到img_mask,实现车牌定位,然后进行识别
                img_src, img_mask = unet_predict(self.unet, self.img_src_path)
                img_src_copy, Lic_img = locate_and_correct(img_src, img_mask)  # 利用core.py中的locate_and_correct函数进行车牌定位和矫正
                #cv2.imwrite('figure/img_tk.jpg', img_src_copy)
                Lic_pred = cnn_predict(self.cnn, Lic_img)  # 利用cnn进行车牌的识别预测,Lic_pred中存的是元祖(车牌图片,识别结果)
                if len(Lic_img) == 0:
                    img_src, img_mask = unet_predict(self.unet_green2, self.img_src_path)
                    img_src_copy, Lic_img = locate_and_correct(img_src, img_mask)
                    if len(Lic_img) == 0:
                        img_src, img_mask = unet_predict(self.unet_green, self.img_src_path)
                        img_src_copy, Lic_img = locate_and_correct(img_src, img_mask)
                    Lic_gray = cv2.cvtColor(Lic_img[0], cv2.COLOR_BGR2GRAY)
                    ret, img_thresh = cv2.threshold(Lic_gray, 100, 255, cv2.THRESH_BINARY_INV)
                    Lic_img_copy = Lic_img[0].copy()
                    Lic_img_copy[:, :, 0][img_thresh==0] = 255
                    Lic_img_copy[:, :, 1][img_thresh==0] = 0
                    Lic_img_copy[:, :, 2][img_thresh==0] = 0
                    Lic_img_copy[:, :, 0][img_thresh==255] = 255
                    Lic_img_copy[:, :, 1][img_thresh==255] = 255
                    Lic_img_copy[:, :, 2][img_thresh==255] = 255
                    Lic_pred = cnn_predict(self.cnn, [Lic_img_copy])

            if Lic_pred:
                img = Image.fromarray(img_src_copy[:, :, ::-1])  # img_src_copy[:, :, ::-1]将BGR转为RGB
                self.img_Tk = ImageTk.PhotoImage(img)
                self.can_src.delete('all')  # 显示前,先清空画板
                self.can_src.create_image(258, 258, image=self.img_Tk,
                                          anchor='center')  # img_src_copy上绘制出了定位的车牌轮廓,将其显示在画板上
                for i, lic_pred in enumerate(Lic_pred):
                    if i == 0:
                        self.lic_Tk1 = ImageTk.PhotoImage(Image.fromarray(lic_pred[0][:, :, ::-1]))
                        self.can_lic.create_image(5, 5, image=self.lic_Tk1, anchor='nw')
                        self.can_pred.create_text(10, 20, text=lic_pred[1], anchor='nw', font=('黑体', 28))
                        #cv2.imwrite('figure/img_lcs.jpg', Lic_img[0])

            else:  # Lic_pred为空说明未能识别
                self.can_pred.create_text(47, 15, text='未能识别', anchor='nw', font=('黑体', 27))

    def clear(self):
        self.can_src.delete('all')
        self.can_lic.delete('all')
        self.can_pred.delete('all')
        self.img_src_path = None

    def closeEvent():  # 关闭前清除session(),防止'NoneType' object is not callable
        keras.backend.clear_session()
        sys.exit()


if __name__ == '__main__':
    win = Tk()
    ww = 600
    wh = 700
    Window(win, ww, wh)
    win.protocol("WM_DELETE_WINDOW", Window.closeEvent)
    win.mainloop()
