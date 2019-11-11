# -*- coding: utf-8 -*-

#  注意！ 我找到了求最小值下标的函数np.argmin(xxx,axis=0/1[每列/每行])
#UI
import os  
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QRect, QSize, QMetaObject, QCoreApplication, QPropertyAnimation  
from PyQt5.QtGui import QFont,QIcon
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QGridLayout, QPushButton,QApplication, QMainWindow, QFileDialog,QInputDialog,QMessageBox
#Algorithms
import PatternR as PR
import time
import random
import spectral.io.envi as envi
import spectral
import datetime
#import ColorMap as cm
import cv2
import numpy as np

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(527, 738)
        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setGeometry(QtCore.QRect(20, 40, 491, 421))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.MinClassDistLabel = QtWidgets.QLabel(self.tab)
        self.MinClassDistLabel.setGeometry(QtCore.QRect(60, 220, 181, 16))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(14)
        self.MinClassDistLabel.setFont(font)
        self.MinClassDistLabel.setObjectName("MinClassDistLabel")
        self.MaxStdvLabel = QtWidgets.QLabel(self.tab)
        self.MaxStdvLabel.setGeometry(QtCore.QRect(60, 170, 201, 16))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(14)
        self.MaxStdvLabel.setFont(font)
        self.MaxStdvLabel.setObjectName("MaxStdvLabel")
        self.MaxIterLabel = QtWidgets.QLabel(self.tab)
        self.MaxIterLabel.setGeometry(QtCore.QRect(60, 100, 161, 16))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(14)
        self.MaxIterLabel.setFont(font)
        self.MaxIterLabel.setObjectName("MaxIterLabel")
        self.MaxStdvdoubleSpinBox = QtWidgets.QDoubleSpinBox(self.tab)
        self.MaxStdvdoubleSpinBox.setGeometry(QtCore.QRect(330, 160, 81, 31))
        self.MaxStdvdoubleSpinBox.setMinimum(0.1)
        self.MaxStdvdoubleSpinBox.setMaximum(10000)
        self.MaxStdvdoubleSpinBox.setSingleStep(0.1)
        self.MaxStdvdoubleSpinBox.setProperty("value", 10.0)
        self.MaxStdvdoubleSpinBox.setObjectName("MaxStdvdoubleSpinBox")
        self.line = QtWidgets.QFrame(self.tab)
        self.line.setGeometry(QtCore.QRect(20, 140, 421, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.MaxIterBox = QtWidgets.QSpinBox(self.tab)
        self.MaxIterBox.setGeometry(QtCore.QRect(330, 90, 81, 41))
        self.MaxIterBox.setProperty("value", 15)
        self.MaxIterBox.setObjectName("MaxIterBox")
        self.classNumLabel = QtWidgets.QLabel(self.tab)
        self.classNumLabel.setGeometry(QtCore.QRect(60, 50, 121, 16))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(14)
        self.classNumLabel.setFont(font)
        self.classNumLabel.setObjectName("classNumLabel")
        self.classNumBox = QtWidgets.QSpinBox(self.tab)
        self.classNumBox.setGeometry(QtCore.QRect(330, 40, 81, 41))
        self.classNumBox.setProperty("value", 4)
        self.classNumBox.setObjectName("classNumBox")
        self.MinDistancedoubleSpinBox = QtWidgets.QDoubleSpinBox(self.tab)
        self.MinDistancedoubleSpinBox.setGeometry(QtCore.QRect(330, 210, 81, 31))
        self.MinDistancedoubleSpinBox.setSingleStep(0.1)
        self.MinDistancedoubleSpinBox.setMaximum(10000)
        self.MinDistancedoubleSpinBox.setProperty("value", 50.0)
        self.MinDistancedoubleSpinBox.setObjectName("MinDistancedoubleSpinBox")
        self.MinDistancedoubleSpinBox_2 = QtWidgets.QDoubleSpinBox(self.tab)
        self.MinDistancedoubleSpinBox_2.setGeometry(QtCore.QRect(330, 310, 81, 31))
        self.MinDistancedoubleSpinBox_2.setMaximum(10.0)
        self.MinDistancedoubleSpinBox_2.setSingleStep(0.1)
        self.MinDistancedoubleSpinBox_2.setProperty("value", 0.6)
        self.MinDistancedoubleSpinBox_2.setObjectName("MinDistancedoubleSpinBox_2")
        self.RunK_2 = QtWidgets.QPushButton(self.tab)
        self.RunK_2.setGeometry(QtCore.QRect(330, 360, 93, 28))
        self.RunK_2.setObjectName("RunK_2")
        self.radioButton = QtWidgets.QRadioButton(self.tab)
        self.radioButton.setGeometry(QtCore.QRect(40, 360, 201, 19))
        self.radioButton.setChecked(True)
        self.radioButton.setObjectName("radioButton")
        self.minSamples = QtWidgets.QLabel(self.tab)
        self.minSamples.setGeometry(QtCore.QRect(60, 270, 181, 16))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(14)
        self.minSamples.setFont(font)
        self.minSamples.setObjectName("minSamples")
        self.minSamplesBox = QtWidgets.QSpinBox(self.tab)
        self.minSamplesBox.setGeometry(QtCore.QRect(330, 260, 81, 31))
        self.minSamplesBox.setMinimum(1)
        self.minSamplesBox.setMaximum(1000)
        self.minSamplesBox.setSingleStep(2)
        self.minSamplesBox.setProperty("value", 50)
        self.minSamplesBox.setObjectName("minSamplesBox")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.kMaxIterBox = QtWidgets.QSpinBox(self.tab_2)
        self.kMaxIterBox.setGeometry(QtCore.QRect(330, 90, 81, 41))
        self.kMaxIterBox.setProperty("value", 10)
        self.kMaxIterBox.setObjectName("kMaxIterBox")
        self.kclassNumBox = QtWidgets.QSpinBox(self.tab_2)
        self.kclassNumBox.setGeometry(QtCore.QRect(330, 40, 81, 41))
        self.kclassNumBox.setProperty("value", 4)
        self.kclassNumBox.setObjectName("kclassNumBox")
        self.kMaxIterLabel_ = QtWidgets.QLabel(self.tab_2)
        self.kMaxIterLabel_.setGeometry(QtCore.QRect(60, 100, 161, 16))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(14)
        self.kMaxIterLabel_.setFont(font)
        self.kMaxIterLabel_.setObjectName("kMaxIterLabel_")
        self.kclassNumLabel = QtWidgets.QLabel(self.tab_2)
        self.kclassNumLabel.setGeometry(QtCore.QRect(60, 50, 121, 16))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(14)
        self.kclassNumLabel.setFont(font)
        self.kclassNumLabel.setObjectName("kclassNumLabel")
        self.RunK = QtWidgets.QPushButton(self.tab_2)
        self.RunK.setGeometry(QtCore.QRect(330, 360, 93, 28))
        self.RunK.setObjectName("RunK")
        self.kradioButton = QtWidgets.QRadioButton(self.tab_2)
        self.kradioButton.setGeometry(QtCore.QRect(40, 360, 201, 19))
        self.kradioButton.setChecked(True)
        self.kradioButton.setObjectName("kradioButton")
        self.tabWidget.addTab(self.tab_2, "")
        self.openButton = QtWidgets.QPushButton(Dialog)
        self.openButton.setGeometry(QtCore.QRect(0, 0, 131, 28))
        self.openButton.setObjectName("openButton")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(130, 0, 121, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setGeometry(QtCore.QRect(20, 460, 491, 271))
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser.append('本程序操作说明:\n')
        self.textBrowser.append('1.通过左上角OpenImg按钮选择ENVI标准格式的图像。\n  若不选择，则采用默认图像\n  老师提供的图像被我命名为teacherSample以供选择\n2. 选项卡可以在ISODATA\K-MEANS算法间切换\n3. 可以在两个算法的面板中选择所需参数4. 上方view each iteration选中时会显示每次迭代的结果图像\n5. 选择好后点击Run运行程序\n6. 本程序经过向量化优化，运算速度有大幅上升')
        self.MinClassDistLabel_2 = QtWidgets.QLabel(self.tab)
        self.MinClassDistLabel_2.setGeometry(QtCore.QRect(60, 320, 181, 16))
        self.setWindowIcon(QIcon('nanamiIcon.ico'))

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "EZ_Classifier_StudyVer."))
        self.MinClassDistLabel.setText(_translate("Dialog", "Minimum Class Distance:"))
        self.MaxStdvLabel.setText(_translate("Dialog", "Maximum Class Stdv:"))
        self.MaxIterLabel.setText(_translate("Dialog", "Maximum Iteration:"))
        self.classNumLabel.setText(_translate("Dialog", "Class Number:"))
        self.RunK_2.setText(_translate("Dialog", "Run"))
        self.radioButton.setText(_translate("Dialog", "view each iteration"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Dialog", "ISODATA"))
        self.kMaxIterLabel_.setText(_translate("Dialog", "Maximum Iteration:"))
        self.kclassNumLabel.setText(_translate("Dialog", "Class Number:"))
        self.RunK.setText(_translate("Dialog", "Run"))
        self.kradioButton.setText(_translate("Dialog", "view each iteration"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Dialog", "K-MEANS"))
        self.openButton.setText(_translate("Dialog", "OpenImg"))
        self.pushButton_2.setText(_translate("Dialog", "Producer"))
        self.minSamples.setText(_translate("Dialog", "Minimum Samples:"))
        self.MinClassDistLabel_2.setText(_translate("Dialog", "Step (M)"))

class Main(QWidget,Ui_Dialog):

    #先列举一下参数
    k=4
    I=10
    maxStdv=1
    minDis=5
    filename="sample.HDR"
    viewIter=True
    minS=4
    #下面是两个展示过程的参数
    cnt_t=0
    iter_t=0
    M=0.8

    #下面是一个构造函数
    def __init__(self):
        QWidget.__init__(self)
        self.setupUi(self)
        #coder
        self.pushButton_2.clicked.connect(self.showInfo)
        #openfile
        self.openButton.clicked.connect(self.openFile)#在构造中就把消息槽都弄好
        #update ISODATA
        self.classNumBox.valueChanged.connect(self.ISODATA_updateData)
        self.MaxIterBox.valueChanged.connect(self.ISODATA_updateData)
        self.MaxStdvdoubleSpinBox.valueChanged.connect(self.ISODATA_updateData)
        self.MinDistancedoubleSpinBox.valueChanged.connect(self.ISODATA_updateData)
        self.radioButton.toggled.connect(self.ISODATA_viewEachIteration)
        self.minSamplesBox.valueChanged.connect(self.ISODATA_updateData)
        #update KMEANS
        self.kclassNumBox.valueChanged.connect(self.KMEANS_updateData)
        self.kMaxIterBox.valueChanged.connect(self.KMEANS_updateData)
        self.kradioButton.toggled.connect(self.KMEANS_viewEachIteration)
        #kmeans RUN
        self.RunK.clicked.connect(self.RunKmeans)
        #ISODATA RUN
        self.RunK_2.clicked.connect(self.RunISODATA)
    def showInfo(self):
        QMessageBox.information(self,"Information","制作者：麻尧崴 \nProduced by YAOWEI MA")

    #下面列举各种要用到的函数
    def openFile(self):
        self.filename, __ = QFileDialog.getOpenFileName(self,"Select Img","./","ENVI HDR(*.HDR)")  
        if(self.filename==''):
            self.filename="sample.HDR"
    #更新数据的函数
    def ISODATA_updateData(self):
        self.k=self.classNumBox.value()
        self.I=self.MaxIterBox.value()
        self.maxStdv=self.MaxStdvdoubleSpinBox.value()
        self.minDis=self.MinDistancedoubleSpinBox.value()
        self.minS=self.minSamplesBox.value()
        self.M=self.MinDistancedoubleSpinBox_2.value()
    def ISODATA_viewEachIteration(self):
        self.viewIter=self.radioButton.isChecked()

    def KMEANS_updateData(self):
        self.k=self.kclassNumBox.value()
        self.I=self.kMaxIterBox.value()
    def KMEANS_viewEachIteration(self):
        self.viewIter=self.kradioButton.isChecked()

    #______________________________K均值-核心算法程序部分___________________________________
    #运行k均值函数
    def RunKmeans(self):
        # 1 readFile
        begin = datetime.datetime.now()
        data = envi.open(self.filename, self.filename.split('.')[0])#这里指的是用.来分割，取第一部分
        # use [M,N,B], M(rows), N(columns), and B(bands)
        #Notice! use for M in range(data.nrows) to traverse the element.
        img=data.load() #SpectralArray is quicker than SpyFile
        # 2 classify
        m,c =self.myKmeans(img,self.k,self.I,self.viewIter)
        #m,c = spectral.kmeans(img,5,6)#m:结果图 c:聚类中心 可以访问m.shape 和 c.shape 这俩是nd.array 在这里可以声明一下RGB分量
        end = datetime.datetime.now()#record end time
        # 3 ColorMapping
        resultImg = PR.ColorMap(m,c)#利用CorlorMap对分类结果上色

        # 4 show time used
        self.textBrowser.append('Time used : %s'%(end-begin))

        # 5 show ResultImage
        cv2.namedWindow("finalResult", cv2.WINDOW_NORMAL)
        cv2.imshow("finalResult",resultImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #把k均值核心代码放在这里了
    def myKmeans(self,img,k,I,viewIter):#图像 类别数 迭代数
        shape = img.shape
        img = img.reshape(-1,shape[2])
        m = np.zeros((shape[0]*shape[1],1))#存放分类结果
        c = np.zeros((k,shape[2]))#存放聚类中心
        pad = np.zeros(img.shape)#存放img和向量相减的内容
        distance = np.zeros((shape[0]*shape[1],k))#点到各聚类中心的距离
        lasttime_cnum=np.zeros((k,1))
        diff_num=0
        #1. generate cluster centers randomly
        for i in range(k):#注意在此示例中，k=3，I=4，img的深度是6
            c[i] = img[i,:]#初次聚类中心是我随意定的
        for iter in range(I):
        #2.求聚类结果和新的聚类中心
            c_sum = np.zeros((k,shape[2]))#用于计算新的聚类中心：保存累加值
            c_num = np.zeros((k,1))#用于计算新的聚类中心：保存数量
            cnt=0
            for i in range(k):
                pad = np.tile(c[i],(shape[0]*shape[1],1))
                pad = img-pad
                distance[:,i] = np.linalg.norm(pad,axis=1)#调用numpy中
        #产生个新的矩阵，矩阵是img和center之间的差值
        #对distance矩阵求每行的最小值，记录最小值对应的列为t，m矩阵对应位置记录t+1；
            m=np.argmin(distance,axis=1)#如果有多个一样的值，这个argmin也只会返回第一个
            for i in range(k):
                t_result=m-i
                t_value=img[np.argwhere(t_result==0)]#这里原来写错了，应该是取出值为0的元素的下标
                c_num[i]=t_value.shape[0]
                c_sum[i]=t_value.sum(axis=0)
            
            """
            矢量化：
                利用argmin求出distance每行最小值第一次出现的坐标（直接就是m了）并不需要单独求最小值
                用m-i使得第i类的位置为0，再用argwhere(xxx==0)得到为0的位置的下标，即第i类在img中的所有下标
                直接把上一步得到的矢量作为下标来访问img（即img[argwhere(xxx==0)]） 得到这一类的所有点
                这些点沿着纵向求和（axis=0）再除以点数，得到新的聚类中心
            """
            #下方代码已经被argmin淘汰——2018.6.1
            # for i in range(shape[0]*shape[1]):
            #     t = np.where(distance[i] == tmin[i])
            #     if(m[i]!=t[0][0]):
            #         cnt = cnt+1    #同时可以记录一下有多少个点发生了变化
            #     m[i] = t[0][0]
            #     c_sum[t] = c_sum[t]+img[i] #把第t类的位置上加上第i个元素的值
            #     c_num[t] = c_num[t]+1 
            t_cnum=(lasttime_cnum-c_num)
            t_cnum=t_cnum.flatten()
            t_cnum2=t_cnum[np.argwhere(t_cnum>0)]
            diff_num=t_cnum2.sum()
            if(iter==0):
                diff_num=c_num.sum()
            lasttime_cnum=c_num
            t=c_sum/c_num
            p_c=c
            c=t
            m = m.flatten()
            m = np.uint8(m)
            self.cnt_t=cnt
            self.iter_t=iter
            self.textBrowser.append('iter: %d, reassign %d points, has %d Classes'%((self.iter_t,diff_num,self.k)))
            if(viewIter):
                tm=m.reshape((shape[0],shape[1]))
                resultImg = PR.ColorMap(tm,c)#利用CorlorMap对分类结果上色
                # 4 show ResultImage
                cv2.namedWindow("DuringIteration", cv2.WINDOW_NORMAL)
                cv2.imshow("DuringIteration",resultImg)
                cv2.waitKey(30)
        #3.是否收敛了？
            
            if((t==p_c).all()):
                m=m.reshape((shape[0],shape[1]))
                self.textBrowser.append('iteration:%d END'%iter)
                return m,c
            

        #4.到达迭代上限
        m=m.reshape((shape[0],shape[1]))
        self.textBrowser.append('iteration:%d END'%iter)
        cv2.destroyAllWindows()
        return m,c
    #______________________________ISODATA-核心算法程序部分___________________________________
    #运行ISODATA的函数
    def RunISODATA(self):
        # 1 readFile
        begin = datetime.datetime.now()
        data = envi.open(self.filename, self.filename.split('.')[0])#这里指的是用.来分割，取第一部分
        # use [M,N,B], M(rows), N(columns), and B(bands)
        #Notice! use for M in range(data.nrows) to traverse the element.
        img=data.load() #SpectralArray is quicker than SpyFile
        # 2 classify
        m,c =self.myISODATA(img,self.k,self.I,self.viewIter,self.maxStdv,self.minDis,self.minS)
        #m,c = spectral.kmeans(img,6,20)#m:结果图 c:聚类中心 可以访问m.shape 和 c.shape 这俩是nd.array 在这里可以声明一下RGB分量
        end = datetime.datetime.now()#record end time
        # 3 ColorMapping
        resultImg = PR.ColorMap(m,c)#利用CorlorMap对分类结果上色

        # 4 show time used
        self.textBrowser.append('Time used : %s'%(end-begin))

        # 5 show ResultImage
        cv2.namedWindow("finalResult", cv2.WINDOW_NORMAL)
        cv2.imshow("finalResult",resultImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    #ISODATA 核心代码部分
    def myISODATA(self,img,k,I,viewIter,maxStdv,minDis,minS):
        shape = img.shape
        img = img.reshape(-1,shape[2])
        m = np.zeros((shape[0]*shape[1],1))
        pad = np.zeros(img.shape)
        c = np.zeros((k,shape[2]))#在迭代中，k的大小会变

        for i in range(k):
            c[i] = img[i,:]   
        for iter in range(I):
    #1  初步骤求解聚类结果 
            #下方这些代码是假设k不动的时候写的
            distance = np.zeros((shape[0]*shape[1],k))

            c_sum = np.zeros((k,shape[2]))
            c_num = np.zeros((k,1))
            cnt=0
            for i in range(k):
                pad = np.tile(c[i],(shape[0]*shape[1],1))
                pad = img-pad
                distance[:,i] = np.linalg.norm(pad,axis=1)
    #2 检查是否有类别 样本数少于 minSamples
            m=np.argmin(distance,axis=1)#如果有多个一样的值，这个argmin也只会返回第一个
            for i in range(k):
                t_result=m-i
                t_value=img[np.argwhere(t_result==0)]#这里原来写错了，应该是取出值为0的元素的下标
                c_num[i]=t_value.shape[0]
                c_sum[i]=t_value.sum(axis=0)
            
            if (c_num<minS).any():
                deleteDic=np.zeros(k)
                t=k
                for i in range(t):
                    if c_num[i]<minS:
                        k=k-1
                        deleteDic[i]=1
                t_distance=np.zeros((shape[0]*shape[1],k))
                t_c=np.zeros((k,shape[2]))
                t_i=0
                for i in range(t):
                    #丢弃对应类别的distance
                    if(deleteDic[i]==0):
                        t_distance[:,t_i]=distance[:,i]
                        t_c[t_i,:]=c[i]
                        t_i=t_i+1
                distance=t_distance
                c=t_c
                #3.2.1 重新求所属
                c_sum = np.zeros((k,shape[2]))
                c_num = np.zeros((k,1))
                cnt=0
                m=np.argmin(distance,axis=1)#如果有多个一样的值，这个argmin也只会返回第一个
                for i in range(k):
                    t_result=m-i
                    t_value=img[np.argwhere(t_result==0)]#这里原来写错了，应该是取出值为0的元素的下标
                    c_num[i]=t_value.shape[0]
                    c_sum[i]=t_value.sum(axis=0)
                #样本数量过少所导致的类别融合到此结束

    #3 若在奇数次或者k<输入值一半的迭代：检查是否有类内方差超过maxStdv的类别 需要进行分裂
            if (((iter % 2) == 0) or k<(self.k/2)) : #奇数次迭代的标号是偶数 self.k是用户输入的值
                b_split=False
                t_maxStd=-1
                for i in range(k):
                    t_result=m-i
                    t_value=img[np.argwhere(t_result==0)]
                    #t_value存储了该类的所有信息 可以用来求该类在各个特征方向上的std
                    std=np.std(t_value,axis=0)
                    if(std>maxStdv).any():
                        #找到最大的方向和最大的类别
                        t_n_feature=np.argmax(std)#必须是stdv最大的方向
                        if(std[0,t_n_feature]>t_maxStd):#必须是stdv最大的类别
                            t_maxStd=std[0,t_n_feature]
                            n_feature=t_n_feature.copy()
                            n_class=i
                            b_split=True
                #已经求出了第n_class类，第n_feature特征是需要被切断的对象
                #######每次改变的百分比M 没有被设置为参数
                if b_split:
                    
                    split_t_result=m-n_class
                    split_t_value=img[np.argwhere(split_t_result==0)]
                    std=np.std(split_t_value,axis=0)

                    k=k+1
                    M=self.M
                    t_row1=c[n_class,:]
                    t_row2=t_row1.copy()#不能等于，要是等于的话就会地址传递
                    t_row1[n_feature]=t_row1[n_feature]-M*std[0,n_feature]
                    t_row2[n_feature]=t_row2[n_feature]+M*std[0,n_feature]
                    ##组成新的聚类中心
                    c[n_class,:]=t_row1.T
                    #新的操作，np.r_[]第一个参数'0,2,1'代表在第一个元素后，所有的对象提升到至少二维（本来row是一维）
                    #提升维度的方法是横向（(4,)变成（1,4））
                    c=np.r_['0,2,1',c,t_row2.T]
                    distance = np.zeros((shape[0]*shape[1],k))
                    c_sum = np.zeros((k,shape[2]))
                    c_num = np.zeros((k,1))
                    cnt=0
                    for i in range(k):
                        pad = np.tile(c[i],(shape[0]*shape[1],1))
                        pad = img-pad
                        distance[:,i] = np.linalg.norm(pad,axis=1)

                    m=np.argmin(distance,axis=1)#如果有多个一样的值，这个argmin也只会返回第一个
                    for i in range(k):
                        t_result=m-i
                        t_value=img[np.argwhere(t_result==0)]#这里原来写错了，应该是取出值为0的元素的下标
                        c_num[i]=t_value.shape[0]
                        c_sum[i]=t_value.sum(axis=0)
    #4 若在偶数次或者k>2倍输入值的迭代：检查是否有类间距离太小的两个类别 需要进行合并
            if (((iter % 2) == 1) or k>(self.k*2)) : 
                b_merge=False
                #以后有空可以改为马氏距离
                #求聚类中心之间的欧氏距离
                EucildDistence=np.zeros((k,k))
                for classi in range(k):
                    t_class=np.tile(c[classi,:],(k,1))
                    t_minus=c-t_class
                    EucildDistence[:,classi]=np.linalg.norm(t_minus,axis=1)
                t_height=k*(k-1)/2
                t_height=np.uint32(t_height)
                distStruct=np.zeros((t_height,5))#分别是：distence、i、j、flag_merge、flag_invalid
                cursor=0
                for classi in range(1,k):#下三角矩阵
                    for classj in range(0,classi):
                        distStruct[cursor,:]=[EucildDistence[classi,classj],classi,classj,0,0]
                        cursor=cursor+1
                distStruct=distStruct[np.lexsort(distStruct[:,::-1].T)]
                for i in range(t_height):
                    if(distStruct[i,4]==0 and distStruct[i,0]<minDis):
                        b_merge=True
                        distStruct[i,3]=1
                        for j in range(t_height):
                            if(distStruct[j,1]==distStruct[i,1] or distStruct[j,2]==distStruct[i,1] or distStruct[j,1]==distStruct[i,2] or distStruct[j,2]==distStruct[i,2]):
                                distStruct[j,4]=1
                # 已经标记完需要合并哪些类别
                # 对每个需要合并的对象（distStruct[X,3]==1的1、2 即为类别） 计算二者对应的c的平均值，并且让k=k-1
                # 搞一个c_t=c.copy() 将均值放在前一个类别里，后一个类别置0,
                # 全部处理完了，把t_c中非零行放到c里面（c是新产生的）
                t_c=c.copy()   
                marker=False #记录下是不是进入了下方的if         
                for i in range(t_height):
                    if(distStruct[i,3]==1):
                        class_a=distStruct[i,1]
                        class_b=distStruct[i,2]
                        class_a=np.uint32(class_a)
                        class_b=np.uint32(class_b)
                        k=k-1
                        #
                        t_c[class_a,:]=(t_c[class_a,:]+t_c[class_b,:])/2
                        t_c[class_b,:]=np.zeros((1,shape[2]))
                        marker=True
                if(marker):
                    c=t_c[np.nonzero(t_c)]
                    c=c.reshape(k,shape[2])
                    distance = np.zeros((shape[0]*shape[1],k))
                    c_sum = np.zeros((k,shape[2]))
                    c_num = np.zeros((k,1))
                    cnt=0
                    for i in range(k):
                        pad = np.tile(c[i],(shape[0]*shape[1],1))
                        pad = img-pad
                        distance[:,i] = np.linalg.norm(pad,axis=1)

                    m=np.argmin(distance,axis=1)#如果有多个一样的值，这个argmin也只会返回第一个
                    for i in range(k):
                        t_result=m-i
                        t_value=img[np.argwhere(t_result==0)]#这里原来写错了，应该是取出值为0的元素的下标
                        c_num[i]=t_value.shape[0]
                        c_sum[i]=t_value.sum(axis=0)

            m = m.flatten()
            m = np.uint8(m)
            self.cnt_t=cnt
            self.iter_t=iter
            self.textBrowser.append('iter: %d, has %d Classes'%((self.iter_t,k)))
            if(viewIter):
                tm=m.reshape((shape[0],shape[1]))
                resultImg = PR.ColorMap(tm,c)
                cv2.namedWindow("DuringIteration", cv2.WINDOW_NORMAL)
                cv2.imshow("DuringIteration",resultImg)
                cv2.waitKey(30)
        #3.是否收敛？
            if(c_num==0).any():
                zero_index=np.argwhere(c_sum==0)
                c_sum[zero_index]=0.01
                c_num[zero_index]=1
            t=c_sum/c_num
            if((t==c).all()):
                m=m.reshape((shape[0],shape[1]))
                self.textBrowser.append('iteration:%d END'%iter)
                return m,c
            c=t

        #4.迭代结束
        m=m.reshape((shape[0],shape[1]))
        self.textBrowser.append('iteration:%d END'%iter)
        cv2.destroyAllWindows()
        return m,c

    #更新下方输出框
        
#主循环所在
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window=Main()
    window.show()
    sys.exit(app.exec_())

