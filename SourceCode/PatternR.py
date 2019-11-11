import spectral.io.envi as envi
import spectral
import datetime
#import ColorMap as cm
import cv2
import numpy as np


# #filename,filepath,k,Iteration,method
# def myKmeansMain(filename,k,I,viewIter):
#     # 1 readFile
#     begin = datetime.datetime.now()
#     data = envi.open(filename, filename.split('.')[0])#这里指的是用.来分割，取第一部分
#     # use [M,N,B], M(rows), N(columns), and B(bands)
#     #Notice! use for M in range(data.nrows) to traverse the element.
#     img=data.load() #SpectralArray is quicker than SpyFile
#     # 2 classify
#     m,c =myKmeans(img,k,I,viewIter)
#     #m,c = spectral.kmeans(img,6,20)#m:结果图 c:聚类中心 可以访问m.shape 和 c.shape 这俩是nd.array 在这里可以声明一下RGB分量
#     end = datetime.datetime.now()#record end time
#     # 3 ColorMapping
#     resultImg = ColorMap(m,c)#利用CorlorMap对分类结果上色
#     # 4 show ResultImage

#     cv2.namedWindow("finalResult", cv2.WINDOW_NORMAL)
#     cv2.imshow("finalResult",resultImg)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     # 5 show time used
#     print('time:%s'%(end-begin))
#     return resultImg


def ColorMap(m,c):
    mshape=m.shape 
    cshape=c.shape 
    ColorStyle=[247,85,151,
                82,183,125,
                144,157,243,
                212,212,210,
                0,144,255,
                125,0,255,
                25,135,0,
                127,102,102,
                42,87,63,
                0,201,87,
                46,139,87,
                160,32,240,
                255,127,80,
                255,172,203]
    ColorStyle=np.uint8(ColorStyle)
    ColorStyle=ColorStyle.reshape(-1,3)
    #flatten是压缩成了全新的一维数组，ndarray本身是不能做索引的
    resultImg=ColorStyle[m.flatten()]
    #这里的uint就不要8了，因为可能会超出。导致截断
    resultImg=resultImg.reshape(np.int64(mshape[0]),np.int64(mshape[1]),3)
    return resultImg

# def myKmeans(img,k,I,viewIter):#图像 类别数 迭代数
#     shape = img.shape
#     img = img.reshape(-1,shape[2])
#     m = np.zeros((shape[0]*shape[1],1))#存放分类结果
#     c = np.zeros((k,shape[2]))#存放聚类中心
#     pad = np.zeros(img.shape)#存放img和向量相减的内容
#     distance = np.zeros((shape[0]*shape[1],k))#点到各聚类中心的距离
#     #1. generate cluster centers randomly
#     for i in range(k):#注意在此示例中，k=3，I=4，img的深度是6
#         c[i] = img[i,:]#初次聚类中心是我随意定的
#     for iter in range(I):
#     #2.求聚类结果和新的聚类中心
#         c_sum = np.zeros((k,shape[2]))#用于计算新的聚类中心：保存累加值
#         c_num = np.zeros((k,1))#用于计算新的聚类中心：保存数量
#         cnt=0
#         for i in range(k):
#             pad = np.tile(c[i],(shape[0]*shape[1],1))
#             pad = img-pad
#             distance[:,i] = np.linalg.norm(pad,axis=1)#调用numpy中
#     #产生个新的矩阵，矩阵是img和center之间的差值
#     #对这k个矩阵每行求np.linalg.norm，结果存储在distance的第k列中 循环k次
#         tmin = np.min(distance,axis=1)
#     #跳出循环
#     #对distance矩阵求每行的最小值，记录最小值对应的列为t，m矩阵对应位置记录t+1；
#         for i in range(shape[0]*shape[1]):
#             t = np.where(distance[i] == tmin[i])
#             if(m[i]!=t[0][0]):
#                 cnt = cnt+1    #同时可以记录一下有多少个点发生了变化
#             m[i] = t[0][0]
#             c_sum[t] = c_sum[t]+img[i] #把第t类的位置上加上第i个元素的值
#             c_num[t] = c_num[t]+1      
            
#         m = m.flatten()
#         m = np.uint8(m)
#         if(viewIter):
#             tm=m.reshape((shape[0],shape[1]))
#             resultImg = ColorMap(tm,c)#利用CorlorMap对分类结果上色
#             # 4 show ResultImage
#             cv2.namedWindow("DuringIteration", cv2.WINDOW_NORMAL)
#             cv2.imshow("DuringIteration",resultImg)
#             cv2.waitKey(50)
#     #3.是否达到最大迭代数？
#     #如果达到了就return 没达到就求出新的c 循环到1的下方
#     #改变c为新的聚类中心
#         t=c_sum/c_num
#         if((t==c).all()):
#             m=m.reshape((shape[0],shape[1]))
#             print('iteration:%d end'%iter)
#             return m,c
#         c=t

#     #4.输出时 reshape成规定格式
#     m=m.reshape((shape[0],shape[1]))
#     print('iteration:%d end'%iter)
#     cv2.destroyAllWindows()
#     return m,c

#def MyISODATA():