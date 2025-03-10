#!/usr/bin/env python
# coding: utf-8

# #                                           numpy 练习题

#

# ### numpy 的array操作

# #### 1.导入numpy库
import numpy as np
import matplotlib.pyplot as plt




# #### 2.建立一个一维数组 a 初始化为[4,5,6], (1)输出a 的类型（type）(2)输出a的各维度的大小（shape）(3)输出 a的第一个元素（值为4）

a = np.array([4, 5, 6])
print("第二题:")
print("类型:",type(a))
print("各维度大小:",a.shape)
print("第一个元素:",a[0])




# #### 3.建立一个二维数组 b,初始化为 [ [4, 5, 6],[1, 2, 3]] (1)输出各维度的大小（shape）(2)输出 b(0,0)，b(0,1),b(1,1) 这三个元素（对应值分别为4,5,2）

b = np.array([ [4, 5, 6],
               [1, 2, 3] ])
print("\n第三题:")
print("各维度大小:",b.shape)
print("b(0,0)元素:",b[0,0])
print("b(0,1)元素:",b[0,1])
print("b(1,1)元素:",b[1,1])



# #### 4.  (1)建立一个全0矩阵 a, 大小为 3x3; 类型为整型（提示: dtype = int）(2)建立一个全1矩阵b,大小为4x5;  (3)建立一个单位矩阵c ,大小为4x4; (4)生成一个随机数矩阵d,大小为 3x2.
a = np.zeros((3,3),dtype=int)
b = np.ones((4,5))
c = np.eye(4)
d = np.random.random((3,2))

print("\n第四题:")
print("全0矩阵:\n",a)
print("全1矩阵:\n",b)
print("单位矩阵:\n",c)
print("随机矩阵:\n",d)






# #### 5. 建立一个数组 a,(值为[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] ) ,(1)打印a; (2)输出  下标为(2,3),(0,0) 这两个数组元素的值

a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

print("\n第五题:")
print("数组a:\n",a)
print("(2,3)元素:",a[2,3])
print("(0,0)元素:",a[0,0])



# #### 6.把上一题的 a数组的 0到1行 2到3列，放到b里面去，（此处不需要从新建立a,直接调用即可）(1),输出b;(2) 输出b 的（0,0）这个元素的值


b = a[0:2,2:4]

print("\n第六题:")
print("输出b:\n",b)
print("输出b(0,0):",b[0,0])



#  #### 7. 把第5题中数组a的最后两行所有元素放到 c中，（提示： a[1:2, :]）(1)输出 c ; (2) 输出 c 中第一行的最后一个元素（提示，使用 -1                 表示最后一个元素）

c = a[1:4,:]
print("\n第七题:")
print("数组a的最后两行:\n",c)
print("c中第一行最后一个元素:",c[0,-1])




# #### 8.建立数组a,初始化a为[[1, 2], [3, 4], [5, 6]]，输出 （0,0）（1,1）（2,0）这三个元素（提示： 使用 print(a[[0, 1, 2], [0, 1, 0]]) ）

a = np.array([[1, 2],
              [3, 4],
              [5, 6]])

print("\n第八题:")
print("输出s(0,0)&(1,1)&(2,0):",a[[0, 1, 2], [0, 1, 0]])




# #### 9.建立矩阵a ,初始化为[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]，输出(0,0),(1,2),(2,0),(3,1) (提示使用 b = np.array([0, 2, 0, 1])                     print(a[np.arange(4), b]))

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

b = np.array([0, 2, 0, 1])
print("\n第九题:")
print("输出(0,0)&(1,2)&(2,0)&(3,1):",a[np.arange(4), b])




# #### 10.对9 中输出的那四个元素，每个都加上10，然后重新输出矩阵a.(提示： a[np.arange(4), b] += 10 ）

a[np.arange(4), b] += 10
print("\n第十题:")
print("全部加10:\n",a)




# ### array 的数学运算

# #### 11.  执行 x = np.array([1, 2])，然后输出 x 的数据类型
print("\n第十一题:")
x = np.array([1, 2])
print("x 的数据类型:",x.dtype)




# #### 12.执行 x = np.array([1.0, 2.0]) ，然后输出 x 的数据类类型

print("\n第十二题:")
x = np.array([1.0, 2.0])
print("x 的数据类型:",x.dtype)



# #### 13.执行 x = np.array([[1, 2], [3, 4]], dtype=np.float64) ，y = np.array([[5, 6], [7, 8]], dtype=np.float64)，然后输出 x+y ,和 np.add(x,y)

print("\n第十三题:")
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
print("输出x+y:\n",x+y)
print("输出np.add(x,y):\n",np.add(x,y))



# #### 14. 利用 13题目中的x,y 输出 x-y 和 np.subtract(x,y)

print("\n第十四题:")
print("输出x-y:\n",x-y)
print("输出np.subtract(x,y):\n",np.subtract(x,y))




# #### 15. 利用13题目中的x，y 输出 x*y ,和 np.multiply(x, y) 还有  np.dot(x,y),比较差异。然后自己换一个不是方阵的试试。

print("\n第十五题:")
print("输出x*y:\n",x*y)
print("输出:np.multiply(x, y)\n",np.multiply(x,y))
print("输出:np.dot(x,y)\n",np.dot(x,y))


# #### 16. 利用13题目中的x,y,输出 x / y .(提示 ： 使用函数 np.divide())

print("\n第十六题:")
print("输出x/y:\n",np.divide(x,y))




# #### 17. 利用13题目中的x,输出 x的 开方。(提示： 使用函数 np.sqrt() )

print("\n第十七题:")
print("输出np.sqrt(x):\n",np.sqrt(x))




# #### 18.利用13题目中的x,y ,执行 print(x.dot(y)) 和 print(np.dot(x,y))


print("\n第十八题:")
print("x.dot(y):\n",x.dot(y))
print("np.dot(x,y):\n",np.dot(x,y))




# ##### 19.利用13题目中的 x,进行求和。提示：输出三种求和 (1)print(np.sum(x)):   (2)print(np.sum(x，axis =0 ));   (3)print(np.sum(x,axis = 1))

print("\n第十九题:")
print("输出np.sum(x):",np.sum(x))
print("输出np.sum(x,axis =0 ):",np.sum(x,axis =0 ))
print("输出np.sum(x,axis = 1):",np.sum(x,axis = 1))



# #### 20.利用13题目中的 x,进行求平均数（提示：输出三种平均数(1)print(np.mean(x)) (2)print(np.mean(x,axis = 0))(3) print(np.mean(x,axis =1))）

print("\n第二十题:")
print("输出np.mean(x):",np.mean(x))
print("输出np.mean(x,axis = 0):",np.mean(x,axis = 0))
print("输出np.mean(x,axis = 1):",np.mean(x,axis =1))




# #### 21.利用13题目中的x，对x 进行矩阵转置，然后输出转置后的结果，（提示： x.T 表示对 x 的转置）

print("\n第二十一题:")
print("输出转置:\n",x.T)




# #### 22.利用13题目中的x,求e的指数（提示： 函数 np.exp()）

print("\n第二十二题:")
print("求e的指数:\n",np.exp(x))




# #### 23.利用13题目中的 x,求值最大的下标（提示(1)print(np.argmax(x)) ,(2) print(np.argmax(x, axis =0))(3)print(np.argmax(x),axis =1))

print("\n第二十三题:")

print("输出np.argmax(x):",np.argmax(x))
print("输出np.argmax(x, axis =0):",np.argmax(x, axis =0))
print("输出np.argmax(x,axis =1):",np.argmax(x,axis =1))



# #### 24,画图，y=x*x 其中 x = np.arange(0, 100, 0.1) （提示这里用到  matplotlib.pyplot 库）

print("\n第二十四题:")

x = np.arange(0, 100, 0.1)
y = x * x

plt.figure(figsize=(5, 3))
plt.plot(x, y, label='$y = x^2$', color='blue')

plt.title('Graph of $y = x^2$')
plt.xlabel('x')
plt.ylabel('y')

plt.legend()
plt.grid(True)
plt.show()

# #### 25.画图。画正弦函数和余弦函数， x = np.arange(0, 3 * np.pi, 0.1)(提示：这里用到 np.sin() np.cos() 函数和 matplotlib.pyplot 库)

print("\n第二十五题:")
plt.rcParams['figure.dpi'] = 300

x = np.arange(0, 3 * np.pi, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(5, 3))
plt.plot(x, y1, label='$y = np.sin(x)$', color='blue')
plt.plot(x, y2, label='$y = np.cos(x)$', color='green')

plt.title('$y1 = np.sin(x)$, $y2 = np.cos(x)$')
plt.xlabel('x')
plt.ylabel('y')

plt.legend()
plt.grid(True)
plt.show()


