import numpy as np
# numpy.array
exam = np.array([[1,2,3],[4,5,6]])
# print(exam)

#定义结构化数据
Stu = np.dtype([('name','S20'),('age','i1')])
# print(Stu)
stu_exam = np.array([('Zhangsan',18),('Wuji',28)],dtype= Stu)
# print(stu_exam)

# ===================================================================
# 举例
a1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print('a[0:2]:::',a[0:2])
# print('a[0:2][0]:::',a[0:2][1])

# print(a[0][2])  # 先行后列

# =====================================================================
#计算 
#数学函数
#1、NumPy 提供了标准的三角函数： sin()、cos()、tan()\
#2、舍入函数：numpy.around()函数返回指定数字的四舍五入值。

a2 = np.array([1.0, 5.55, 123, 0.567, 25.532])
# print(np.around(a))
# [  1.   6. 123.   1.  26.]

# numpy.floor()返回小于或等于指定表达式的最大整数，及向下取整.
# print(np.floor(a))
# [  1.   5. 123.   0.  25.]

#numpy.ceil()返回大于或者指定表达式的最小整数，即向上取整。
# print(np.ceil(a))
# [  1.   6. 123.   1.  26.]

# ====================================================================
# https://www.runoob.com/numpy/numpy-statistical-functions.html
# numpy.amin() 用于计算数组中的元素沿指定轴的最小值。
# numpy.amax() 用于计算数组中的元素沿指定轴的最大值。
# numpy.ptp()函数计算数组中元素最大值与最小值的差（最大值 - 最小值)。
# numpy.percentile() 百分位数是统计中使用的度量，表示小于这个值的观察值的百分比。 函数numpy.percentile()接受以下参数。
# numpy.median() 函数用于计算数组 a 中元素的中位数（中值）
# numpy.mean() 函数返回数组中元素的算术平均值。 如果提供了轴，则沿其计算。
a3 = np.array([[3,7,5],[8,4,3],[2,4,9]])
# print ('调用 amin() 函数：')
# print (np.amin(a3,1)) # [3 3 2]
# print ('\n')
# print ('再次调用 amin() 函数：') 
# print (np.amin(a3,axis = 0)) # [2 4 3]
# print ('\n')
# print ('调用 amax() 函数：')
# print (np.amax(a3)) # 9
# print ('\n')
# print ('再次调用 amax() 函数：')
# print (np.amax(a3,0))  # axis = 0 相当于 0   # [8 7 9]


a4 = np.array([[3,7,5],[8,4,3],[2,4,9]])  
print ('我们的数组是：')
print (a4)
print ('\n')
print ('调用 ptp() 函数：')
print (np.ptp(a4))
print ('\n')
print ('沿轴 1 调用 ptp() 函数：')
print (np.ptp(a4, axis =  1))
print ('\n')
print ('沿轴 0 调用 ptp() 函数：')
print (np.ptp(a4, axis =  0))

# numpy.percentile() 百分位数是统计中使用的度量，表示小于这个值的观察值的百分比。函数numpy.percentile()接受以下参数。
# numpy.percentile(a, q, axis)
# a: 输入数组
# q: 要计算的百分位数，在 0 ~ 100 之间
# axis: 沿着它计算百分位数的轴
# 第 p 个百分位数是这样一个值，它使得至少有 p% 的数据项小于或等于这个值，且至少有 (100-p)% 的数据项大于或等于这个值。
# 举个例子：高等院校的入学考试成绩经常以百分位数的形式报告。比如，假设某个考生在入学考试中的语文部分的原始分数为 54 分。
# 相对于参加同一考试的其他学生来说，他的成绩如何并不容易知道。但是如果原始分数54分恰好对应的是第70百分位数，
# 我们就能知道大约70%的学生的考分比他低，而约30%的学生考分比他高。
# 这里的 p = 70。