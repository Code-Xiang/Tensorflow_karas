import numpy as np
# numpy.array
exam = np.array([[1,2,3],[4,5,6]])
print(exam)

#定义结构化数据
Stu = np.dtype([('name','S20'),('age','i1')])
print(Stu)
stu_exam = np.array([('Zhangsan',18),('Wuji',28)],dtype= Stu)
print(stu_exam)