import random
import bisect
import numpy as np
# SIZE = 7
# random.seed(1729) 
# my_list = []
# for i in range(SIZE):
#     new_item = random.randrange(SIZE * 2)
#     bisect.insort(my_list, new_item)
#     print('%2d ->' % new_item, my_list)
# random.seed(1)
# for i in range(1,5):
#     print(random.randrange(10))
a = np.arange(6).reshape((3, 2))
print(a)
print('-------\n')
print(np.reshape(a,(-1,1)))
print('-------\n')
print(np.reshape(a,(2,-1)))