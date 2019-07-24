import random 
import math
def main():
    print '请输入迭代的次数：'
    n=int(raw_input())   #n是随机的次数  
    total=0   #total是所有落入圆内的随机点
    for i in xrange(n):
        x=random.random()
        y=random.random()
        if math.sqrt(x**2+y**2)<1.0:   #判断是否落入圆内
            total+=1
    mypi=4.0*total/n   #得到Pi值
    print '迭代次数是',n,'Pi的值是：',mypi
    print '数学pi：',math.pi
    print '误差是：',abs(math.pi-mypi)/math.pi   #计算误差
    
main()
