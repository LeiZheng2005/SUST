# 第十三届国赛python_B组

## [试题A：斐波那契与7](https://blog.csdn.net/qq_52007481/article/details/125612645?spm=1001.2014.3001.5506#A7_9)

填空题基本都是有规律的，先打印出来看就OK，斐波那契基本就是60一次循环。

```python

f1 = 0
f2 = 1
f3 = 1
c = 0   # 统计出现的次数
for i in range(60):
    print(f3, end=' ')  # 斐波那契数列中的数字
    f3 = (f1+f2)%10
    if f3 ==7:
        c+=1
    f1 = f2
    f2 = f3
# 1 1 2 3 5 8 3 1 4 5 9 4 3 7 0 7 7 4 1 5 6 1 7 8 5 3 8 1 9 0 9 9 8 7 5 2 7 9 6 5 1 6 7 3 0 3 3 6 9 5 4 9 3 2 5 7 2 9 1 0
# 经过验证，60个一循环 每个循环里面有8个个位为7的数字
print()
print(c)
print(202202011200//60)  # 一共3370033520次循环
print((202202011200//60)*8) # 26960268160

```



## [试题 B: 小蓝做实验](https://blog.csdn.net/qq_52007481/article/details/125612645?spm=1001.2014.3001.5506#_B__38)

这个和十四届的第二题类似，只不过是去寻找素数的个数。十四届是寻找最大面积其实，预估15届也是一个txt文件，但是需要就是知道怎么读如txt文件很重要。

```python
# 读如txt时候这样子：
f=open(r'nums.txt','r',encoding='utf-8')
list0=f.read().split()
```



```python
# 运用埃氏筛法进行解题
# 因为只有少部分的数据大于10**8,将数据分为两部份，小于10**8的，大于10**8
f = open(r'primes.txt','r',encoding='utf-8')  
txt = f.read().split()         # 讲文件内的东西转化为列表 
arr1 = [int(i) for i in txt if len(i)<=8]  # 根据长度分，然后在转换为整型
arr2 = [int(i) for i in txt if len(i)>8]   # 长度为170，所以单独判断花费的时间并不长

# 先默认所有的都为质数(这部分可以看我质数筛的文章)
# 埃氏筛选法效率非常高2分42秒能够找出10*8以内的质数
nums = [True for i in range(10**8+1)] 
for i in (range(3,10**8+1)):
    if nums[i]:
        for j in range(i+i,10**8,i):
            nums[j] = False
c=0 # 记录次数
for i in arr1:  # 根据列表里面的值判断是否为质数
    if nums[i]:
        c+=1
for i in arr2:   # 对大于10**8的数进行判断
    for j in range(2,int(i**0.5)+1):  
            if i%j == 0:
                break
    else:
        c+=1
print(c)

```



## [试题 C: 取模](https://blog.csdn.net/qq_52007481/article/details/125612645?spm=1001.2014.3001.5506#_C__74)

这个暴力就OK，哦对有一个反证法：就是这个如果是不存在的话那么对1，2，3，4，5，6，，，，n-1这些数字取模的话得到的就是0，1，2，3，4，5，，，，n-2这些肯定都是不也一样的，不然就会存在。
```python
# 暴力
t = int(input())
nums = []
for i in range(t):
    n,m = input().split()
    n = int(n)
    m = int(m)
    f = False
    for j in range(1,m+1):
        if f:
            break
        for k in range(j+1,m+1):
            if n%j==n%k:
                f = True
                nums.append('Yes')
                break
        
    else:
        nums.append('No')
for i in nums:
    print(i)

```



```python
# 反证法代码：
from sys import stdin
T = int(input())
for _ in range(T) :
    n, m = map(int, stdin.readline().split())
    flag = True
    for i in range(1, m + 1) :
        if n % i != i - 1 :
            print('Yes')
            flag = False
            break
    if flag :
        print('No')
```



## [试题 D: 内存空间](https://blog.csdn.net/qq_52007481/article/details/125612645?spm=1001.2014.3001.5506#_D__101)

第四题也就是蓝桥杯国赛大题的第二题，是一道大模拟题目，和十四届是一样的，这两道题目需要重点学习一下，虽然不考察算法，但是考验代码的基本功，考验考试的心态，这道题目放最后做感觉心态不好就做不出来。

```python
t = int(input())
zong = 0 # 总大小，单位B
for i in range(t):
    s_lst = input().split()
    if s_lst[0] == 'int':     # 根据不同的输入情况进行分类
        st1 = s_lst[1].split(',')
        zong+=len(st1)*4  
    elif s_lst[0] == 'long':
        st1 = s_lst[1].split(',')
        zong+=len(st1)*8 
    elif  s_lst[0] == 'String':
        st1 = s_lst[1].split(',')
        num = 0
        for item in st1:
            num+=len(item.split('=')[1])-2
        zong+=num-1
    elif s_lst[0] == 'int[]':
        num = 0
        for it in range(1,len(s_lst)):
            if 'long' in s_lst[it] and ';' not in s_lst[it]:
                num += int(s_lst[it][4:-1])
            elif 'long' in s_lst[it] and ';' in s_lst[it]:
                num += int(s_lst[it][4:-2])
        zong+=num*4
    elif  s_lst[0] == 'long[]':
        num = 0
        for it in range(1,len(s_lst)):
            if 'long' in s_lst[it] and ';' not in s_lst[it]:
                num += int(s_lst[it].split(',')[0][5:-1])
            elif 'long' in s_lst[it] and ';' in s_lst[it]:
                num += int(s_lst[it][5:-2])
                
        zong+=num*8

z = [0,0,0,0]   # B，KB，MB，GB 前的数值
for i in range(4):
    z[i]=zong%1024
    zong = zong//1024
    if zong <=0:
        break

result = ''
result_st = ['GB','MB','KB','B']
for i in range(1,len(z)+1):
    if z[4-i] != 0:
        result+=f'{z[4-i]}{result_st[i-1]}'
print(result)


```



## [试题 E: 近似 GCD](https://blog.csdn.net/qq_52007481/article/details/125612645?spm=1001.2014.3001.5506#_E__GCD_159)

这个可以把能整除的看成0，不能的看成1，然后求一段连续的子序列至少长度为2的，有多少个满足最多只有一个1。

```python
import math
n,g=map(int,input().split())
a=[0]+list(map(int,input().split()))

re=0
left=1
right=1
temp=0#记录的上一个不是g的数的位置
for right in range(1,n+1):
    t=math.gcd(g,a[right])
    if t != g:#如果当前这个数不是g的倍数
        left=temp+1
        temp=right
    if right-left+1>=2:re+=right-left
print(re)
```



## [试题 F: 交通信号](https://blog.csdn.net/qq_52007481/article/details/125612645?spm=1001.2014.3001.5506#_F__205)

pass



## [试题 G: 点亮](https://blog.csdn.net/qq_52007481/article/details/125612645?spm=1001.2014.3001.5506#_G__209)

Pass



## [试题 H: 打折](https://blog.csdn.net/qq_52007481/article/details/125612645?spm=1001.2014.3001.5506#_H__214)

pass



## [试题 I: owo](https://blog.csdn.net/qq_52007481/article/details/125612645?spm=1001.2014.3001.5506#_I_owo_258)

pass



## [试题 J: 替换字符](https://blog.csdn.net/qq_52007481/article/details/125612645?spm=1001.2014.3001.5506#_J__261)

```python
# 暴力模拟就OK,只过了80%但非常多了，这还是最后一道题目，相当于是写出来了的。
s = input()
m = int(input())
for i in range(m):
    nums = input().split()
    l = int(nums[0])
    r = int(nums[1])
    x = nums[2]
    y = nums[3]

    s1 = s[0:l-1]
    s2 = s[l-1:r]
    s3 = s[r:]
    s2 = s2.replace(x,y)
    s = s1+s2+s3
print(s)

```



