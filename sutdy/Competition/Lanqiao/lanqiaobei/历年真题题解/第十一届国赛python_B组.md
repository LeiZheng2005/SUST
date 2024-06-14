# 第十一届国赛python_B组

> 参考[小蓝刷题](https://blog.csdn.net/m0_55148406)
>
> 【蓝桥真题】——2020年蓝桥python组国赛真题+解析+代码（通俗易懂版）
>
> https://blog.csdn.net/m0_55148406/article/details/122863206?spm=1001.2014.3001.5506

> 参考[吧唧吧唧orz](https://blog.csdn.net/H20211009)
>
> 【python语言】第十一届蓝桥杯国赛 pyb组
>
> https://blog.csdn.net/H20211009/article/details/138362478?spm=1001.2014.3001.5506

## [试题A：美丽的2](https://blog.csdn.net/m0_55148406/article/details/122863206?spm=1001.2014.3001.5506#t1)

```python
cnt=0                         #计数器初始化
for i in range(1,2021):       #遍历公元1年到2020年
    if str(i).count("2")>0:   #如果年份中至少有一个2
        cnt+=1                #计数器+1
print(cnt)                    #输出结果：563
```

## [试题B：合数个数](https://blog.csdn.net/m0_55148406/article/details/122863206?spm=1001.2014.3001.5506#t5)

```python
#判断合数
def heshu(x):
    for i in range(2,x):
        if x%i==0:
            return True
    return False
 
#主函数
cnt=0
for i in range(1,2021):     #遍历1~2020
    if heshu(i)==True:      #如果判断为合数
        cnt+=1              #计时器+1
print(cnt)                  #输出结果：1713
    
```



## [试题C：阶乘约数](https://blog.csdn.net/m0_55148406/article/details/122863206?spm=1001.2014.3001.5506#t9)

> 这里有一个约数定理

```python
1.创建1~100的质数集
def prime(x):
    for i in range(2,int(x**0.5)+1):
        if x%i==0:
            return False
    return True
zhishu=[]
for i in range(2,101):
    if prime(i)==True:
        zhishu.append(i)
#zhishu=[2,3,5,7,11···,97]
        
#2.计算约数个数
p=[0]*101                #创建计数数组
for num in range(1,101): #遍历1~100
    x=num                #当前变量赋值
    for i in zhishu:     #遍历质数数组
        while x%i==0:    #判断约数
            p[i]+=1      #对应计数+1
            x//=i        #循环条件
#p=[0, 0, 97···0,0]
            
#3.遍历结果
ans=1
for i in range(1,101):   #遍历1~100
    if p[i]!=0:          #计数数组不为0
        ans*=(p[i]+1)    #根据公式累乘
print(ans)               #39001250856960000
```



## [试题D：本质上升序列](https://blog.csdn.net/m0_55148406/article/details/122863206?spm=1001.2014.3001.5506#t13)

```python
#本质上升序列
s='tocyjkdzcieoiodfpbgcncsrjbhmugdnojjddhllnofawllbhfiadgdcdjstemphmnjihecoapdjjrprrqnhgccevdarufmliqijgihhfgdcmxvicfauachlifhafpdccfseflcdgjncadfclvfmadvrnaaahahndsikzssoywakgnfjjaihtniptwoulxbaeqkqhfwl'
dp=[1]*len(s)               #dp:递增子序列的个数初始化为1
for i in range(len(s)):     #遍历0~len(s)-1
    for j in range(i):      #遍历0~i-1
        if s[i]>s[j]:       #如果当前字符大于以前的字符
            dp[i]+=dp[j]    #把当前字符添加到前面递增子序列的末尾    
        if s[i]==s[j]:      #如果当前字符等于以前的字符  
            dp[i]-=dp[j]    #去掉重复字符个数
print(sum(dp))              #输出结果：3616159
 
```



## [试题E：玩具蛇](https://blog.csdn.net/m0_55148406/article/details/122863206?spm=1001.2014.3001.5506#t22)

> DFS 写

```python
import sys
import os
import math
 
ans = 0    #数目统计
direction = [[1,0], [0,1], [-1,0], [0,-1]]  #4个方向选择
flag = [[0]*4 for _ in range(4)]
def dfs(x, y, count):
    global ans  
    # 终止条件
    if count==16:
        ans += 1
        return
     
    # 对4个方向进行搜素
    for d in direction:
        dx = x + d[0]
        dy = y + d[1]
        if 0<=dx<4 and 0<=dy<4 and flag[dx][dy]!=1:
            flag[dx][dy] = 1    #标记
            dfs(dx, dy, count+1)    #基于该点继续搜索
            flag[dx][dy] = 0    #上一条路径搜索完毕，消除痕迹
 
for i in range(4):
    for j in range(4):
        flag[i][j] = 1  #从盒子的不同起点开始依次搜索：A~P
        dfs(i, j, 1)
        flag[i][j] = 0  #取消上一次搜索的起点（以备下一次可选择）
print(ans)
```



## [试题F：天干地支](https://blog.csdn.net/m0_55148406/article/details/122863206?spm=1001.2014.3001.5506#t23)

```python
import sys
import os
import math
 
year = int(input())
tiangan = {1:"jia", 2:'yi', 3:'bing', 4:'ding', 5:'wu', 6:'ji', 7:'geng', 8:'xin', 9:'ren', 10:'gui'}
dizhi = {1:'zi', 2:'chou', 3:'yin', 4:'mao', 5:'chen', 6:'si', 7:'wu', 8:'wei', 9:'shen', 10:'you', 11:'xu', 12:'hai'}
 
left = 2020%12
year -= left
x = year%10+1
y = year%12+1
print(tiangan[x]+dizhi[y])
```



## 试题J：重复字符串

> 这个题目好像有点熟悉啊，是不是国赛题目考过一个，虽然只有四届，但是确实有类似的就是划分成k组然后怎样怎样呢

```python
import sys
import os
import math
 
k = int(input())
s = input()
ans = 0
 
"""
将序列分成了k份
每一份的内容都要相同
若字符串长度不为k的倍数 则无法修改
"""
if len(s)%k!=0:
    print("-1")
else:
    duration = len(s)//k
    """
    在每列中 重复最多的字母不用修改 其余字母修改为重复最多的字母
    一共可以划分为k组 每组有duration个元素
    若按列再划分 一共有duration个组 每组有k个元素
    """
    for i in range(duration):
        ds = dict()
        for j in range(k):
            d = s[duration * j + i] #按列获取元素
            if d not in ds: #在寻找元素中构建字典并计数
                ds[d] = 1
            else:
                ds[d] += 1
        ans += k-max(ds.values())
print(ans)
```



## [试题H：答疑](https://blog.csdn.net/m0_55148406/article/details/122863206?spm=1001.2014.3001.5506#t24)

```python
import sys
import os
import math
 
n = int(input())
s=[0]*n;a=[0]*n;e=[0]*n
for i in range(n):
    s[i], a[i], e[i] = map(int, input().split())
 
# 计算每位同学的总用时
t = [0]*n
for i in range(n):
    t[i] = a[i] + s[i] + e[i]
 
time = 0
time_collection = []
t_new = sorted(t)
index = [t.index(x) for x in t_new]
for i in range(len(t_new)):
    time += t_new[i]
    time_collection.append(time-e[index[i]])    #发消息的时间
print(sum(time_collection))
```



## 试题X：总结

> 可以写的题目：填空题ABCDE
