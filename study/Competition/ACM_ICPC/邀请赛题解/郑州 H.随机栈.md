# 郑州 H.随机栈

> 这个我的思路就是把输入的数组倒过来看，这样对于遇到-1的时候就相当于一个分界处，把后续读入的连续的不为-1的数字当成一组数字，这组数字输出的位置就放在b中变成一个列表，刚交了一发wa5，写题解的时候想到哪里错了，就是这个连续的这一段不一定是放在这个地方一起输出的，取决于按输入顺序时候，哪一段是输入的大于输出的-1的数目的时候这个时候输出的就是随机的，否则就是连续的一段随机。好吧，呜呜呜。想错了，估计写不出来了

***看了题解，思路是先最终的序列是定值，按照定值序列模拟看每一位是否能取到就行***，尝试了Counter，dict，list,也尝试过去掉len，还有在查询是否在temp数组中等等还是超时。好吧都超时在第九个样例，属实写不出了，明天还有早八在阿院呜呜呜，写不出来呜呜呜。

> 想起之前div4的不要算速度了，wok把速度提前算完会超时，不算，一个个模拟就不会超时，可是假设所有的点都需要算不同的速度我提前把速度算完有啥问题呜呜

```python
from sys import stdin
def input():
    return stdin.readline().rstrip()
# 超时了第九个样例，用Counter记录次数看能过不,TLE---2024.05.15.02:11
# 我不管思路没有问题
# from collections import Counter
def solve(a):
    global ans,p,q
    # 这个定义的是取出来之后的序列，这个一定是固定的所以写成std数组
    std=[]
    for item in a:
        if item==-1:
            continue
        else:
            std.append(item)
    std.sort()
    # print(std)
    # 这个temp代表我现在栈中有的哪些数组
    temp=[0]*(n+1)
    le=0
    ans=1
    p,q=1,1
    num=0
    maxn=-1
    # 遍历给定的顺序
    for i in range(2*n):
        # 如果是-1就是看能否弹出一个符合条件的数值，如果不行就是False
        if a[i]==-1:
            need=std[num]
            if need>maxn or temp[need]==0:
                return False
            # 否则就是计算下temp中的need的数目记录到p中累乘然后删除这个
            else:
                p*=temp[need]
                q*=le
                le-=1
                temp[need]-=1
            num+=1
        else:
            temp[a[i]]+=1
            le+=1
            maxn = max(maxn, a[i])
    else:
        return True

n=int(input())
a=list(map(int,input().split()))
mod=998244353
print(p*pow(q,-1,mod)%mod if solve(a) else 0)


```

