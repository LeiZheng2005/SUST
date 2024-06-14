# 郑州 D.距离之比

> 直接看题解吧



![距离之比题解](D:\01系统\桌面文件\距离之比.png)

```python
from sys import stdin
def input():
    return stdin.readline().rstrip()

for _ in range(int(input())):
    n=int(input())
    list0=[]
    for __ in range(n):
        x,y=map(int,input().split())
        list0.append([x,y])
    # print(list0)
    ans=float('-inf')
    list0.sort(key=lambda x: x[0]+x[1])
    for i in range(n-1):
        deltax=abs(list0[i][0]-list0[i+1][0])
        deltay=abs(list0[i][1]-list0[i+1][1])
        cnt=(deltax+deltay)/pow(deltax**2+deltay**2,0.5)
        ans=max(ans,cnt)


    list0.sort(key=lambda x:x[0]-x[1])
    for i in range(n-1):
        deltax=abs(list0[i][0]-list0[i+1][0])
        deltay=abs(list0[i][1]-list0[i+1][1])
        cnt=(deltax+deltay)/pow(deltax**2+deltay**2,0.5)
        ans=max(ans,cnt)

    print(ans)
```

