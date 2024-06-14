# 郑州 A.Once In My Life

>这道题目就是一开始想到的就是和之前写过的题目一样,找到最终的状态,比如题目要求是需要1-9,那就把1-9都先写上,然后是把d给加上,然后想的是这个数字就是构造出来的n*k,然后计算或者说不断打乱去写,后面发现这个不对,于是看了题解

>题解的话前半部分和我想的是一样的,但是这个构造n*k是不对的,可以在原有的构造上把后面补充n的长度个0
>
>然后把这个构造好的数字/n然后向上取整
>
>就是结果了,我想应该是写清楚了,题解用的是那个10**log10(n)之类的,和这个n的长度个0是差不多.

```python
from sys import stdin
def input():
    return stdin.readline().rstrip()
from math import ceil
for _ in range(int(input())):
    n,d=map(int,input().split())
    x=(1234567890+d)*(10**len(str(n)))
    print(ceil(x/n))
```

