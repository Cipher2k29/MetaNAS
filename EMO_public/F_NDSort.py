import numpy as np
from public import sortrows

def F_NDSort(PopObj,Operation):
    list = ["all", "half", "first"] # list 与 array 不同, matlab 中大多数操作 仅支持 np.array
    kind = list.index(Operation)
    N,M = PopObj.shape
    FrontNO = np.inf*np.ones((1, N))
    MaxFNO = 0
    PopObj, rank = sortrows.sortrows(PopObj)


    while (kind <= 0 and np.sum(FrontNO < np.inf) < N) or (kind <= 1 and np.sum(FrontNO < np.inf) < (N/2)) or(kind <= 2 and MaxFNO < 1):
        MaxFNO += 1
        for i in range(N):
            if FrontNO[0, i] == np.inf:
                Dominated = False
                for j in range(i-1, -1, -1):
                    if FrontNO[0, j] == MaxFNO:
                        m=2
                        while (m <= M) and (PopObj[i, m-1] >= PopObj[j, m-1]):
                            m += 1
                        Dominated = m > M
                        if Dominated or (M == 2):
                            break
                if not Dominated:
                    FrontNO[0,i] = MaxFNO
    # temp=np.loadtxt("temp.txt")
    # print((FrontNO==temp).all())
    front_temp = np.zeros((1,N))
    front_temp[0, rank] = FrontNO
    # FrontNO[0, rank] = FrontNO 不能这么操作，因为 FrontNO值 在 发生改变 ，会影响后续的结果


    return front_temp, MaxFNO





