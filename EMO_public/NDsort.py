import numpy as np
import time
from EMO_public import sortrows

def NDSort(PopObj,Remain_Num):

    N,M = PopObj.shape
    print("obj Shape", PopObj.shape)
    print("Remain_Num", Remain_Num)
    FrontNO = np.inf*np.ones((1, N))
    MaxFNO = 0
    PopObj, rank = sortrows.sortrows(PopObj)
    print("rank", rank)


    while (np.sum(FrontNO < np.inf) < Remain_Num) and (MaxFNO <= 8): # 230919: add MaxFNO<=8 to avoid meaningless cycle 
        MaxFNO += 1
        print(MaxFNO)
        # time.sleep(2)
        for i in range(N):
            if FrontNO[0, i] == np.inf:
            
                Dominated = False
                print(i, end=" ")
                for j in range(i-1, -1, -1):
                    if FrontNO[0, j] == MaxFNO:
                        m=2
                        while (m <= M) and (PopObj[i, m-1] >= PopObj[j, m-1]):
                            m += 1
                        Dominated = m > M
                        print(m, Dominated, end=" ")
                        if Dominated or (M == 2):
                            break
                if not Dominated:
                    FrontNO[0,i] = MaxFNO
                    print("value, changed", end=" ")
                print()
    # temp=np.loadtxt("temp.txt")
    # print((FrontNO==temp).all())
    front_temp = np.zeros((1,N))
    front_temp[0, rank] = FrontNO
    # FrontNO[0, rank] = FrontNO 不能这么操作，因为 FrontNO值 在 发生改变 ，会影响后续的结果


    return front_temp, MaxFNO





