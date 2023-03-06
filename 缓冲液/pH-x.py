from matplotlib.pylab import plt
import numpy as np
import pandas as pd
from scipy import optimize as op


def x_pH(ph,pk1=2.34,pk2=9.60,C=0.025,pkw=14):
    M = 10**(pk1-ph)+10**(ph-pk2)+1
    x = (10**(pk1-ph)-10**(ph-pk2))*C/M+10**(-ph)-10**(ph-pkw)
    return -x


def show_data(data,label,C):
    d = 0.0001 
    start,end = 0,14

    X = np.linspace(start, end, int((end-start)/d),endpoint=True) 
    Y = pd.DataFrame(X)
    Y = Y.apply(lambda i:x_pH(i,C=C)).values #反解真实曲线

    fig, ax = plt.subplots()
    ax.scatter(label,data)  #数据点
    ax.plot(Y, X, linewidth=2.0) #真实曲线
    ax.set(xlim=(-0.03, 0.03),ylim=(0,13))
    plt.show()


    a = np.linspace(5.9,6.1,10000)
    for i in a:
        if abs(x_pH(i))<1e-10:
            print(i)
            break
    print(f'等电点{(9.6+2.34)/2}时加入的当量{x_pH((9.6+2.34)/2)}')
    print(f'pK2时加入的当量{x_pH((9.6))}')
    print(f'pK2时加入的当量{x_pH((2.34))}')




def show_fit(data,label,C):
    data = np.array(data)
    label = np.array(label)
    pk1,pk2 = op.curve_fit(lambda ph,pK1,pK2:x_pH(ph,pK1,pK2),data,label,bounds=(0, 10))[0]
    print(pk1,pk2)

    d = 0.0001
    start,end = 0,14
    X = np.linspace(start, end, int((end-start)/d),endpoint=True)
    Y = pd.DataFrame(X)
    Y = Y.apply(lambda i:x_pH(i)).values #反解真实曲线
    Yf = pd.DataFrame(X)
    Yf = Yf.apply(lambda ph:x_pH(ph,pk1=pk1,pk2=pk2)).values #拟合曲线

    fig, ax = plt.subplots()
    ax.scatter(label,data)  #数据点
    ax.plot(Yf, X, linewidth=2.0) #拟合曲线
    ax.plot(Y, X, linewidth=2.0) #真实曲线
    ax.set(xlim=(-0.03, 0.03),ylim=(0,13))
    plt.show()

if __name__=="__main__":
    data1_1 = [3,3.5,4,4.5,5,5.5,5.5,5.5,6.5,7.5,8,9.5,8,9,10]
    data1_2 = [3,3.5,4,4.5,5,5.5,5.5,5.5,6.5,7.5,8,9.5,9.5,9.5,10]
    data1_3 = [2,2.5,2.5,3,3,3.5,3.5,6,8,8.5,9,9.5,9.5,10,11]
    data2 = [1,2,2,3,3,4,5,6,8,9,9,9,9,10,11]
    label = [-0.025,-0.02,-0.015,-0.0125,-0.01,-0.005,-0.0025,0,0.0025,0.005,0.01,0.0125,0.015,0.02,0.025]
    C = 0.025 

    data = data1_2
    show_data(data,label,C)
    show_fit(data,label,C)