from matplotlib.pylab import plt
import numpy as np
from statistics import stdev
from TG import TisTG
import csv

def avg_sd(b,l,m):
    sds =[stdev(b),stdev(l),stdev(m)]
    avgs = [sum(b)/len(b),sum(l)/len(l),sum(m)/len(m)]
    return avgs,sds

def print_result(avgs,sds):
    print(f'血清中甘油三酯的含量为{avgs[0]},标准误差为{sds[0]}')
    print(f'肝脏中甘油三酯的含量为{avgs[1]},标准误差为{sds[1]}')
    print(f'肌肉中甘油三酯的含量为{avgs[2]},标准误差为{sds[2]}')

def fig(b,l,m):
    avgs,sds =avg_sd(b,l,m)
    print_result(avgs,sds)
    plt.figure(figsize=(6,6), dpi=100)
    plt.subplot(1, 1, 1)
    # 包含每个柱子下标的序列
    index = np.arange(1,4)
    plt.bar(index, 
            avgs,
            width = 0.35,
            yerr = sds,
            capsize=5,
            ecolor = '#424874',
            color="#A6B1E1")
    X = [1]*len(b)+[2]*len(l)+[3]*len(m)
    Y = b+l+m
    plt.scatter(X,Y,color='#424874')
    plt.ylabel('TG concentration(mmol/L)')
    plt.xlabel('tissues')
    plt.title('TG concentrations in different tissues')
    plt.xticks(index, ('Serum','Liver','Muscle'))
    plt.savefig('TG_tissues.png')

    plt.show()

if __name__=='__main__':

    datas= [{'标准':0.292,
            '血清':[0.020,0.009,0.032],
            '肝脏':[0.026,0.013,0.025],
            '肌肉':[0.080,0.036,0.024]},
            
            {'标准':0.307,
            '血清':[0.027,0.015,0.027],
            '肝脏':[0.018,0.024,0.033],
            '肌肉':[0.041,0.067,0.049]}]

    bs = list()
    ls = list()
    ms = list()
    header  = ['血清','肝脏','肌肉']

    data =list()

    for line in datas:
        mTG = TisTG()
        mTG.setdata(line)
        b,l,m = mTG.get_result()
        bs.extend(b)
        ls.extend(l)
        ms.extend(m)

    bs.insert(0,'血清')
    ls.insert(0,'肝脏')
    ms.insert(0,'肌肉')

    data = np.array([bs,ls,ms]).T.tolist()
    with open('tissues_C_TG.csv','w',encoding='utf8',newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    
    fig(bs[1:],ls[1:],ms[1:])

