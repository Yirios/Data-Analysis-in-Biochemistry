from matplotlib.pylab import plt
import numpy as np
from statistics import stdev


class TisTG:
    def __init__(self,C0=2.26) -> None:
        self.__c0 = C0
        self.__a0 = 0
        
    def __C(self,A):
        c = (A-self.__a0)/(self.__ac-self.__a0)*self.__c0
        return c
    
    def setdata(self,data):

        self.__ac = data['标准']
        self.__abs = data['血清']
        self.__als = data['肝脏']
        self.__ams = data['肌肉']

        self.__cbs = [self.__C(i) for i in self.__abs]
        self.__cls = [self.__C(j) for j in self.__als]
        self.__cms = [self.__C(k) for k in self.__ams]
        self.__yerrs =[stdev(self.__cbs),stdev(self.__cls),stdev(self.__cms)]
        self.__values = [sum(self.__cbs)/len(self.__cbs),
                         sum(self.__cls)/len(self.__cls),
                         sum(self.__cms)/len(self.__cms),]
        
    def get_result(self):
        return self.__cbs,self.__cls,self.__cms

    def fig(self):
        plt.figure(figsize=(6,6), dpi=100)
        plt.subplot(1, 1, 1)
        # 包含每个柱子下标的序列
        index = np.arange(1,4)
        plt.bar(index, 
                self.__values,
                width = 0.35,
                yerr = self.__yerrs,
                capsize=5,
                ecolor = '#424874',
                color="#A6B1E1")
        X = [1]*len(self.__cbs)+[2]*len(self.__cls)+[3]*len(self.__cms)
        Y = self.__cbs+self.__cls+self.__cms
        plt.scatter(X,Y,color='#424874')
        plt.ylabel('TG concentration(mmol/L)')
        plt.xlabel('tissues')
        plt.title('TG concentrations in different tissues')
        plt.xticks(index, ('Serum','Liver','Muscle'))
        plt.savefig('TG_tissues1.png')

        plt.show()


if __name__ =="__main__":
    data0 = {'标准':0.292,
            '血清':[0.020,0.009,0.032],
            '肝脏':[0.026,0.013,0.025],
            '肌肉':[0.080,0.036,0.024]}
    
    data1 = {'标准':0.307,
            '血清':[0.027,0.015,0.027],
            '肝脏':[0.018,0.024,0.033],
            '肌肉':[0.041,0.067,0.049]}
    

    C0 = 2.26 #mmol/L
    mTG = TisTG(C0)
    mTG.setdata(data1)
    mTG.fig()


