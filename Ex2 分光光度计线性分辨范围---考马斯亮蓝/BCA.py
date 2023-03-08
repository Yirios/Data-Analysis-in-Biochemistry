from matplotlib.pylab import plt
import numpy as np
import csv
from sklearn.linear_model import LinearRegression

class Opsp:

    def __init__(self,data,r,label) -> None:
        self.__data = data
        self.__label = label
        self.__r = r
        self.__len = len(label)
        self.__result = []
        self.__best = None
        self.__group = '_test'
    
    def name(self,group):
        self.__group = group

    
    def fit(self):
        for n in range(self.__len,4,-1):
            rl = list()
            for i in range(self.__len-n+1):
                r = self.__linefit(i,i+n)
                rl.append(r)
            if max(rl)>=self.__r:
                index = rl.index(max(rl))
                self.__best = self.__result[index+n-self.__len-1]
                self.__type = [self.__group,'good']+self.__best
                self.__d = self.__best[3]
                self.__k = self.__best[4]
                break
        else:
            index = rl.index(max(rl))
            self.__best = self.__result[index+n-self.__len-1]
            self.__type = [self.__group,'bad']+self.__best
            self.__d = self.__best[3]
            self.__k = self.__best[4]


    def __linefit(self,i,j):
        Y = np.array(self.__data[i:j])
        X = np.array(self.__label[i:j]).reshape((-1, 1))
        model_line = LinearRegression()
        model_line.fit(X,Y)
        r = model_line.score(X,Y)
        d = model_line.intercept_
        k = model_line.coef_[0]
        self.__result.append([i+1,j,r,d,k])
        return r

    def fig(self,savefig=False,showfig=True):
        i = self.__best[0]-1
        j = self.__best[1]

        l = 0.001
        st = self.__label[i]
        ed =  self.__label[j-1]
        X_line = np.linspace(st,ed,int((ed-st)/l),endpoint=True) 
        Y_line = self.__k*X_line+self.__d


        X_u = self.__label[i:j]
        Y_u = self.__data[i:j]
        X_nu = self.__label[0:i]+self.__label[j:]
        Y_nu = self.__data[0:i]+self.__data[j:]

        fig,ax = plt.subplots()

        ax.set_title(f'{self.__group}:A-C') #标题
        ax.set_ylabel('A')  #y轴标签
        ax.set_xlabel('C')  #x轴标签

        ax.scatter(X_u,Y_u,color = 'hotpink')
        ax.scatter(X_nu,Y_nu, color = '#88c999')  
        ax.plot(X_line, Y_line, linewidth=1.5,label=f'A={self.__k}x+{self.__d}',color='blue')
        plt.legend(loc=0)

        if savefig :
            plt.savefig(f'result/fig/fig{self.__group}.png',dpi=720)
        
        if showfig:
            plt.show()
    
    def get_best(self):

        return self.__type
    
    def result(self,savedata = False):
        if savedata:
            with open(f'result/process/process{self.__group}.csv','w',encoding='utf-8',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['起始组','终止组','r','d','k'])
                writer.writerows(self.__result)
        result = f'''第{self.__group}组，在R>={self.__r}的条件下，R为{self.__best[2]}，线性部分的起始浓度为{self.__label[self.__best[0]-1]}终止浓度为{self.__label[self.__best[1]-1]}，截距为{self.__d}，斜率为{self.__k}'''
        return result
    
if __name__ == '__main__':
    label = [0,0.005,0.010,0.02,0.04,0.06,0.08,0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10]
    data = [0,0.024,0.021,0.015,0.132,0.202,0.207,0.221,0.474,0.807,1.246,1.439,1.67,2.016,2.030,2.023,2.017,2.011]
    R = 0.98

    model = Opsp(data,R,label)
    model.name('BCA')
    model.fit()
    model.fig(savefig=True,showfig=True)
    print(model.result(True))