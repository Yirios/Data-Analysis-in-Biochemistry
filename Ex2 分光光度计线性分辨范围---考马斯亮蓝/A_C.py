from matplotlib.pylab import plt
import numpy as np
import pandas as pd 
from scipy.stats import chi2
from scipy import optimize as op
import math as m
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

def Cchi2(x,c,k,df):
    return c*chi2.pdf(m.exp(k)*x, df)

def count(bst,R,label,confidence = 0.95):
    bst = pd.DataFrame(bst,columns=['group','type','start','end','R','d','k'])
    bst_bad = bst[bst['type'] =='bad']
    print('\n舍去的组：')
    print(bst_bad)
    bst_good = bst[bst['type'] =='good']
    bst_good_A = bst_good[bst_good['group'] == 0].values.tolist()[0]
    bst_good_B = bst_good[bst_good['k']>2.8]
    k_mean_b = np.mean(bst_good_B['k'])
    bst_good_B = bst_good_B.values.tolist()
    bst_good_C = bst_good[bst_good['k']<2].values.tolist()
    print('\n保留的组：')
    print(bst_good)

    #bst_good_c=[[bst_good_A[0],label[bst_good_A[2]-1],label[bst_good_A[3]-1],bst_good_A[4]-R]]
    bst_good_c=[]
    
    for line in bst_good_B:
        g = line[0]
        s = label[line[2]-1]
        e = label[line[3]-1]
        bst_good_c.append([g,s,e,line[4]-R])

    for line in bst_good_C:
        g = line[0]
        s = label[line[2]-1]*line[6]/k_mean_b
        e = label[line[3]-1]*line[6]/k_mean_b
        bst_good_c.append([g,s,e,line[4]-R])
    
    #bst_good = pd.DataFrame(bst_good_c,columns=['group','start','end','R'])

    n = 100000
    st,en = 0,0.5
    X = [i/n*(en-st) for i in range(n+1)]
    Y = [0]*(n+1)
    for mo in bst_good_c:
        for i in range(n+1):
            if mo[1]<X[i]<mo[2]:
                Y[i]+=mo[3]
    X_f = np.array(X)
    Y_f = np.array(Y)
    C,k,df = op.curve_fit(Cchi2,X_f,Y_f,bounds=(0,5))[0]
    print(f'C={C},k={k},df={df}')
    X_f = pd.DataFrame(X)
    Y_f = X_f.apply(lambda x:Cchi2(x,c=C,k=k,df=df)).values.reshape(n+1,)

    for x in X:
        x1 = x/m.exp(k)
        cdf_a = chi2.cdf(x,df)
        x2 = chi2.ppf(cdf_a+confidence,df)
        if chi2.pdf(x2,df)<chi2.pdf(x1,df):
            a,b = x1,x2
            break
    
    a,b  = a/m.exp(k),b/m.exp(k)
    print(a,b)

    X = np.array(X)
    Y = np.array(Y)
    Y0 = np.array([0]*(n+1))
    fig,ax = plt.subplots()
    ax.plot(X,Y,color = 'hotpink')
    ax.plot(X,Y_f,color = 'r')
    plt.fill_between(X,Y_f,Y0,where=(a<X)&(X<b),color='dodgerblue', alpha=0.5)
    plt.show()

if __name__=='__main__':
    datas = pd.read_excel('datao.xlsx')
    label = [0,0.005,0.010,0.02,0.04,0.06,0.08,0.1,0.2,0.4,0.6,0.8,1]
    R = 0.98
    i = 1 
    bst = []

    for i in range(13):
        da = datas.loc[i,0:1]
        da = da.values.tolist()
        model = Opsp(da,R,label)
        model.name(i)
        model.fit()
        #model.fig(savefig=True,showfig=True)
        print(model.result(True))
        bst.append(model.get_best())
        i+=1
    
    confidence = 0.90
    count(bst,R,label,confidence)
