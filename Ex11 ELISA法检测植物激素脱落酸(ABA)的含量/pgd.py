from matplotlib.pylab import plt
import numpy as np
import math as m
from sklearn.linear_model import LinearRegression

class line():
    def __init__(self,blank) -> None:
        self.__standards = [0,5,10,20,40,80]#miug/l
        self.__blank = blank

    def __perf(self,unIAAs):
        self.__pcs = dict()
        xp = dict()
        for color,As in unIAAs.items():
            self.__pcs[color] = [(A-self.__blank-self.__d)/self.__k for A in As]
            xp[color] = (sum(self.__pcs[color])/3,sum(As)/3-self.__blank)
        return xp
    
    def set_data(self,Astd):
        self.__Y= [A-self.__blank for A in Astd]

    def fit(self):
        X = np.array(self.__standards).reshape((-1, 1))
        Y = np.array(self.__Y)

        model_line = LinearRegression()
        model_line.fit(X,Y)
        self.__r = model_line.score(X,Y)
        self.__d = model_line.intercept_
        self.__k = model_line.coef_[0]

    def line_result(self):
        return self.__r,self.__d,self.__k
    
    def pcs_result(self):
        return self.__pcs
    
    def fig(self,unIAAs,title,savefig=False,showfig=True):
        pcs_avg = self.__perf(unIAAs)

        l = 0.0001
        st = self.__standards[0]
        ed = self.__standards[-1]
        X_line = np.linspace(st,ed,int((ed-st)/l),endpoint=True) 
        Y_line = self.__k*X_line+self.__d


        fig,ax = plt.subplots()
        fig.set_figheight(8*0.618)
        fig.set_figwidth(8)

        ax.set_title('A-C R:{:g}'.format(self.__r,2)) #标题
        ax.set_ylabel('A')  #y轴标签
        ax.set_xlabel('C')  #x轴标签
        
        ax.plot(X_line, Y_line, linewidth=1.5,
                label='A = {:.2f} C + {:.2f}'.format(self.__k,self.__d),
                color='blue')

        ax.scatter(self.__standards,self.__Y,
                   label='Standards',
                   color = 'hotpink')
        

        for color,xy in pcs_avg.items():
            ax.scatter(xy[0],xy[1],
                       label='Unknown IAA',
                       color = color)
            plt.legend(loc=0)

            plt.annotate('{:.0f}'.format(xy[0]), xy = (xy[0],xy[1]), xytext = (xy[0],xy[1]-0.2),
                        arrowprops = {'headwidth': 1, # 箭头头部的宽度
                                    'headlength': 1, # 箭头头部的长度
                                    'width': 1, # 箭头尾部的宽度
                                    'color': 'black', # 箭头的颜色
                                    'shrink': 0.1, # 从箭尾到标注文本内容开始两端空隙长度
                                    },
                        family='Times New Roman',  # 标注文本字体
                        fontsize = 12,  # 文本字号
                        color='black',  # 文本颜色
                        ha = 'center' # 水平居中
                        )

        if savefig :
            plt.savefig(title,dpi=720)
        
        if showfig:
            plt.show()

def group(blank,stadAs,unIAAs,title):
    model = line(blank)
    model.set_data(stadAs)
    model.fit()
    r,d,k = model.line_result()
    print('相关系数:{}\n截距:{}\n斜率:{}'.format(r,d,k))
    model.fig(unIAAs,title,savefig=True)
    pcs = model.pcs_result()
    print(pcs)

if __name__ == '__main__':
    print('第一组')
    group(0.136,
          [0.136,0.237,0.378,0.681,1.259,2.092],
          {'red':[0.358,0.327,0.282],'orange':[0.266,0.325,0.315]},
          'group1'
          )#标准管*5，{植物样品1*3，植物样品2*3}
