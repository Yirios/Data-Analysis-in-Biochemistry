from matplotlib.pylab import plt
import numpy as np
import math as m
from sklearn.linear_model import LinearRegression

class line():
    def __init__(self,length) -> None:
        self.__length = length
        self.__standards = [120000,70000,55000,45000,35000,25000,15000]
        self.__lgM = [m.log10(M) for M in self.__standards]

    def __perf(self,nup):
        self.__pM = list()
        lgm,mr = list(),list()
        for mtp in nup:
            mr.append(mtp/self.__length)
            lgm.append(self.__k*(mr[-1])+self.__d)
            self.__pM.append(m.pow(10,lgm[-1]))
        return mr,lgm

    def set_data(self,ds):
        self.__ds = ds
        self.__mrs = [d/self.__length for d in ds]

    def fit(self,i=2):
        X = np.array(self.__mrs[i-1:]).reshape((-1, 1))
        Y = np.array(self.__lgM[i-1:])
        self.__start = i
        model_line = LinearRegression()
        model_line.fit(X,Y)
        self.__r = model_line.score(X,Y)
        self.__d = model_line.intercept_
        self.__k = model_line.coef_[0]

    def line_result(self):
        return self.__r,self.__d,self.__k
    
    def pM_result(self):
        return self.__pM
    
    def fig(self,nup,title,savefig=False,showfig=True):
        X_p ,Y_p = self.__perf(nup)

        i = self.__start -1

        l = 0.001
        st = self.__mrs[i]
        ed = self.__mrs[-1]
        X_line = np.linspace(st,ed,int((ed-st)/l),endpoint=True) 
        Y_line = self.__k*X_line+self.__d


        X_u = self.__mrs[i:]
        Y_u = self.__lgM[i:]
        X_nu = self.__mrs[:i]
        Y_nu = self.__lgM[:i]

        fig,ax = plt.subplots()
        fig.set_figheight(8*0.618)
        fig.set_figwidth(8)

        ax.set_title('log Mr - mR R:{:g}'.format(self.__r,2)) #标题
        ax.set_ylabel('log Mr')  #y轴标签
        ax.set_xlabel('mR')  #x轴标签
        
        ax.plot(X_line, Y_line, linewidth=1.5,
                label='log Mr = {:.2f} mR + {:.2f}'.format(self.__k,self.__d),
                color='blue')

        ax.scatter(X_u,Y_u,
                   label='Standards',
                   color = 'hotpink')
        
        ax.scatter(X_nu,Y_nu,
                   label='Void Standards',
                   color = '#88c999')
        
        ax.scatter(X_p,Y_p,
                label='Unknown protein',
                color = 'red')  
        plt.legend(loc=0)

        for t in range(len(nup)):
            plt.annotate('{:.0f}'.format(self.__pM[t]), xy = (X_p[t],Y_p[t]), xytext = (X_p[t]-0.05,Y_p[t]-0.2),
                        arrowprops = {'headwidth': 3, # 箭头头部的宽度
                                    'headlength': 3, # 箭头头部的长度
                                    'width': 1.5, # 箭头尾部的宽度
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

def md(l,data,nup,title,i=2):
    model = line(l)
    model.set_data(data)
    model.fit(i)
    r,d,k = model.line_result()
    print('相关系数:{}\n截距:{}\n斜率:{}'.format(r,d,k))
    model.fig(nup,title,savefig=True)
    pM = model.pM_result()
    for i in range(len(nup)):
        print('第{0}条带迁移距离为{1:}对应分子量为{2:.0f}'.format(i+1,nup[i],pM[i]))

if __name__ == '__main__':
    print('第一组')
    md(27,[2,3.7,4.7,6.3,8.4,14.1,18.8],[3.4,6.1],'group1_1.png')
    md(27,[2,3.7,4.7,6.3,8.4,14.1,18.8],[3.4,6.1],'group1_2.png',i=1)
    print('第二组')
    md(3.51,[0.33,0.58,0.8,1.09,1.41,2.16,2.64],[0.71],'group2.png')
    print('第三组')
    md(36,[4,7.4,10,13,16,22.5,26.5],[],'group3.png')
    print('第四组')
    md(3.57,[0.38,0.62,0.82,1.05,1.44,2.22,2.66],[0.77],'group4.png')
