#输入数据
rL1<-c(0.360,1.080,0.883)
rL2<-c(0.537,0.389,0.899)
rL3<-c(0.654,0.774,1.005)
rL4<-c(0.564,0.550,0.563)

rM1<-c(0.458,0.386,0.440)
rM2<-c(0.565,0.586,0.497)
rM3<-c(0.433,0.336,0.449)
rM4<-c(0.627,0.336,0.583)

A01<-0.502
    A03<-0.513
A02<-0.511
    A04<-0.513

A01a<-0.502
    A03a<-0.506
A02a<-0.509
    A04a<-0.513


#计算糖原含量
    L1<-c(0.360*0.555/A01/0.1,1.080*0.555/A01/0.1,0.883*0.555/A01a/0.1)
L2<-0.555*rL2/A01/0.1 # x*0.555/A01/0.1
    L3<-c(0.654*0.555/A02/0.1,0.774*0.555/A02/0.1,1.005*0.555/A02a/0.1)
L4<-0.555*rL4/A02/0.1 #x*0.555/A02/0.1

    M1<-c(0.458*0.555/A03/0.1,0.386*0.555/A03/0.1,0.440*0.555/A03a/0.1)
M2<-0.555*rM2/A03/0.1 #x*0.555/A03/0.1
    M3<-c(0.627*0.555/A04/0.1,0.336*0.555/A04/0.1,0.583*0.555/A04a/0.1)
M4<-0.555*rM4/A04/0.1 #x*0.555/A04/0.1
#数据清洗 
L1<-c(1.080*0.555/A01/0.1,0.883*0.555/A01a/0.1) #x*0.555/A01a/0.1
L2<-c(0.537*0.555/A01/0.1,0.389*0.555/A01/0.1) # x*0.555/A01/0.1
L3<-c(0.654*0.555/A02/0.1,0.774*0.555/A02/0.1)

#boxplot()
opar<-par(no.readonly=TRUE)
par(mfrow=c(2,4))
boxplot(L1,main='Boxplot of L1')
boxplot(L2,main='Boxplot of L2')
boxplot(L3,main='Boxplot of L3')
boxplot(L4,main='Boxplot of L4')

boxplot(M1,main='Boxplot of M1')
boxplot(M2,main='Boxplot of M2')
boxplot(M3,main='Boxplot of M3')
boxplot(M4,main='Boxplot of M4')
par(opar)
# Tissue内数据清洗
Liver<-c(L1,L2,L3)
Muscle<-c(M1,M2,M3,M4)
GlyC<-c(Liver,Muscle)
ID<-c(replicate(6,'Liver'),replicate(12,'Muscle'))
SampleData<-data.frame(ID,GlyC)
boxplot(GlyC~ID,SampleData,
    main='Boxplot of Liver and Muscle',
    ylab='Glycogen Concentration(mg/g)',
    horizontal=FALSE)
#result
boxplot(GlyC~ID,SampleData,
    main='Glycogen Concentration in Liver and Muscle',
    ylab='Glycogen Concentration(mg/g)',
    xlab='Different Sample Tissue',
    pars = list(boxwex = 0.3, staplewex = 0.5, outwex = 0.8),#box,staple宽度，
    col = "bisque",
    horizontal=FALSE)
abline(h=c(7.908,5.436),lty=2,col='red')
library(Hmisc)#添加次要刻度线
minor.tick(ny=2,tick.ratio=1)
minor.tick(ny=4,tick.ratio=0.5)

# 缩小图片尺寸
#Sample剔除数据前和剔除数据后
#Result
#Summary
'''
> summary(Liver)
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
  4.301   6.228   7.755   7.908   9.423  11.940

> summary(Muscle)
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
  3.635   4.664   5.745   5.436   6.315   6.783
'''
summary(Liver)
summary(Muscle)
'''
#绘制子图
attach(mtcars)
opar<-par(no.readonly=TRUE) #??
par(mfrow=c(2,2))
plot(wt,mpg)
plot(wt,disp)
hist(wt)
boxplot(wt)
par(opar)
detach(mtcars)
'''

