# 输入数据
## A1数据
```r
rL1<-c(0.360,1.080,0.883)
rL2<-c(0.537,0.389,0.899)
rL3<-c(0.654,0.774,1.005)
rL4<-c(0.564,0.550,0.563)

rM1<-c(0.458,0.386,0.440)
rM2<-c(0.565,0.586,0.497)
rM3<-c(0.433,0.336,0.449)
rM4<-c(0.627,0.336,0.583)
```

## A0数据及A0a数据
```r
#
A01<-0.502
    A03<-0.513
A02<-0.511
    A04<-0.513
A01a<-0.502
    A03a<-0.506
A02a<-0.509
    A04a<-0.513
```

# 计算糖原含量
```r
#
L1<-c(0.360*0.555/A01/0.1,1.080*0.555/A01/0.1,0.883*0.555/A01a/0.1)# x*0.555/A01/0.1
L2<-0.555*rL2/A01/0.1 
L3<-c(0.654*0.555/A02/0.1,0.774*0.555/A02/0.1,1.005*0.555/A02a/0.1)#x*0.555/A02/0.1
L4<-0.555*rL4/A02/0.1 

M1<-c(0.458*0.555/A03/0.1,0.386*0.555/A03/0.1,0.440*0.555/A03a/0.1)#x*0.555/A03/0.1
M2<-0.555*rM2/A03/0.1 
M3<-c(0.627*0.555/A04/0.1,0.336*0.555/A04/0.1,0.583*0.555/A04a/0.1)#x*0.555/A04/0.1
M4<-0.555*rM4/A04/0.1 
```
# 数据清洗
## Sample 内数据检查
```r
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
```
## Tissue 内数据清洗
```r
Liver<-c(L1,L2,L3,L4)
Muscle<-c(M1,M2,M3,M4)
GlyC<-c(Liver,Muscle)
ID<-c(replicate(12,'Liver'),replicate(12,'Muscle'))
SampleData<-data.frame(ID,GlyC)
boxplot(GlyC~ID,SampleData,
    horizontal=FALSE)
```
![Alt text](1.png)

# 结果输出
```r
Liver<-c(L1,L2,L3,L4)
Muscle<-c(M1,M2,M3,M4)
GlyC<-c(Liver,Muscle)
ID<-c(replicate(12,'Liver'),replicate(12,'Muscle'))
SampleData<-data.frame(ID,GlyC)
boxplot(GlyC~ID,SampleData,
    main='Glycogen Concentration in Liver and Muscle',
    ylab='Glycogen Concentration(mg/g)',
    xlab='Different Sample Tissue',
    col = "bisque",
    horizontal=FALSE)
```
![Alt text](2.png)
