#箱线图检查剔除数据 偏离点和波动范围手动剔除
#如何衡量显著性差异？ ?kruskal.test

#数据输入 数据计算
#大小叶
Bleaves<-c(0.646,0.603,0.636)
Sleaves<-c(0.630,0.613,0.573)
Leaves<-c(Bleaves,Sleaves)
AA<-(Leaves-0.0737)/0.0054*1*44/0.1
LabelBS<-c(rep('BigLeaves',3),rep('SmallLeaves',3))
DataAA<-data.frame(LabelBS,AA)
#逆境非逆境
Tplant<-c(0.749,0.637,0.798)
NTplant<-c(0.557,0.557,0.567)
Plant<-c(Tplant,NTplant)
Pro<-(Plant+0.0021)/0.0521/0.1
LabelT<-c(rep('Kept in alcohol',3),rep('Kept in water',3))
DataPro<-data.frame(LabelT,Pro)

#boxplot数据清洗
opar<-par(no.readonly=TRUE)
par(mfrow=c(1,2))
boxplot(AA~LabelBS,DataAA,main='Boxplot of Experiment1')
boxplot(Pro~LabelT,DataPro,main='Boxplot of Experiment2')
par(opar) #没有数据需要排除，手动需要的也没有

#描述性统计数字 统计特征和切片
summary(AA[1:3])
summary(AA[4:6])
summary(Pro[1:3])
summary(Pro[4:6])
summary(Sleaves)
summary(T)

#显著性差异检验
kruskal.test(list(AA[1:3],AA[4:6]))   #无显著差异
kruskal.test(list(Pro[1:3],Pro[4:6])) #有显著差异

#结果输出
opar<-par(no.readonly=TRUE)
par(mfrow=c(1,2))
boxplot(AA~LabelBS,DataAA,
    main='AminoAcid in Plant Leaves',
    xlab='Leave Type',
    ylab='Concentration of Amino Acid(µg/g)',
    pars = list(boxwex = 0.3, staplewex = 0.5, outwex = 0.8),#box,staple宽度，
    col = "bisque",
    horizontal=FALSE)
abline(h=c(mean(AA[1:3])),lty=2,col='red')
abline(h=c(mean(AA[4:6])),lty=2,col='red')

boxplot(Pro~LabelT,DataPro,
    main='Proline in Plant Leaves',
    xlab='Plant Type',
    ylab='Concentration of Proline(µg/g)',
    pars = list(boxwex = 0.3, staplewex = 0.5, outwex = 0.8),#box,staple宽度，
    col = "bisque",
    horizontal=FALSE)
abline(h=c(mean(Pro[1:3])),lty=2,col='red')
abline(h=c(mean(Pro[4:6])),lty=2,col='red')
par(opar) 
