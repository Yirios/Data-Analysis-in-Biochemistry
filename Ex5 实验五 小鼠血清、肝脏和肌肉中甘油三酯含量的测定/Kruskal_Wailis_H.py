import pandas as pd
from scipy import stats  # 导入相应模块

data = pd.read_csv('tissues_C_TG.csv')
#print(data)

data1 = data['血清']
data2 = data['肝脏']
data3 = data['肌肉']
print('血清、肝脏、肌肉：',stats.kruskal(data1,data2,data3)[1]) 

print('血清、肝脏',stats.kruskal(data1,data2)[1]) 
print('血清、肌肉：',stats.kruskal(data1,data3)[1]) 
print('肝脏、肌肉：',stats.kruskal(data2,data3)[1]) 