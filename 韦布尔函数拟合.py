import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.metrics import r2_score,mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei']  #  设置中文显示
plt.rcParams['axes.unicode_minus'] = False  #  正常显示负号
plt.rcParams['font.size'] = 16  #  设置字体大小

#  智能电表
total_num=24079
num=[15, 6, 5,9,10,8,11,15,19,20,16,27,24,28,73,73,101,169,211,245,289,274,
    316,371,341,434,424,642,785,1004,1064,1220,1102,943,900,847,717,0.737,676,724,989,721,778,637]

#print(len(ft))

Ft=[]
Rt=[]
s=0
for i in range(len(num)):
    s+=num[i]
    Ft.append(s/total_num)
    Rt.append(1-s/total_num)

#print(Ft)
t=np.arange(19,44+19,1)
t=np.array(t)
#print(t)
Y=np.log(-np.log(Rt))
A=np.vstack([np.log(t),np.ones(len(t))]).T
#print(A)
para,residuals,_,_=np.linalg.lstsq(A,Y,rcond=None)
k,b=para
r=np.exp(-1*b/k)
print(f"k={k},b={b},r={r}")

#  回归直线
plt.figure(figsize=(8,6))
plt.plot(np.log(t),Y,'o',label='(X,Y)',markersize=6)
plt.plot(np.log(t),k*np.log(t)+b,label='拟合直线',linewidth=3)
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

y_pred=k*np.log(t)+b
r2=r2_score(Y,y_pred)
print(f"R^2={r2},残差={residuals[0]}")

Cf,Cp=150,30
def Rfun(T):
    return np.exp(-1*((T/r)**k))

def Cfun(T):
    v, err = integrate.quad(Rfun, 0, T)
    return (Cf*(1-Rfun(T)) + Cp*Rfun(T))/v

Ct=[]
for i in range(1,80+1):
    Ct.append(Cfun(i))

#print(Ct)
print(np.argmin(Ct))

plt.figure(figsize=(8,6))
plt.plot(np.arange(1,80+1),Ct,label='C(t)',linewidth=3)
plt.legend()
plt.xlabel('t')
plt.ylabel('C(t)')
plt.show()