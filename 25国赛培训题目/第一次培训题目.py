import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def func1(x,y):
    """用polyfit函数,最小二乘法,拟合直线"""
    # 使用最小二乘法拟合一条直线
    coefficients = np.polyfit(x, y, 1)  # 1 表示拟合一次多项式（直线）
    poly = np.poly1d(coefficients)  # 生成多项式对象

    # 输出拟合的直线方程
    print(f"拟合的直线方程: y = {coefficients[0]:.4f}x + {coefficients[1]:.4f}")

    y_pred = coefficients[0] * x + coefficients[1]
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    # 计算调整后的R^2
    n = len(y)
    p = 2  # 模型中参数的数量
    adj_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
    #计算AE,APE
    AE=np.abs(y-y_pred)
    APE=AE/y*100
    print(
          f"\nR^2={r2:.4f}, MSE={mse:.4f},调整后的R^2={adj_r2:.4f},平均APE:{APE.mean():.4f}%"
          f"\n模型预测值：\n{y_pred}")
    """# 生成拟合直线的x和y值
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = poly(x_fit)

    # 绘制原始数据点和拟合直线
    plt.scatter(x, y, color='blue', label='数据点')
    plt.plot(x_fit, y_fit, color='red', label='拟合直线')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('最小二乘法拟合直线')
    plt.show()"""

def func2(x,y):
    """statemodels库最小二乘法拟合直线"""
    X=sm.add_constant(x)
    model=sm.OLS(y,X).fit()
    #print(model.summary())

    a,b=model.params
    y_pred = a*x+b
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    # 计算调整后的R^2
    n = len(y)
    p = 2  # 模型中参数的数量
    adj_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
    print(f"\n拟合曲线表达式：y={a:.4f}*x+{b:.4f}"
          f"\nR^2={r2:.4f}, MSE={mse:.4f},调整后的R^2={adj_r2:.4f}"
          f"\n模型预测值：\n{y_pred}")

    #print(model.params)
    """# 绘制原始数据点
    plt.scatter(x, y, color='blue', label='原始数据')
    # 绘制拟合的回归直线
    x_fit = [min(x) - 1, max(x) + 1]  # 扩展x的范围以更好地显示直线
    y_fit = [a + b * x_val for x_val in x_fit]

    plt.plot(x_fit, y_fit, color='red', label='回归直线')

    # 添加图例和标签
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('回归直线拟合')

    # 显示图形
    plt.show()"""


from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

def func3(x,y):
    """用scipy.optimize.curve_fit函数,拟合直线"""
    def model(x,a,b):
        return a*x**b


    popt,pcov=curve_fit(model,x,y)

    # 输出拟合参数，评价指标
    y_pred = model(x, popt[0], popt[1])
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    # 计算调整后的R^2
    n = len(y)
    p = 2  # 模型中参数的数量
    adj_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
    # 计算AE,APE
    AE = np.abs(y - y_pred)
    APE = AE / y * 100

    print(f"\n拟合曲线表达式：y={popt[0]:.4f}*x^{popt[1]:.4f}"
          f"\nR^2={r2:.4f}, MSE={mse:.4f},调整后的R^2={adj_r2:.4f},平均APE:{APE.mean():.4f}%"
          f"\n模型预测值：\n{y_pred}")


    """# 生成拟合曲线的数据
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = model(x_fit, popt[0], popt[1])

    # 绘制原始数据点和拟合直线
    plt.scatter(x, y, label='原始数据')
    plt.plot(x_fit, y_fit, label='拟合直线', color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('原始数据与拟合曲线')
    plt.show()"""


def func3_1(x, y):
    """用scipy.optimize.curve_fit函数,拟合直线"""

    def model(x, k, w,p):
        return k * (x-w) ** p

    popt, pcov = curve_fit(model, x, y)

    # 输出拟合参数，评价指标
    y_pred = model(x, popt[0], popt[1],popt[2])
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    # 计算调整后的R^2
    n = len(y)
    p = 3  # 模型中参数的数量
    adj_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
    # 计算AE,APE
    AE = np.abs(y - y_pred)
    APE = AE / y * 100

    print(f"\n拟合曲线表达式：y={popt[0]:.4f}*(x-{popt[1]:.4f})^{popt[2]:.4f}"
          f"\nR^2={r2:.4f}, MSE={mse:.4f},调整后的R^2={adj_r2:.4f},平均APE:{APE.mean():.4f}%"
          f"\n模型预测值：\n{y_pred}")

    # 生成拟合曲线的数据
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = model(x_fit, popt[0], popt[1],popt[2])


if __name__ == '__main__':

    X = [56, 62, 69, 77, 85, 94, 105]
    Y = [305, 327, 358, 380, 394, 418, 436]
    file_path="第一次练习数据.xlsx"
    #print(df1)

    #np.polyfit
    #func1(X,Y)
    #sm.OLS
    #func2(X,Y)
    #scipy.optimize.curve_fit
    #func3(X,Y)
    #func3_1(X, Y)

    xls=pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    #print(xls.sheet_names)
    #sheet_names = ['男子田径', '女子田径', '男子游泳', '女子游泳']
    for sheet_name in sheet_names:
        print(f"\n项目：{sheet_name}")
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        func1(df.iloc[:, 1], df.iloc[:, 4])
        func3(df.iloc[:, 1], df.iloc[:, 4])
        func3_1(df.iloc[:, 1], df.iloc[:, 4])
