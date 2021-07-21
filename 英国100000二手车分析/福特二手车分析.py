import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max.columns', None)

data = pd.read_csv("./项目/数据分析/英国100000二手车分析/archive/ford.csv")
print(data.shape)
data.head()

'''查看记录中是否有空值'''
data.apply(lambda x: sum(x.isnull()))
'''数据很干净，没有空值'''
data.describe()


'''EDA分析'''
'''变速器'''
data["transmission"].value_counts()


sns.countplot(data["transmission"])
plt.show()
'''福特汽车中手动挡的占了绝大多数,数量为15517'''


'''汽车类型统计'''
print(data["model"].value_counts() / len(data))
sns.countplot(y = data["model"])
plt.show()
'''福特旗下各类型汽车数量中,排名前三的是Fiesta、Focus和Kuga，占大众汽车的70以上%'''


'''燃料类型统计'''
data["fuelType"].value_counts()
sns.countplot(data["fuelType"])
plt.show()

'''燃油类型为汽油的数量达到12000左右，是柴油机的近两倍'''


'''注册年份统计'''
sns.countplot(y = data["year"])
plt.show()
'''2013到2019'''


'''售价分析'''
plt.figure(figsize=(15,5),facecolor='w')
sns.barplot(x = data["year"], y = data["price"])
plt.show()
'''与之前生产的汽车相比，最近生产的汽车的平均售价更高。'''


'''驾驶类型分析'''
sns.barplot(x = data["transmission"], y = data["price"])
plt.show()
'''自动挡的福特汽车价格平均要比手动挡的要高一些'''


'''里程数与价格分析'''
plt.figure(figsize=(15,10),facecolor='w')
sns.scatterplot(data["mileage"], data["price"],hue = data["year"])
plt.show()
'''大多数二手福特汽车的价格在30000以下，里程数在100000以下'''


'''发动机分析'''
plt.figure(figsize=(15,5),facecolor='w')
sns.scatterplot(data["mileage"], data["price"], hue = data["fuelType"])
plt.show()
'''以柴油为发动机的二手福特汽车价格普遍高于以汽油为发动机的'''


'''两两特征图'''
sns.pairplot(data)
plt.show()


data['year'].value_counts()
'''现在计算一个年龄域，从年份域减去2021，然后去掉年份域'''
data["age_of_car"] = 2021 - data["year"]
data_vw = data.drop(columns = ["year"])
data_vw.sample(10)

'''将注册年份转换为独热编码'''
data_vw_expanded = pd.get_dummies(data_vw)
data_vw_expanded.head()


std = StandardScaler()
data_vw_expanded_std = std.fit_transform(data_vw_expanded)
data_vw_expanded_std = pd.DataFrame(data_vw_expanded_std, columns = data_vw_expanded.columns)
print(data_vw_expanded_std.shape)
data_vw_expanded_std.head()


X_train, X_test, y_train, y_test = train_test_split(data_vw_expanded_std.drop(columns = ['price']), data_vw_expanded_std[['price']])
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



'''模型构建'''

'''
为模型选择最佳特征

因为在一次独热编码之后，数据集中有36个变量，
所以我使用sklearn的SelectKBest选项从数据集中选择最好的特征来应用回归。
为此，我在f_regression回归上执行SelectKBest（），将37个变量降维到3个变量，
看看我们在哪里能得到最好的评分。
'''

column_names = data_vw_expanded.drop(columns = ['price']).columns

no_of_features = []
r_squared_train = []
r_squared_test = []

for k in range(3, 36, 2):
    selector = SelectKBest(f_regression, k = k)
    X_train_transformed = selector.fit_transform(X_train, y_train)
    X_test_transformed = selector.transform(X_test)
    regressor = LinearRegression()
    regressor.fit(X_train_transformed, y_train)
    no_of_features.append(k)
    r_squared_train.append(regressor.score(X_train_transformed, y_train))
    r_squared_test.append(regressor.score(X_test_transformed, y_test))

sns.lineplot(x = no_of_features, y = r_squared_train, legend = 'full')
sns.lineplot(x = no_of_features, y = r_squared_test, legend = 'full')
plt.show()