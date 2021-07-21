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

data = pd.read_csv("./项目/数据分析/英国100000二手车分析/vw.csv")
print(data.shape)
data.head()

'''查看记录中是否有空值'''
data.apply(lambda x: sum(x.isnull()))

data.describe()


'''EDA分析'''
'''变速器'''
data["transmission"].value_counts()


sns.countplot(data["transmission"])
plt.show()
'''大众二手车中自动挡的数量为1960，手动挡为9417,半自动为3780,
数据集上的大多数汽车都采用手动变速器，只有少数汽车采用自动变速器和seim自动变速器'''

print(data["model"].value_counts() / len(data))
sns.countplot(y = data["model"])
plt.show()
'''数据集中排名前三的是Golf、Polo和Tiguan，占大众汽车的64%，其他所有汽车占36%'''

'''燃料类型统计'''
data["fuelType"].value_counts()
sns.countplot(data["fuelType"])
plt.show()
'''是汽油的数量为8553，柴油的数量为6372，
这两次类型占了绝大多数，除此之外还又极少量混合动力型'''




'''注册年份统计'''
sns.countplot(y = data["year"])
plt.show()
plt.figure(figsize=(15,5),facecolor='w')
sns.barplot(x = data["year"], y = data["price"])
plt.show()
'''与之前生产的汽车相比，最近生产的汽车（年份=2018年、2019年）的平均售价更高。'''

sns.barplot(x = data["transmission"], y = data["price"])
plt.show()


plt.figure(figsize=(15,10),facecolor='w')
sns.scatterplot(data["mileage"], data["price"], hue = data["year"])
plt.show()
'''有图可知近四年年注册的二手车的价格普遍高于四年前的，
且大部分这些车的里程数都在0~50000英里左右'''


plt.figure(figsize=(15,5),facecolor='w')
sns.scatterplot(data["mileage"], data["price"], hue = data["fuelType"])
plt.show()
'''由此图可知，以柴油为发动机的而二手车普遍价格和里程数
都要高于同类型的以汽油为发动机的二手车'''

'''两两特征图'''
sns.pairplot(data)
plt.show()

'''现在计算一个年龄域，从年份域减去2020，然后去掉年份域'''
data["age_of_car"] = 2020 - data["year"]
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

因为在一次独热编码之后，数据集中有40个变量，
所以我使用sklearn的SelectKBest选项从数据集中选择最好的特征来应用回归。
为此，我在f_regression回归上执行SelectKBest（），将3个变量降维到40个变量，
看看我们在哪里得到最好的分数。
'''

column_names = data_vw_expanded.drop(columns = ['price']).columns

no_of_features = []
r_squared_train = []
r_squared_test = []

for k in range(3, 40, 2):
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

'''
在曲线稳定之前，我们得到了23个变量的0.88分。
因此保持k为23，从数据集中选择23个最佳变量
'''

selector = SelectKBest(f_regression, k = 23)
X_train_transformed = selector.fit_transform(X_train, y_train)
X_test_transformed = selector.transform(X_test)
column_names[selector.get_support()]


def regression_model(model):
    """
    Will fit the regression model passed and will return the regressor object and the score
    """
    regressor = model
    regressor.fit(X_train_transformed, y_train)
    score = regressor.score(X_test_transformed, y_test)
    return regressor, score



model_performance = pd.DataFrame(columns = ["Features", "Model", "Score"])

models_to_evaluate = [LinearRegression(), Ridge(), Lasso(), SVR(), RandomForestRegressor(), MLPRegressor()]

for model in models_to_evaluate:
    regressor, score = regression_model(model)
    model_performance = model_performance.append({"Features": "Linear","Model": model, "Score": score}, ignore_index=True)

model_performance
'''我们得到的最好分数是RandomForestRegressor（），分数为0.9513'''





'''线性回归变量选择的逆向选择
线性回归模型的拟合及模型参数的检验
'''

regressor = sm.OLS(y_train, X_train).fit()
print(regressor.summary())

X_train_dropped = X_train.copy()

while True:
    if max(regressor.pvalues) > 0.05:
        drop_variable = regressor.pvalues[regressor.pvalues == max(regressor.pvalues)]
        print("Dropping " + drop_variable.index[0] + " and running regression again because pvalue is: " + str(drop_variable[0]))
        X_train_dropped = X_train_dropped.drop(columns = [drop_variable.index[0]])
        regressor = sm.OLS(y_train, X_train_dropped).fit()
    else:
        print("All p values less than 0.05")
        break

'''8个变量被删除，因为p值高于我们的α水平0.05。
我们用剩下的变量来拟合模型，见下面的摘要。
我们可以看到，在我们前面的步骤中，使用SKLearn拟合得到了0.87的rèsquare值，
这与我们的rèsquare值0.89相比，线性回归略有改善'''


print(regressor.summary())


'''
多项式特征拟合
我想进一步研究一下数据集，看看多项式变量模型是否在相同的模型上表现更好。
我正在使用PolynomialFeatures（）从数据集中设计多项式特征。
我们有大约820个来自PolynomialFeatures（）的特征，
所以再次使用SelectKBest来查看我们的最佳特征集大小是多少
'''

poly = PolynomialFeatures()
X_train_transformed_poly = poly.fit_transform(X_train)
X_test_transformed_poly = poly.transform(X_test)

print(X_train_transformed_poly.shape)

no_of_features = []
r_squared = []

for k in range(10, 277, 5):
    selector = SelectKBest(f_regression, k = k)
    X_train_transformed = selector.fit_transform(X_train_transformed_poly, y_train)
    regressor = LinearRegression()
    regressor.fit(X_train_transformed, y_train)
    no_of_features.append(k)
    r_squared.append(regressor.score(X_train_transformed, y_train))

sns.lineplot(x = no_of_features, y = r_squared)
plt.show()
'''从上面的图表中我们可以看到，我们在110个特征上达到了0.93分。'''

selector = SelectKBest(f_regression, k = 110)
X_train_transformed = selector.fit_transform(X_train_transformed_poly, y_train)
X_test_transformed = selector.transform(X_test_transformed_poly)

models_to_evaluate = [LinearRegression(), Ridge(), Lasso(), SVR(), RandomForestRegressor(), MLPRegressor()]

for model in models_to_evaluate:
    regressor, score = regression_model(model)
    model_performance = model_performance.append({"Features": "Polynomial","Model": model, "Score": score}, ignore_index=True)

model_performance


'''
结论：
得到了随机森林回归的多项式数据的最大r^2得分0.955。
作为下一步，我可以专注于单个特性，
并对每个特性进行一些转换（如日志转换），以使模型的性能更好。
'''