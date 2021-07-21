import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('./项目/数据分析/英国100000二手车分析/archive/bmw.csv')

data.head()

data.shape
data.dtypes

data.isnull().sum()


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


'''相关性热图'''
plt.figure(figsize=(12,10))
ax = sns.heatmap(data.corr())
plt.show()

fig = plt.figure(figsize = (20,15))
ax = fig.gca()
data.hist(ax=ax)
plt.show()


data = pd.get_dummies(data=data, columns=['model','transmission','fuelType'])

data.shape


'''划分标签和待训练数据'''
X = data.drop('price',axis=1)
y = data['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33)



'''构建岭回归模型并评价'''
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import Ridge

model_1 = Ridge(alpha=1.0)
model_1.fit(X_train, y_train)

model_1.score(X_test,y_test)

y_pred = model_1.predict(X_test)
np.sqrt(MSE(y_test, y_pred))


'''构建梯度提升模型并评价'''
import xgboost as xg
model_2 = xg.XGBRegressor(objective ='reg:linear',
                          n_estimators = 10, seed = 123)

model_2.fit(X_train, y_train)
model_2.score(X_test,y_test)

y_pred = model_2.predict(X_test)
np.sqrt(MSE(y_test, y_pred))



'''构建弹性网络模型并评价'''
from sklearn.linear_model import ElasticNet
model_3 = ElasticNet(random_state=0)
model_3.fit(X_train, y_train)
y_pred = model_3.predict(X_test)
np.sqrt(MSE(y_test, y_pred))




'''构建神经网络模型'''
import tensorflow as tf

model = tf.keras.models.Sequential([
    # 全连接层
    tf.keras.layers.Dense(64, activation='relu',kernel_initializer='normal',input_dim=37),
    tf.keras.layers.Dense(32, activation='relu',kernel_initializer='normal'),
    tf.keras.layers.Dropout(0.2),
    # 输出层
    tf.keras.layers.Dense(1, activation='linear')
])
model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=['mean_absolute_error'])

history = model.fit(X_train,y_train,batch_size=256,epochs=600,
                    validation_split=0.2,validation_freq=1,
                    )
model.summary()


# 显示训练集和验证集的loss曲线
loss = history.history['loss']
val_loss = history.history['val_loss']


plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


y_pred = model.predict(X_test)
np.sqrt(MSE(y_test,y_pred))