import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

wine = pd.read_csv('Wine.csv', header=0)
#print(wine)
features_columns=wine.columns.drop(['Class'])


X, y = wine.iloc[:, 1:].values, wine.iloc[:, 0].values 

#預處理：先將資料轉成一樣的比例尺
scaler = StandardScaler()
scaler.fit(wine)
scaled_X = scaler.transform(wine)

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y,test_size=0.3, random_state=0) 

pca = PCA().fit(scaled_X)

#累積已解釋變量圖
print(features_columns)
#plt.plot(range(len(pca.explained_variance_ratio_)), np.cumsum(pca.explained_variance_ratio_))

pca = PCA(n_components=2)

#主成分二維圖
"""
projected = pca.fit_transform(X)
plt.scatter(projected[:,0], projected[:,1], 
            c=y, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('RdBu', 10))
plt.title("PCA");plt.xlabel('component1');plt.ylabel('component2');plt.colorbar()
"""

score=[]
for i in range(len(features_columns)):
    pca = PCA(n_components=(i+1)) #保留2個主成分，從累積已解釋變量圖來設定
    lr = LogisticRegression() # 創建邏輯迴歸
    X_train_pca = pca.fit_transform(X_train) # 把原始訓練集映射到主成分组成的子空間
    X_test_pca = pca.transform(X_test) 
    lr.fit(X_train_pca, y_train) 
    score.append(lr.score(X_test_pca, y_test))

plt.plot(score)
plt.title("PCA")
plt.xlabel("number of component")
plt.ylabel("accuracy score")


