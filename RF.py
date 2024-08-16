# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 18:10:32 2023

@author: ankit
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:34:50 2023

@author: ankit
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from osgeo import gdal
import numpy as np

#estimators = 100
cpu_cores =6

print('Start loading data...')
df = pd.read_csv(r"/wd/users/d21111/ToHPC/positive.csv", index_col=False) 
df.dropna(inplace=True)
print(df.head())
X= df[['b1_composi','b2_composi','b3_composi','b4_composi','b5_composi','b6_composi','b7_composi']]
y = df['FID']

x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=20,train_size=0.7)
print('Shape of x,y train=>', x_train.shape)
print(x_train.head())
print('Shape of x,y test=>', x_test.shape, '\n')




print('Start training the model...')
clf = RandomForestClassifier(verbose=True, n_jobs=cpu_cores, oob_score=True, warm_start=True)
print('RandomForestClassifier', clf.get_params())
print('Model created!\n')


#################
error_rate = {}

# Range of `n_estimators` values to explore.
min_estimators = 1
max_estimators = 250


for i in range(min_estimators, max_estimators + 1):
    clf.set_params(n_estimators=i)
    clf.fit(x_train, y_train)

    # Record the OOB error for each `n_estimators=i` setting.
    oob_error = 1 - clf.oob_score_
    error_rate[i] = oob_error
    
plt.plot(list(error_rate.keys()), list(error_rate.values()), color='#0c343d')

SMA = pd.Series(list(error_rate.values())).ewm(span=14, adjust=False).mean()
plt.plot(SMA, label='SMA', color='#bf9000', linestyle="--")

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.grid(True)
plt.show()



pred_train = clf.predict(x_train)
print('Training Set F1-Score=>', f1_score(y_train, pred_train).round(3), '\n')

# Evaluating on Test set
pred_test = clf.predict(x_test)
print('Testing Set F1-Score=>', f1_score(y_test, pred_test).round(3), '\n')


print('Calculating...')
pred_test = clf.predict(x_test)



print('Start loading raster arrays...')
rasternames = [r"/wd/users/d21111/ToHPC/Clipped/1.tif"]
v1 = gdal.Open(rasternames[0])

col = v1.RasterXSize
rows = v1.RasterYSize
nelem = col * rows
driver = v1.GetDriver()
print(col, ' x ', rows, ' pixels')
print(v1.RasterCount, ' bands')

v1val = v1.GetRasterBand(1).ReadAsArray().flatten() #curvatura
v2val = v1.GetRasterBand(2).ReadAsArray().flatten() #esposizion
v3val = v1.GetRasterBand(3).ReadAsArray().flatten() #flow_acc
v4val = v1.GetRasterBand(4).ReadAsArray().flatten() #pendenza
v5val = v1.GetRasterBand(5).ReadAsArray().flatten() #quote
v6val = v1.GetRasterBand(6).ReadAsArray().flatten() #soglia_plu
v7val = v1.GetRasterBand(7).ReadAsArray().flatten() #GEO
 #LAND
 #LAND

#v1.GetRasterBand(4).GetNoDataValue()

v1val[v1val==-9999] = 0
v2val[v2val==0] = 0
v3val[v3val==0] = 0
v4val[v4val==-0] = 0
v5val[v5val==0] = 0
v6val[v6val==0] = 0
v7val[v7val==-9999] = 0







DATA = np.stack((v1val.flatten(),
                 v2val.flatten(),
                 v3val.flatten(),v4val.flatten(),v5val.flatten(),v6val.flatten(),v7val.flatten()),
                 axis=1)

print('Start prediction...')
result = clf.predict(DATA)
resultproba = clf.predict_proba(DATA)

print('Prediction ended!\n')

for i in range(len(DATA)):
    if result[i]>0:
        print("%s ==> %s | %s" % (DATA[i], result[i], resultproba[i][1]))

print('Plotting...')
plt.hist(result, bins=30, histtype='bar', ec='black', color='b')

plt.imshow((resultproba[:, 1]).reshape((rows, col))), plt.colorbar()
plt.show()






print('Writing TIF...')
# write_result

pca1 = driver.Create(r"/wd/users/d21111/ToHPC/1.tif"+ ".tif", col, rows, 1, gdal.GDT_Float32)

# Write metadata
pca1.SetGeoTransform(v1.GetGeoTransform())
pca1.SetProjection(v1.GetProjection())

pca1dataarray = (resultproba[:, 1])

pca1dataarray[pca1dataarray == None] = -9999

pca1.GetRasterBand(1).WriteArray(pca1dataarray.reshape(rows, col))
pca1.GetRasterBand(1).SetNoDataValue(-9999)

pca1 = None
del pca1
print('DONE!')
#plt.imshow(v5val)
#
plt.imshow(v5val)