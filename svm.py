# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 10:40:25 2023

@author: ankit
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:34:50 2023
@author: ankit
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from osgeo import gdal
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Set the number of CPU cores for parallel processing
cpu_cores = 20

print('Start loading data...')
df = pd.read_csv(r"/wd/users/d21111/UPDATED_HP_LSM/data.csv", index_col=False) 
df.dropna(inplace=True)
print(df.head())
X= df[['b1_Factors','b2_Factors','b3_Factors','b4_Factors','b5_Factors','b6_Factors','b7_Factors','b8_Factors','b9_Factors','b10_Factors','b11_Factors','b12_Factors','b13_Factors']]
y = df['FID']

x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=20,train_size=0.7)



print('Start training the model...')
# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


  # Create an SVM classifier
svm = SVC(probability=True)
param_grid = {
    'C': [ 0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.5, 0.8, 1, 2, 5],
    'kernel': ['sigmoid'],
    'gamma': [ 0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.5, 0.8, 1, 2, 5]
}
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=3, n_jobs=cpu_cores)
grid_search.fit(x_train, y_train)

# Get the best SVM model from hyperparameter tuning
best_svm = grid_search.best_estimator_
print('Best SVM Model:', best_svm)

# Fit the best SVM model
best_svm.fit(x_train, y_train)
print('SVM Model created!\n')

# Calculate and print the F1-score on the training set
pred_train = best_svm.predict(x_train)
print('Training Set F1-Score =>', f1_score(y_train, pred_train).round(3), '\n')

# Calculate and print the F1-score on the test set
pred_test = best_svm.predict(x_test)
print('Testing Set F1-Score =>', f1_score(y_test, pred_test).round(3), '\n')
print('Start loading raster arrays...')

print('Start loading raster arrays...')
rasternames = [r"/wd/users/d21111/UPDATED_HP_LSM/Factors_13_Mapping.tif"]
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
v8val = v1.GetRasterBand(8).ReadAsArray().flatten() #LAND
v9val = v1.GetRasterBand(9).ReadAsArray().flatten() #LAND
v10val = v1.GetRasterBand(10).ReadAsArray().flatten() 
v11val = v1.GetRasterBand(11).ReadAsArray().flatten()
v12val = v1.GetRasterBand(12).ReadAsArray().flatten()
v13val=v1.GetRasterBand(13).ReadAsArray().flatten()

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
v8val[v8val==0] = 0
v9val[v9val==0] =0
v10val[v10val==0] =0
v11val[v11val==0] =0
v12val[v12val==0] =0
v13val[v13val==0] =0







DATA = np.stack((v1val.flatten(),
                 v2val.flatten(),
                 v3val.flatten(),v4val.flatten(),v5val.flatten(),v6val.flatten(),v7val.flatten(),v8val.flatten(),v9val.flatten(),v10val.flatten(),v11val.flatten(),v12val.flatten(),v13val.flatten()),
                 axis=1)



print('Data has been loaded!\n')

print('Start prediction...')
# Standardize the raster data
DATA = scaler.transform(DATA)



result = best_svm.predict(DATA)


resultproba = best_svm.predict_proba(DATA)
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
pca1 = driver.Create(r"/wd/users/d21111/UPDATED_HP_LSM/Predict/LSM_svm"+ ".tif", col, rows, 1, gdal.GDT_Float32)

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
