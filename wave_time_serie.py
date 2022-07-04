"import packages"
import pandas as pd
import numpy as np

"read data from data table downloaded from NOAA"

data = pd.read_csv('h2.csv')

print(data)

"clean the inaccurate data"
data = data[data.WVHT != 99 ]

"reset the index"
data.reset_index(inplace = True)

"extract the wave data column"
wh = data["WVHT"]

"Bulid the zero matrixes"
D_0 = np.zeros(len(wh)-3-1)
D_1 = np.zeros(len(wh)-3-1)
D_2 = np.zeros(len(wh)-3-1)
D_plus_1 = np.zeros(len(wh)-3-1)

"build a loop of delays( 3 delay and one step ahead)"
for i in range(3 , len(wh)-1):
    D_0[i-3] = wh[i-0]
    D_1[i-3] = wh[i-1]
    D_2[i-3] = wh[i-2]

    D_plus_1[i-3] = wh[i+1]

D_0.shape
"buid a pandas dataframe of delays"
data_time_series = pd.DataFrame({"D_0":D_0 , "D_1":D_1 , "D_2":D_2, "D_plus_1":D_plus_1})

"import MLP function"
from sklearn.neural_network import MLPRegressor

"Assign X, Y for feature matrix" 
X = data_time_series[['D_0' ,'D_1' , 'D_2']]
Y = data_time_series['D_plus_1']

"define a model"
model = MLPRegressor()

"split the data to train, test , split"

from sklearn.model_selection import train_test_split

x_train  , x_test , y_train, y_test = train_test_split(X,Y , test_size = 0.3)

x_train
y_train

"fit the model to data"
model.fit(x_train , y_train)

"calculate the accuracy of model"
model.score(x_train , y_train)

"visualize the model output and real output on the same graph to find the accuracy"
import matplotlib.pyplot as plt
plt.scatter(model.predict(x_test), y_test)
