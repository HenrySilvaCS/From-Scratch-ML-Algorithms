# # Libraries

# In[403]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import stats
import seaborn as sns
from sklearn.datasets import load_boston #boston dataset
from sklearn.model_selection import train_test_split
from scipy import linalg


# # Data Analysis

# In[404]:


print('Data Analysis:')
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)
print(data.head(5))
print(data.describe())


# Checking for outliers in the data

# In[405]:


for k, v in data.items():
        q1 = v.quantile(0.25)
        q3 = v.quantile(0.75)
        irq = q3 - q1
        v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
        perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
        print("Column %s outliers = %.2f%%" % (k, perc))


# Covariance Matrix

# In[406]:


data = data[~(data['MEDV'] >= 50.0)] #removing MEDV outliers
plt.figure(figsize=(20, 10))
sns.heatmap(data.corr().abs(),  cmap="YlGnBu", annot=True)
plt.show()


# Plotting the features with cov > 0.5 with respect to MEDV(PREDICTION)

# In[407]:


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
x = data.loc[:,column_sels]
Y = data['MEDV']
x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for i, k in enumerate(column_sels):
    sns.regplot(y=Y, x=x[k], ax=axs[i])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()


# Separating the Dataset into a training set and a test set

# In[408]:


X, y = load_boston(return_X_y=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
ones = np.ones((404,1))
ones2 = np.ones((102,1))
X_train = np.hstack((ones,X_train))
X_test = np.hstack((ones2,X_test))
X_train = X_train.reshape((404,14))
X_test = X_test.reshape((102,14))
Y_train = Y_train.reshape((404,1))
Y_test = Y_test.reshape((102,1))


# # Linear Regression With QR Decomposition

# In[409]:


def regressionQR(x,y):
    q, r = np.linalg.qr(x) 
    b = q.T @ y
    theta = linalg.solve_triangular(r,b)
    return theta
theta = regressionQR(X_train,Y_train)


# In[410]:


def calculate_error(x,y,parameters):
    prediction = (x @ parameters).reshape((np.size(x,0)),1)
    errors = np.square(prediction - y)
    return errors


# In[411]:


def MSE(x,t):
    MSE = (sum(x))/t
    return MSE


# # Calculations

# In[412]:


#Calculating MSE for the first regression, i.e., using all parameters
errors = calculate_error(X_test,Y_test,theta)
print("The Mean Squared Error for the first regression is:")
MSE1 = MSE(errors,102)
print(MSE1)


# In[413]:


#Removing features with cov < 0.5
X_train_new = np.delete(X_train,[1,2,4,9,12],1)
X_test_new = np.delete(X_test,[1,2,4,9,12],1)
#Calculating the new theta(theta_new)
theta_new = regressionQR(X_train_new,Y_train)
errors2 = calculate_error(X_test_new,Y_test,theta_new)
print("The Mean Squared Error for the second regression is:")
MSE2 = MSE(errors2,102)
print(MSE2)


# Comparisons

# In[414]:


print("The absolute difference between the two MSE's is:")
print(np.absolute(MSE1-MSE2))



# # Basis Expansions and other models for regression

# First lets try a regular polynomial regression, raising the nth feature to the nth power, according to the following:
# y = b0 + b1x1 + b2(x2^2) + ... + bn(xn^n)
# *In this section, we rewind the data to its original form at the beggining of each segment due to the basis expansions

# In[417]:


X_train = X_train.reshape((404,14))
X_test = X_test.reshape((102,14))
Y_train = Y_train.reshape((404,1))
Y_test = Y_test.reshape((102,1))
for i in range(13):
    X_train[:,i] = (X_train[:,i] **(i)) #raises the ith feature to the ith power
    X_test[:,i] = (X_test[:,i] **(i))
parameters1 = regressionQR(X_train,Y_train)
errors4 = calculate_error(X_test,Y_test,parameters1)
MSE4 = MSE(errors4,102)
print("The Mean Squared error for the polynomial regression is:")
print(MSE4)


# Now lets try the log and square root expansions of the features, according to the following:
# h(X(i)) = sqrt(X(i));
# h(X(i)) = log(X(i))
# 

# In[418]:



X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, y, test_size = 0.2, random_state=5)
ones = np.ones((404,1))
ones2 = np.ones((102,1))
X_train1 = np.hstack((ones,X_train1))
X_test1 = np.hstack((ones2,X_test1))
X_train1 = X_train1.reshape((404,14))
X_test1 = X_test1.reshape((102,14))
Y_train1 = Y_train1.reshape((404,1))
Y_test1 = Y_test1.reshape((102,1))
for i in range(13):
        X_train1[:,i] = np.sqrt(X_train1[:,i])
        X_test1[:,i] = np.sqrt(X_test1[:,i])
parameters2 = regressionQR(X_train1,Y_train1)
errors5 = calculate_error(X_test1,Y_test1,parameters2)
MSE5 = MSE(errors5,102)
print("The Mean Squared error for the sqrt expanded regression is:")
print(MSE5)


# In[419]:



X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X, y, test_size = 0.2, random_state=5)
ones = np.ones((404,1))
ones2 = np.ones((102,1))
X_train2 = np.hstack((ones,X_train2))
X_test2 = np.hstack((ones2,X_test2))
X_train2 = X_train2.reshape((404,14))
X_test2 = X_test2.reshape((102,14))
Y_train2 = Y_train2.reshape((404,1))
Y_test2 = Y_test2.reshape((102,1))
for i in range(13):
    if i == 0 or i == 2 or i == 4: # some features have value equal to zero thus making the log expansion impossible, and so we do not operate on those. We also remove the log expansion on the intercept.
        X_train2[:,i] = X_train2[:,i]
        X_test2[:,i] = X_test2[:,i]
    else:
        X_train2[:,i] = np.log(X_train2[:,i])
        X_test2[:,i] = np.log(X_test2[:,i])
parameters3 = regressionQR(X_train2,Y_train2)
errors6 = calculate_error(X_test2,Y_test2,parameters3)
MSE6 = MSE(errors6,102)
print("The Mean Squared error for the log expanded regression is:")
print(MSE6)


# We see that the polynomial regression was the best fit amognst the other expansions, since it presented an decrease in the MSE of the model.

# Now let's try the ridge regression model for regularization

# In[420]:



X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X, y, test_size = 0.2, random_state=5)
ones = np.ones((404,1))
ones2 = np.ones((102,1))
X_train3 = np.hstack((ones,X_train3))
X_test3 = np.hstack((ones2,X_test3))
X_train3 = X_train3.reshape((404,14))
X_test3 = X_test3.reshape((102,14))
Y_train3 = Y_train3.reshape((404,1))
Y_test3 = Y_test3.reshape((102,1))
def ridge_regression(x,y,const):
    identity = np.eye(np.size(x,1))
    theta = np.linalg.inv(x.T @ x + const * identity) @ x.T @ y 
    return theta 
theta_ridge = ridge_regression(X_train3,Y_train3,0.1)
theta_ridge = theta_ridge.reshape((14,1))
errors_ridge = calculate_error(X_test3,Y_test3,theta_ridge)
MSE_ridge = MSE(errors_ridge,102)
print("The MSE for the ridge regression is:")
print(MSE_ridge)


# We see that the ridge regression does not lower the MSE when using the unexpanded data, now lets try with the polynomial expansion for our model

# In[421]:



theta_ridge = ridge_regression(X_train,Y_train,0.1)
theta_ridge = theta_ridge.reshape((14,1))
errors_ridge = calculate_error(X_test,Y_test,theta_ridge)
MSE_ridge = MSE(errors_ridge,102)
print("The MSE for the ridge regression is:")
print(MSE_ridge)


# With the polynomial expansion, the ridge regression yields a decrease to the MSE when lambda = 0.1, however, this decrease is not so significant(~0.008).

# # 4-Fold Cross-Validation

# Now let's try to implement a 4-Fold Cros-Validation. We first split the training dataset into k = 4 parts, and then calculate the prediction error for each k = 1,2,3,4, using k-1 parts as the new training set. This gives us 4 different estimates for the prediction error. After this we combine them and average over the total number of samples N = 404 of the dataset, giving us the final CV prediction error. We applied 4-Fold CV for the standard regression(linear regression) and for the polynomial regression implemented earlier.

# In[422]:



XT1, XT2, XT3, XT4 = X_train[:101,:].reshape((101,14)), X_train[101:202,:].reshape((101,14)).reshape((101,14)), X_train[202:303,:].reshape((101,14)), X_train[303:,:].reshape((101,14))
YT1, YT2, YT3, YT4 = Y_train[:101].reshape((101,1)), Y_train[101:202].reshape((101,1)), Y_train[202:303].reshape((101,1)), Y_train[303:].reshape((101,1))
X_matrix = np.array((XT1,XT2,XT3,XT4)) # 3D matrix that indexes the partitioned datasets
Y_matrix = np.array((YT1,YT2,YT3,YT4))
error_index = np.zeros((4,1))
CV_index = np.zeros((2,1))
for j in range(4):
    training_setX = np.zeros((1,14))
    training_setY = np.zeros((1,1))
    for w in range(4):
        if w != j :
            training_setX = np.vstack((training_setX,X_matrix[w,:,:]))
            training_setY = np.vstack((training_setY,Y_matrix[w,:,:]))
        else :
            training_setX = training_setX
            training_setY = training_setY
    training_setX = np.delete(training_setX,0,0)
    training_setY = np.delete(training_setY,0,0)
    predictors = regressionQR(training_setX,training_setY)
    error_index[j] = sum(calculate_error(X_matrix[j,:,:],Y_matrix[j,:,:],predictors))
CV_index[0] = (sum(error_index))/404     
for j in range(4):
    training_setX = np.zeros((1,14))
    training_setY = np.zeros((1,1))
    for w in range(4):
        if w != j :
            training_setX = np.vstack((training_setX,X_matrix[w,:,:]))
            training_setY = np.vstack((training_setY,Y_matrix[w,:,:]))
        else :
            training_setX = training_setX
            training_setY = training_setY
    training_setX = np.delete(training_setX,0,0)
    training_setY = np.delete(training_setY,0,0)
    predictors = ridge_regression(training_setX,training_setY,0.1)
    error_index[j] = sum(calculate_error(X_matrix[j,:,:],Y_matrix[j,:,:],predictors))
CV_index[1] = (sum(error_index))/404
print("The CV estimation for the prediction error of the two models is:")
print(CV_index)


# We see that, as expected, the CV estimation for the prediction error of the ridge regression is slightly lower than the prediction for the standard regression, indicating that the overall accuracy of the second model is slightly better.
