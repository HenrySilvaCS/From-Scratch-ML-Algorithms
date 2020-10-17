
import numpy as np 
def probability_k1(row,parameters):
	result = np.exp(row @ parameters)/(1 + np.exp(row @ parameters))
	return result 
def probability_vector(dataset,parameters):
	p = np.zeros((np.size(dataset,0),1))
	for i in range(np.size(dataset,0) - 1):
		p[i] = probability_k1(dataset[i,:],parameters)
	return p 
def weight_matrix(dataset,parameters):
	w = np.eye((np.size(dataset,0)))
	for i in range(np.size(dataset,0) - 1):
		w[i,i] = probability_k1(dataset[i,:],parameters) * (1 - probability_k1(dataset[i,:],parameters))
	return w 
def irls(X,y,n_iter):
	ones = np.ones((np.size(X,0),1))
	X = np.hstack((ones,X))
	scaler = StandardScaler()
	scaler.fit(X)
	X = scaler.transform(X)
	#Newton_step
	theta = np.zeros((np.size(X,1),1))
	for i in range(n_iter):
		z = X @ theta + np.linalg.pinv(weight_matrix(X,theta)) @ (y - probability_vector(X,theta))
		theta = np.linalg.pinv(X.T @ weight_matrix(X,theta) @ X) @ X.T @ weight_matrix(X,theta) @ z 
	return theta
