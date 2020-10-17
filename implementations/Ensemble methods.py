#!/usr/bin/env python
# coding: utf-8

# # Ensemble Methods for Machine Learning

# In this notebook, we'll take a look on how to implement three different ensemble learning algorithms, using some functionatilies from numpy and sklearn. The algorithms are: AdaBoost, Gradient Boosting and Random Forests.

# ## AdaBoost 

# The AdaBoost is an ensemble model for supervised learning first described by Yoav Freund and Robert Schapire. The algorithm consists of a boosting technique, that takes a weak classifier,e.g., decision bump, and improves its performance by applying weights to each sucessive prediction on each iteration. At the end of the iteration loop, the algorithm makes a weighted "vote" on the final prediction. This implementation of the AdaBoost only works for classification problems with $y \in \{-1,1\} $.
# 

# Dependencies

# In[73]:


import numpy as np 
from sklearn.tree  import DecisionTreeClassifier


# ### Implementation 

# In[74]:


def misclassification(y,prediction):
    """
    Calculates the number of misclassifications between the prediction and the output.
    Inputs:
        y: int
            Input a single value of the output variable y.
        prediction: int
            Input a single valut of the predicted output.
    Returns:
        misclassifications: array
            Output 1 if the values do not match and 0 if they do.
    """
    y=y.reshape((-1,1))
    prediction = prediction.reshape((-1,1))
    misclassifications = 1*(y != prediction)
    return misclassifications #returns the number of misclassifications on the prediction

class AdaBoostClassifier:
    """
    AdaBoost algorithm for weak classifiers, that can fit discrete classification problems.
    Methods:
        fit(x,y) -> Performs the boosting algorithm on the training set(x,y).
        predict(x) -> Predict class for X.
        get_tree_weights() -> Returns the weights for each of the n_iter trees generated during the boosting task.
    """
    def __init__(self,n_estimators):
        """
        Initialize self.
        Inputs:
            n_estimators: int
                input the number of trees(stumps) to grow.
        """
        self.n_estimators = n_estimators

    def fit(self,X,y):
        """
        Fits the AdaBooster on a given dataset.
        Inputs:
            X: array
                input the array of input points
            y: array
                input the array of output points, with y E {-1,1}   
        """
        self.input_train = X 
        self.output_train = y 
        alphas = np.zeros((self.n_estimators,1)) 
        predictions_train = np.zeros((len(self.output_train),self.n_estimators)) 
        staged_weights = np.zeros((len(self.output_train),self.n_estimators)) 
        weighted_error = np.zeros((self.n_estimators,1)) 
        stumps = list()

        for i in range(len(self.output_train)):
            staged_weights[i,0] = 1/len(self.output_train) #initialized the weights with value 1/num_samples

        for m in range(self.n_estimators):

            curr_weights = staged_weights[:,m] #current staged weights
            stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            stump = stump.fit(self.input_train,self.output_train,sample_weight=curr_weights) # fits a decision stump with curr_weights as the sample weight

            predictions_train[:,m] = stump.predict(self.input_train) # stores the current predictions
            curr_weights = curr_weights.reshape((-1,1)) 
            weighted_misclassification = curr_weights * misclassification(self.output_train,predictions_train[:,m]) #sets the weighted misclassification as the product between the current weights and the misclassification between the current prediction and the original output
            weighted_error[m] = np.sum(weighted_misclassification)/np.sum(curr_weights) #calculates the weighted error
            alphas[m] = np.log((1 - weighted_error[m])/weighted_error[m]) # calculates the constants alpha
            curr_weights = curr_weights * np.exp(alphas[m] * misclassification(self.output_train,predictions_train[:,m])) #sets the new weights
            curr_weights = curr_weights.reshape((-1,)) # reshapes it to be 1D

            if m + 1 < self.n_estimators: # if this iteration isn't the last one, sets the weights for the next iteration
                staged_weights[:,m+1] = curr_weights
            stumps.append(stump)    

        self.tree_weights = alphas
        self.stumps_list = stumps
        self.weighted_error = weighted_error

    def predict(self,x):
        """
        Makes a prediction based on the weights and classifiers calculated by the Adabooster
        Inputs:
            x: array_like
                input the array of input points
        Returns:
            self.predict: array_like
                outputs the array of predictions
        """   
        indiv_predictions = np.array([self.classifier.predict(x) for self.classifier in self.stumps_list])
        prediction = indiv_predictions.T @ self.tree_weights   #multiplies each tree prediction by the trees calculated weight      
        prediction = np.sign(prediction) #the adaboost final prediction will be the sign of the previous operation.

        return prediction

    def get_tree_weights(self):
        """
        Gives the weights calculated by the AdaBooster
        """
        return self.tree_weights


# ## Gradient Boosting

# The Gradient Boosting is an ensemble model for supervised learning first described by Jerome H. Friedman. It consists of a boosting model, that iteratively calculates the generalized(or partial) residuals at each iteration M, and then fits a new decision tree(regression or classification) targeting these residuals. The algorithm also finds the constant values(gamma) for each split on the decision tree that minimizes the inserted loss function. This model differs from AdaBoost in the sense that it allows for any type of loss function to be used, so long as the function is differentiable. This means that the Gradient Boosting algorithm can fit any type of classification or regression task, which is a big improvement with respect to AdaBoost, since it's most basic implementation can only fit classification problems with target values $y \in \{-1,1\} $. There are versions of AdaBoost that allow for regression tasks, but the model itself is based on the aforementioned assumption about the nature of the output set, so it is far more limited than the Gradient Boosting technique. In this example, we'll be implementing a Gradient Boosted Regressor.

# Dependencies

# In[75]:


import numpy as np 
from sklearn.tree  import DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF


# ### Implementation

# In[76]:


def residuals_func(y,prediction,loss_func):
    """
    Calcultes the generalized(or partial) residuals between the prediction and the output. The residuals are defined to be the negative value of the gradient of the loss function w.r.t. prediction.
    Inputs:
        y: array
            Input the array of outputs.
        prediction: array
            Input the array of predictions.
        loss_func: string
            Input the string that identifies the loss function to use when calculating the residuals; loss_func can be 'multinomial deviance'(multinomial deviance loss), 'entropy'(Cross-entropy loss), 'mse'(Mean Squared Error), 'mae'(Mean Absolute error).
    """
    possible_loss_funcs = ['mse','mae']
    assert loss_func in possible_loss_funcs

    if loss_func == possible_loss_funcs[0]:
        return (y - prediction)
    else:
        return (2*(y - prediction) - 1).reshape((-1,1))

def optimal_gamma(y,prediction,loss_func):
    """
    Calculates the optimal value for the gamma constant based on a certain loss_func.
    Inputs:
        y: array
            input the array of outputs/targets.
        prediction: array
            input the array of predictions.
        loss_func: string
            input the string that identifies the loss function to be used; loss_func can be: 'mse'(Mean Squared Error) or 'mae'(Mean Absolute Error)
    """
    assert loss_func in ["mse","mae"]

    if loss_func == 'mae':
        res = y - prediction
        return np.median(res)  

    elif loss_func == 'mse':
        res = y - prediction
        return np.mean(res)

class GBRegressor:
    """
    GradientBoost algorithm for supervised learning, that can fit regression problems.
    Methods:
        fit(X,y) -> Performs the gradient boosting algorithm on the training set(x,y).
        predict(x) -> Predict value for X.
    
    """
    def __init__(self,n_estimators,loss_func,max_depth=None,random_state=None):
        """
        Inilialize self.
        Inputs:
            n_estimators: int
                input the number of trees to grow.
            loss_func: string
                input the string that identifies the loss function to use when calculating the residuals; loss_func can be  'mse'(Mean Squared Error), 'mae'(Mean Absolute error).
            max_depth: int
                input the maximum depth of the tree; default is set to None.
            random_state: int
                input the random_state to be used on the sklearn DecisionTreeClassifier; default is set to None.
            
                
        """
        possible_params = ['mse', 'mae']
        assert n_estimators > 0
        assert loss_func in possible_params
        if max_depth != None:
            assert max_depth >= 1
            
        self.n_estimators = n_estimators
        self.loss_func = loss_func
        self.random_state = random_state
        self.max_depth = max_depth

    def fit(self,X,y):
        """
        Fits the GradientBooster model.
        Inputs:
            X: array
                input array of input points.
            y: array
                input array of output points.

        """
        self.input_train = X
        self.output_train = y
        self.trained_trees_list = list()
        self.gammas = list()
        self.gamma_0 = optimal_gamma(self.output_train,np.zeros(self.output_train.shape[0]),self.loss_func) #initializes gamma as the optimal value between the output and a vector of zeroes
        raw_pred = np.ones((self.output_train.shape[0])) * self.gamma_0 #makes the initial prediction

        for m in range(self.n_estimators):
            residuals = residuals_func(self.output_train,raw_pred,self.loss_func) # calculates the residuals between the initial prediction and the output
            model = DecisionTreeRegressor(criterion = self.loss_func, random_state = self.random_state,max_depth=self.max_depth) 
            tree = model.fit(self.input_train,residuals) #fits a tree targeting those residuals
            terminal_regions = tree.apply(self.input_train) # gets terminal nodes for the tree
            gamma = np.zeros((len(tree.tree_.children_left))) # generates a gamma vector, with size = number of terminal nodes

            for leaf in np.where(tree.tree_.children_left == TREE_LEAF)[0]: # searches through the tree for terminal nodes(leafs)
                mask = np.where(terminal_regions == leaf) # stores the position of each leaf
                gamma[leaf] = optimal_gamma(self.output_train[mask],raw_pred[mask],self.loss_func) #finds the best gamma for each leaf

            raw_pred += gamma[terminal_regions] # those gammas are then summed to the initial prediction
            self.trained_trees_list.append(tree)
            self.gammas.append(gamma)

                
    def predict(self,x):
        """
        Predicts the value or class of a given group of inputs points based on the trained trees.
        Inputs:
            x: array_like
                input the input point/array to be predicted.
        Returns:
            final_prediction: array_like
                outputs the class/value prediction of the input made by the gradient booster model.
        """
        raw_pred = np.ones(x.shape[0])* self.gamma_0

        for tree,gamma in zip(self.trained_trees_list,self.gammas):
            terminal_regions = tree.apply(x)
            raw_pred += gamma[terminal_regions] #the final prediction of the gradient boosting regressor will be the initial predictions summed with all the optimal gammas found for all leafs

        return raw_pred                        
            


# ## Random Forests

# The Random Forests is an ensemble model for supervised learning first described by Breiman (2004). It uses bagging to build a group of subsets of the initial dataset, and then builds a tree(decision or regression) for each subset, by randomly choosing a group of features in each subset, so that n_features_used < n_features_total. Here I implement two different Random Forest models, for classification and regression. These implementations are built on top of the sklearn and numpy libraries.

# Dependencies

# In[77]:


import numpy as np 
from sklearn.tree  import DecisionTreeClassifier
from sklearn.tree  import DecisionTreeRegressor


# ### Implementation

# #### Random Forests for Classification

# In[78]:



class RandomForestClassifier:
    """
    Class of the Random Forest Classifier Model.
    Methods:
        fit(X,y) -> Performs the random forests algorithm on the training set(x,y).
        predict(x) -> Predict class for X.
    
    """
    def __init__(self,n_estimators,n_classes,max_depth=None,criterion='gini',random_state=None,max_features='sqrt'):
        """
        Initialize self.
        Inputs:
            n_estimators: int
                input the number of trees to grow.
            n_classes: int
                input the number of classes of the classification task.
            max_depth: int
                input the maximum depth of the tree; default is set to None.
            criterion: string
                input string that identifies the criterion to be used when deciding how to split each node. criterion can be: 'gini' or 'entropy' .default is set to 'gini'.
            max_features = string or int/float:
                input string or int/float that identifies the maximum number of features to be used when splitting each decision tree; if string can be: 'sqrt' or 'log2'; if int max_fetures will be the maximum number of features; if float the maximum number of features will be int(max_features * n_features).                   
            random_state: int
                input the random_state to be used on the sklearn DecisionTreeClassifier. default is set to None.
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.random_state = random_state
        self.n_classes = n_classes
        self.max_features = max_features
        self.max_depth = max_depth

        possible_criterion = ['gini','entropy']
        assert self.criterion in possible_criterion 

    def fit(self,X,y):
        """
        Fits the RandomForestClassifier model.
        Inputs:
            X: array
                input array of input points.
            y: array
                input array of output points.
        """
        self.input_train = X
        self.output_train = y.reshape((-1,1))
        self.trained_trees_list = list()

        for i in range(self.n_estimators):
            train_inds = np.random.choice(int(self.input_train.shape[0]/2),int(self.input_train.shape[0]/2),False) #generates the indices for a bootstrap sample with size N/2 and with values that don't repeat(doing this improves the efficiency of the algorithm and does not deprecate performance)
            model_tree = DecisionTreeClassifier(criterion = self.criterion, random_state = self.random_state,max_features = self.max_features,max_depth=self.max_depth)
            model_tree = model_tree.fit(self.input_train[train_inds,:],self.output_train[train_inds,:]) #fits a decision tree with the bootstrap sample
            self.trained_trees_list.append(model_tree)

    def predict(self,x):
        """
        Predicts the class of a given group of inputs points based on the trained trees. 
        Inputs:
            x: array_like
                input the input point/array to be predicted.
        Returns:
            final_prediction: array_like
                outputs the class prediction of the input made by the random forest model.
        """
        indiv_predictions = np.array([self.classifier.predict(x) for self.classifier in self.trained_trees_list]).T
        final_prediction = np.zeros((indiv_predictions.shape[0],))
        counter_vec = np.zeros((self.n_classes,1))

        for i in range(indiv_predictions.shape[0]):
            for j in range(indiv_predictions.shape[1]):
                counter_vec[indiv_predictions[i][j]] += 1
                final_prediction[i] = np.argmax(counter_vec) #the final prediction of the random forest classifier will be a majority vote made w.r.t. to all trees,i.e., for each output point, predict the class that appears the most on the group of all prediction made by the forest
            counter_vec = np.zeros((counter_vec.shape))

        return final_prediction
  
            
        
        
        


# #### Random Forests for Regression

# In[79]:


class RandomForestRegressor:
    """
    Class of the Random Forest Classifier Model.
    Methods:
        fit(X,y) -> Performs the random forests algorithm on the training set(x,y).
        predict(x) -> Predict regression value for X.
    
    """
    def __init__(self,n_estimators,max_features =1/3,max_depth=None,criterion='mse',random_state=None):
        """
        Initialize self.
        Inputs:
            n_estimators: int
                input the number of trees to grow.
            max_depth: int
                input the maximum depth of the tree; default is set to None.
            criterion: string
                input string that identifies the criterion to be used when deciding how to split each node. criterion can be: 'mse', 'friedman_mse' and 'mae' .default is set to 'mse'.
            max_features = string or int/float:
                input string or int/float that identifies the maximum number of features to be used when splitting each decision tree; if string can be: 'sqrt' or 'log2'; if int max_fetures will be the maximum number of features; if float the maximum number of features will be int(max_features * n_features); default is set to 1/3.             
            random_state: int
                input the random_state to be used on the sklearn DecisionTreeClassifier. default is set to None.
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.random_state = random_state
        self.max_features = max_features
        self.max_depth = max_depth
        possible_criterion = ['mse', 'friedman_mse', 'mae']
        assert self.criterion in possible_criterion 

    def fit(self,X,y):
        """
        Fits the RandomForestRegressor model.
        Inputs:
            X: array
                input array of input points.
            y: array
                input array of output points.
        """
        self.input_train = X
        self.output_train = y.reshape((-1,1))
        self.trained_trees_list = list()
        
        for i in range(self.n_estimators):
            train_inds = np.random.choice(self.input_train.shape[0],self.input_train.shape[0],True) #generates the indices for a bootstrap sample with size N aand with repeating values(in the case of regression, there is a slight increase in error when using the technique described in the Classifier code)
            model_tree = DecisionTreeRegressor(criterion = self.criterion, random_state = self.random_state,max_features = self.max_features,max_depth=self.max_depth)
            model_tree = model_tree.fit(self.input_train[train_inds,:],self.output_train[train_inds,:]) #fits a decision tree with the bootstrap sample
            self.trained_trees_list.append(model_tree)

    def predict(self,x):
        """
        Predicts the value of a given group of inputs points based on the trained trees.
        Inputs:
            x: array_like
                input the input point/array to be predicted.
        Returns:
            final_prediction: array_like
                outputs the value prediction of the input made by the random forest model.
        """
        indiv_predictions = np.array([self.classifier.predict(x) for self.classifier in self.trained_trees_list]).T
        final_prediction = np.zeros((indiv_predictions.shape[0],))
        
        for i in range(indiv_predictions.shape[0]):
            final_prediction[i] = np.sum(indiv_predictions[i][:])/indiv_predictions.shape[1] #the final prediction of the Random Forest Regressor will be the average of all predictions made by the group of trees
        
        return final_prediction


