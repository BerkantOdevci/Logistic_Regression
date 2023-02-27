#######################################################################
#   Logistic Regression
#######################################################################
#
#   @Class Name(s): LogisticRegressionProcess
#
#   @Description:   Cat Anaylsis
#
#
#   @Note:  In the data set formed with the image, the result is 1 if there is a cat in the image, 0 otherwise.
#
#   Version 0.0.1:  LogisticRegressionProcessclass
#                   30 KASIM 2022 Monday, 16:30 - Hasan Berkant Ödevci
#
#
#
#   @Author(s): Hasan Berkant Ödevci
#
#   @Mail(s):   berkanttodevci@gmail.com
#
#   30 KASIM 2022 Monday, 16:30 
#
#
########################################################################

# Libraries
try:
    import numpy as np
    import copy
    import matplotlib.pyplot as plt
    import h5py
except ImportError:
    print("Please Check Library")

# Global Variables
train_dataset = h5py.File("D:/Python_Proje/Deep_Learning/Logistic_Regression/Cat_Datasets/train_catvnoncat.h5","r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:])
train_set_y_orig = np.array(train_dataset["train_set_y"][:])

test_dataset = h5py.File("D:/Python_Proje/Deep_Learning/Logistic_Regression/Cat_Datasets/test_catvnoncat.h5","r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:])
test_set_y_orig = np.array(test_dataset["test_set_y"][:])

classes = np.array(test_dataset["list_classes"][:])

# Reshape array
train_set_y_orig = train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
test_set_y_orig = test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))

# Reshape the training and test data sets so that images of size (num_px, num_px, 3) are flattened into single vectors of shape (num_px  ∗  num_px  ∗  3, 1).
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

# Standardize datasets
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

# Calculate y_hat with sigmoid function sigma(wT*x+b)
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

# Initiliaze w and b parameters
def initialize_with_zero(x):
    w = np.zeros([x,1])
    b = 0.0
    return w,b

# Forward and Backward Propagation
def propagate(w,b,x,y):
    m = x.shape[1]

    # Calculate y_hat
    y_hat = sigmoid(np.dot(np.transpose(w),x)+b)

    # Calculate Cost Function
    cost = -(1/m)*np.sum((y*np.log(y_hat))+(1-y)*np.log(1-y_hat))

    # Backward Propagation
    dw = (1/m)*np.dot(x,np.transpose(y_hat-y))
    db = (1/m)*np.sum(y_hat-y)

    # Np.squeeze kullanarak tek boyutlu girdileri ortadan kaldırıyoruz.
    cost = np.squeeze(np.array(cost))

    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

# To update the parameters using gradient descent.
def optimize(w, b, x, y, num_iterations, learning_rate, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):
        grads,cost = propagate(w,b,x,y)

        # Gradient parameters
        dw = grads["dw"]
        db = grads["db"]

        # Update gradient
        w = w - learning_rate*dw
        b = b - learning_rate*db

        # Record the costs
        if (i % 100 == 0):
            costs.append(cost)

            if(print_cost):
                print("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
        "b": b}
    
    grads = {"dw": dw,
        "db": db}

    return params,grads,costs

# To use w and b to predict the labels for a dataset X
def predict(w,b,x):
    m = x.shape[1]

    # Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    y_prediction = np.zeros((1,m))
    
    # W.shape is always equal to x.shape[0]
    w = w.reshape(x.shape[0], 1)
    
    # Compute vector "y_hat" predicting the probabilities of a cat being present in the picture
    y_hat = sigmoid(np.dot(w.T, x) + b)

    for i in range(x.shape[1]):
        if(y_hat[0,i]>0.5):
            y_prediction[0,i] = 1
        else:
            y_prediction[0,i] = 0
    
    return y_prediction

def model(x_train, y_train, x_test, y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zero(x_train.shape[0])

    params,grads,costs = optimize(w, b, x_train, y_train, num_iterations, learning_rate, print_cost)

    w = params["w"]
    b = params["b"]

    y_prediction_train = predict(w,b,x_train)
    y_prediction_test = predict(w,b,x_test)

    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": y_prediction_test, 
         "Y_prediction_train" : y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d    

logistic_regression_model = model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, num_iterations=2000, learning_rate=0.005, print_cost=True)

# Test if it is cat, y = 1 but if not, y = 0    
for i in range(len(test_set_x)):
    plt.imshow(test_set_x[:, i].reshape((64, 64, 3)))
    plt.show()
    print ("y = " + str(test_set_y_orig[0,i]) + ", you predicted that it is a \"" + classes[int(logistic_regression_model['Y_prediction_test'][0,i])].decode("utf-8") +  "\" picture.")
