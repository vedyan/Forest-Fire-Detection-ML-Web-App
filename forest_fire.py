#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

import numpy as np # mathimical operations
import pandas as pd # loding and cleaning csv data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle # to convert python obj(here model) into bite stream and store in
# file/database transport data over network
warnings.filterwarnings("ignore")

# saving my Forest_fire data and save it into data(variable) which is
# basically pandas dataframe
data = pd.read_csv("Forest_fire.csv")
data = np.array(data)
# This line of code extracts a subset of the data array.

# The expression data[1:, 1:-1] selects rows starting from the second row (index 1)
# to the last row, and columns starting from the second column to the second-to-last column.
# This operation is typically used to separate the input
# features (independent variables) from the dataset.
X = data[1:, 1:-1] # X will contain the input features (independent variables) of the dataset.
# selects rows starting from the second row (index 1) to the last row,
# and only the last column (-1 index).
# This operation is typically used to extract the target variable (dependent variable)
# from the dataset.
y = data[1:, -1] # y will contain the target variable (dependent variable) of the dataset.
y = y.astype('int') # This line converts the data type of the variable y to integer
X = X.astype('int')
print(X,y)

# split our data in train and test ie half of model is used for training and other half
# will be used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# train_test_split fn take Xandy pred var and target var respectively 0.7 means on 70% of data we train our model and
# rest 30%data we use for testing our model
# and take random rows for fair result, also we return values to xtrain,xtest, ytrain ytest

# now to perform Logistic regression for model name log_reg
log_reg = LogisticRegression()

# Now we try to fit our model
log_reg.fit(X_train, y_train)

inputt=[int(x) for x in "45 32 60".split(' ')] # splits the string "45 32 60" using
# the space (' ')
# ['45', '32', '60']. inputt will contain a list of integers [45, 32, 60]

# creates a NumPy array from the list inputt and assigns it to the list final
# array([45, 32, 60]).
final=[np.array(inputt)]

# log_reg is logistic regression model
b = log_reg.predict_proba(final)
'''predict_proba() method is used to predict the probabilities of the input data belonging 
to each class. In this case, final represents the input data. Since logistic regression is
 a binary classification algorithm, predict_proba() will return the predicted probabilities 
 for both classes (e.g., class 0 and class 1). The variable b will contain these 
 predicted probabilities.'''


pickle.dump(log_reg,open('model.pkl','wb'))
'''pickle.dump() function to serialize (i.e., convert into a byte stream) the 
logistic regression model log_reg and save it to a file named 'model.pkl'. 
The 'wb' mode indicates that the file is opened for writing in binary mode, 
allowing the serialized object to be written to the file.'''

model=pickle.load(open('model.pkl','rb'))
'''pickle.load() function to deserialize (i.e., convert from a byte stream back 
into a Python object) the logistic regression model stored in the file 'model.pkl' and 
assigns it to the variable model. The 'rb' mode indicates that the file is opened for 
reading in binary mode, allowing the serialized object to be read from the file.'''

