import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# Load the dataset
data = pd.read_excel('cancer_data.xlsx')

# Label Encoding
# Labels: Low= 1 Medium= 2 High= 0 
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

data['Stage'] = encoder.fit_transform(data['Level'])
print(data)

# Creating dependent and independent variable
x=data.drop(['Patient Id','Age','Gender','Level','Stage'],axis=1)
y=data.iloc[:,-1].values

# Splitting data into training and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=3)

# Importing Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier  

# Creating model objects
model_lr = LogisticRegression(max_iter = 1000)
model_svm = SVC(kernel = 'linear',random_state=0)
model_knn= KNeighborsClassifier()  
model_dt = DecisionTreeClassifier()  

# Training the models
model_lr.fit(x_train,y_train)
model_svm.fit(x_train,y_train)
model_knn.fit(x_train, y_train)  
model_dt.fit(x_train, y_train)  

# Save the Logistic Regression model
joblib.dump(model_lr, 'logistic_regression_model.pkl')

# Save the SVM model
joblib.dump(model_svm, 'svm_model.pkl')

# Save the KNN model
joblib.dump(model_knn, 'knn_model.pkl')

# Save the Decision Tree model
joblib.dump(model_dt, 'decision_tree_model.pkl')

