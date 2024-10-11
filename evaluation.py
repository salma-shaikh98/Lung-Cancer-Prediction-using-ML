from sklearn.metrics import confusion_matrix,precision_score,accuracy_score
import joblib
import pandas as pd

# Load your models from the .pkl files
model_lr = joblib.load('logistic_regression_model.pkl')
model_svm = joblib.load('svm_model.pkl')
model_knn = joblib.load('knn_model.pkl')
model_dt = joblib.load('decision_tree_model.pkl')

# Load the dataset
data = pd.read_excel('cancer_data.xlsx')

#Label Encoding
#label Low=1 Medium=2 High=0
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

data['Stage'] = encoder.fit_transform(data['Level'])
print(data)

#creating dependent and independent variable
x=data.drop(['Patient Id','Age','Gender','Level','Stage'],axis=1)
y=data.iloc[:,-1].values

#splitting data into training and testing set
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=3)

#Prediction
y_pred_lr = model_lr.predict(x_test)
y_pred_svm = model_svm.predict(x_test)
y_pred_knn = model_knn.predict(x_test)
y_pred_dt = model_dt.predict(x_test)

# Calculate accuracy,precision and confusion matrix for Logistic Regression model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr, average='macro')
confusion_lr = confusion_matrix(y_test, y_pred_lr)

# Calculate accuracy,precision and confusion matrix for SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='macro')
confusion_svm = confusion_matrix(y_test, y_pred_svm)

# Calculate accuracy,precision and confusion matrix for KNN model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn, average='macro')
confusion_knn = confusion_matrix(y_test, y_pred_knn)

# Calculate accuracy,precision and confusion matrix for Decision Tree model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, average='macro')
confusion_dt = confusion_matrix(y_test, y_pred_dt)

#  Printing the Values
print("Logistic Regression Model:")
print("Accuracy:", accuracy_lr)
print("Precision:", precision_lr)
print("Confusion Matrix:")
print(confusion_lr)

print("\nSVM Model:")
print("Accuracy:", accuracy_svm)
print("Precision:", precision_svm)
print("Confusion Matrix:")
print(confusion_svm)

print("\nKNN Model:")
print("Accuracy:", accuracy_knn)
print("Precision:", precision_knn)
print("Confusion Matrix:")
print(confusion_knn)

print("\nDecision Tree Model:")
print("Accuracy:", accuracy_dt)
print("Precision:", precision_dt)
print("Confusion Matrix:")
print(confusion_dt)


