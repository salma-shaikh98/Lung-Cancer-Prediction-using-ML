import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_excel('cancer_data.xlsx')
data.head(10)

# Label Encoding
# Labels: Low=1 Medium=2 High=0
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

data['Stage'] = encoder.fit_transform(data['Level'])
print(data)

# Exploratory Data Analysis (EDA)
print(data['Stage'].value_counts())

# Pie plot
import matplotlib.pyplot as plt
plt.pie(data['Stage'].value_counts(),labels = ['High','Medium','Low'],autopct = '%0.2f')
plt.show()

# Scatter Plot on Age
plt.scatter(data['Level'], data['Age'], color='blue', alpha=0.5)
plt.title('Co-relation of Age and Level')
plt.show()
# Age doesn't Matter

# Scatter Plot on Gender
plt.scatter(data['Level'], data['Gender'], color='blue', alpha=0.5)
plt.title('Co-relation of Gender and Level')
plt.show()
# Gender doesn't Matter

# scatter Plot on Air Pollution
plt.scatter(data['Level'], data['Air Pollution'], color='blue', alpha=0.5)
plt.title('Co-relation of Air Pollution and Level')
plt.show()
# high means high

# scatter Plot on Dust Allergy
plt.scatter(data['Level'], data['Dust Allergy'], color='blue', alpha=0.5)
plt.title('Co-relation of Dust Allergy and Level')
plt.show()
# high means high

# scatter Plot on OccuPational Hazards
plt.scatter(data['Level'], data['OccuPational Hazards'], color='blue', alpha=0.5)
plt.title('Co-relation of OccuPational Hazards and Level')
plt.show()
# high means high

# scatter Plot on Genetic Risk
plt.scatter(data['Level'], data['Genetic Risk'], color='blue', alpha=0.5)
plt.title('Co-relation of Genetic Risk and Level')
plt.show()
# high means high

# scatter Plot on chronic Lung Disease
plt.scatter(data['Level'], data['chronic Lung Disease'], color='blue', alpha=0.5)
plt.title('Co-relation of chronic Lung Disease and Level')
plt.show()
# high means high

# scatter Plot on Balanced Diet
plt.scatter(data['Level'], data['Balanced Diet'], color='blue', alpha=0.5)
plt.title('Co-relation of Balanced Diet and Level')
plt.show()
# high means high

# scatter Plot on Obesity
plt.scatter(data['Level'], data['Obesity'], color='blue', alpha=0.5)
plt.title('Co-relation of Obesity and Level')
plt.show()
# high means high

# scatter Plot on Smoking
plt.scatter(data['Level'], data['Smoking'], color='blue', alpha=0.5)
plt.title('Co-relation of Smoking and Level')
plt.show()
# high means high

# scatter Plot on Passive Smoker
plt.scatter(data['Level'], data['Passive Smoker'], color='blue', alpha=0.5)
plt.title('Co-relation of Passive Smoker and Level')
plt.show()
# high means high

# scatter Plot on Chest Pain
plt.scatter(data['Level'], data['Chest Pain'], color='blue', alpha=0.5)
plt.title('Co-relation of Chest Pain and Level')
plt.show()
# high means high

# scatter Plot on Coughing of Blood
plt.scatter(data['Level'], data['Coughing of Blood'], color='blue', alpha=0.5)
plt.title('Co-relation of Coughing of Blood and Level')
plt.show()
# high means high

# scatter Plot on Fatigue
plt.scatter(data['Level'], data['Fatigue'], color='blue', alpha=0.5)
plt.title('Co-relation of Fatigue and Level')
plt.show()
# high means high

# scatter Plot on Weight Loss
plt.scatter(data['Level'], data['Weight Loss'], color='blue', alpha=0.5)
plt.title('Co-relation of Weight Loss and Level')
plt.show()
# high means high

# scatter Plot on Shortness of Breath
plt.scatter(data['Level'], data['Shortness of Breath'], color='blue', alpha=0.5)
plt.title('Co-relation of Shortness of Breath and Level')
plt.show()
# high means high

# scatter Plot on Wheezing
plt.scatter(data['Level'], data['Wheezing'], color='blue', alpha=0.5)
plt.title('Co-relation of Wheezing and Level')
plt.show()
# high means high

# scatter Plot on Swallowing Difficulty
plt.scatter(data['Level'], data['Swallowing Difficulty'], color='blue', alpha=0.5)
plt.title('Co-relation of Swallowing Difficulty and Level')
plt.show()
# high means high

# scatter Plot on Clubbing of Finger Nails
plt.scatter(data['Level'], data['Clubbing of Finger Nails'], color='blue', alpha=0.5)
plt.title('Co-relation of Clubbing of Finger Nails and Level')
plt.show()
# high means high

# scatter Plot on Frequent Cold
plt.scatter(data['Level'], data['Frequent Cold'], color='blue', alpha=0.5)
plt.title('Co-relation of Frequent Cold and Level')
plt.show()
# high means high

# scatter Plot on Dry Cough
plt.scatter(data['Level'], data['Dry Cough'], color='blue', alpha=0.5)
plt.title('Co-relation of Dry Cough and Level')
plt.show()
# high means high

# scatter Plot on Snoring
plt.scatter(data['Level'], data['Snoring'], color='blue', alpha=0.5)
plt.title('Co-relation of Snoring and Level')
plt.show()
# high means high