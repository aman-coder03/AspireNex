import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))import pandas as pd

titanic_df = pd.read_csv('/kaggle/input/titanic-datasets/titanic.csv')

titanic_df.head(10)
titanic_df.info()
titanic_df.describe()
titanic_df.columns
print(titanic_df.isnull().sum())
titanic_df['Age'].isnull().sum()
print(titanic_df['Sex'].unique())
print(titanic_df.tail())
titanic_df.boxplot()
titanic_df.hist()
survival_rate = titanic_df['Survived'].count()
print(survival_rate)

import matplotlib.pyplot as plt

survival_by_class = titanic_df.groupby('Pclass')['Survived'].count()
survival_by_class.plot(kind='bar',color='green',edgecolor='black')
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()
print(survival_by_class)

plt.hist(titanic_df['Age'],bins=20,color='green',edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

import seaborn as sns

sns.histplot(titanic_df['Age'].dropna(),bins=20,kde=True)
plt.title('Age Distribution')
plt.show()

sns.histplot(x='Age',hue='Survived',data=titanic_df,bins=20,kde=True)
plt.title('Age Distribution by survival status')
plt.show()

numeric_columns = titanic_df.select_dtypes(include=['float64','int64']).columns

correlation_matrix = titanic_df[numeric_columns].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

sns.countplot(x='Sex',data=titanic_df)
plt.title('Gender Distribution')
plt.show()

sns.barplot(x='Sex',y='Survived',data=titanic_df)
plt.title('Survival Rate by Gender')
plt.show()

      
