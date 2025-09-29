import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
df=pd.read_csv(r"C:\Users\IM_HOME\Desktop\intern\stockmarket.csv")
print(df)

#checking the missing values in the column
print(df.isnull().sum())

#Visualizing the missing values through MatrixPlot
import missingno as ms
ms.matrix(df)
plt.title("Missing Values matrix")
plt.show()

#Handling missing values through mean and median
df['Open']=df['Open'].fillna(df['Open'].median())
print("\nValues after replacing with median:\n",df['Open'])

df['Close']=df['Close'].fillna(df['Close'].mean())
print("\nValues after replacing with mean:\n",df['Close'])

#check for missing after replacing
print("\ncolumns afer replacing missing values:\n",df.isnull().sum())

#visualize Outliers through scatterplot
plt.scatter(df.index, df['Volume'], color='red')
plt.title("Scatter Plot of Volume to Detect Outliers")
plt.xlabel("Index")
plt.ylabel("Volume")
plt.show()

#Detect the Outliers
upper_limit = df['Volume'].quantile(0.99)
lower_limit = df['Volume'].quantile(0.01)

outliers = (df['Volume'] < lower_limit) | (df['Volume'] > upper_limit)
print("Number of outliers:", outliers.sum())

#log transformation
import numpy as np
df['Volume_log'] = np.log1p(df['Volume'])  

plt.scatter(df.index, df['Volume_log'], color='blue', alpha=0.6)
plt.title("Volume after Outlier Imputation")
plt.xlabel("Index")
plt.ylabel("Volume")
plt.show()

#create nextclose
df['Next_Close'] = df['Close'].shift(-1) # next day’s closing price
df.dropna(subset=['Next_Close'], inplace=True)


#create target and see the changes with random rows
df['Label'] = (df['Next_Close'] > df['Close']).astype(int)
df[['Close', 'Next_Close', 'Label']].sample(10)

#select features and target
x = df[['Open','High','Low','Close','Volume_log']]
y=df['Label']

#Normalize/scale features using StandardScaler
from sklearn.preprocessing import StandardScaler
sca=StandardScaler()
x_sca=sca.fit_transform(x)