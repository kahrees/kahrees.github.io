layout: page
title: "Kaggle-Calorie-Burn-Prediction"
permalink: https://kahrees.github.io/CALORIE-BURN


Load Data

#Imports

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')

#Load Data (with saved file)

train = pd.read_csv("train.csv",index_col='id') 
test = pd.read_csv("test.csv",index_col='id') 

#Load Data (With API)
# train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv',index_col='id')
# test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv',index_col='id')

Summaries

#First Six Rows
print(train.head())

#First Six Rows of Test
print(test.head())

'Sex' is the only categorical feature. 'id' has been shifted into proper index and not a column

#Find out how many rows
train.shape

test.shape

Preliminary Cleaning

# Check for missing values

missing = train.isnull().sum()

print("Missing values in each column:")
print (missing)

test.isna().sum()

No missing values in either data frame. Both sets are high quality

# Remove duplicate rows
train = train.drop_duplicates( )
print("DataFrame after removing duplicates: ")
print(train.info())

After removing duplicates, the count of rows drops down by 2,841 (From 750,000 to 747,159)

train.describe()

test.describe()

Now, let's begin
Exploratory Data Analysis

Histogram to visualize distribution of numerical columns.

train_new = train.select_dtypes(exclude='object') #Removes Categorical Columns (in this case, 'Sex')

#for loop that plots a new histogram for each column left
for column in train_new:
    fig, ax = plt.subplots(figsize=(18, 5))
    fig = sns.histplot(data=train_new, x=column, bins=50, kde=True)
    plt.show()

Most features are either left or right skewed but some possess more normal distributions, ie more values around the centre
Distribution Plots

# Distribution Plot

sns.displot(train, x="Calories", hue="Sex", multiple="dodge")

sns.displot(train, x="Calories", hue="Sex", kind="kde")

Sex seems to influence Calories

Determine if 'test' and 'train' have similar distributions for each feature. This will help explain if the model fitted on train will work well on test

cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
fig, ax = plt.subplots(4,2,figsize=(8,18))
ax = ax.flatten()
for i,col in enumerate(cols):
    sns.kdeplot(data=train,x=col,ax=ax[i])
    sns.kdeplot(data=test,x=col,color='r',ax=ax[i])
    ax[i].set_yticks([])
    ax[i].set_title(col)

sns.kdeplot(data=train,x='Calories',ax=ax[-1])
ax[-1].set_yticks([])
ax[-1].set_title('Calories')

plt.suptitle('Distributions')
plt.tight_layout()
plt.show()

Correlation Matrix

plt.figure(figsize = (30,20))
sns.heatmap(train.corr(numeric_only = True), annot = True, cmap = 'Reds')
plt.show

Calories is strongly correlated with Duration, Heart Rate, Body Temperature.

Domain Knowledge: Age has an effect on calories lost because muscle definition and retention lessens. The rate of calories burned slows. Essentially, a 20 year old looking to lose 100 calories in a workout will not need as much intensity as a 40 year-old who wishes to do the same.
Violinplots

sns.violinplot(x=train["Age"], inner="quart")

sns.violinplot(x=train["Body_Temp"], inner="quart")

sns.violinplot(x=train["Duration"], inner="quart")

sns.violinplot(x=train["Heart_Rate"], inner="quart")

I started with Multiple Linear Regressions and had a bunch of different results. I then applied back selection but the mean squared error didn't budge significantly even though I combined domain knowledge with back selection, recursive feature selection, test-train split and the correlation matrix. This led to me utiliing the Decision Tree Regressor instead.
Feature Engineering
Binning the Gender

train_bin = pd.get_dummies(train, columns=['Sex'], drop_first=True, dtype=int) #Removes the Sex Column while adding a One Hot Encoded sex column
print(train_bin.head())

y = train['Calories']
X = train_bin.drop('Calories', axis=1)

Ading New Columns

# Creating a column BMI
X["BMI"] = X["Weight"]/(X["Height"]/100)**2

#Create a Column Intensity
X["Intensity"] = X["Duration"] * X["Heart_Rate"]

#Create a Column Weight-Based Intensity
X["Weight-Based Intensity"] = X["Duration"] * X["Heart_Rate"] * X["Weight"]

The cell below is code for a proposed column called Metabolic Rate that incorporates age and gender into the calculations but the memory involved was too large. Output: Unable to allocate 5.70 MiB for an array with shape (747159,) and data type float64

# def BMR_male(weight, height, age):
#     BMR_m = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
#     return BMR_m

# def BMR_female(weight, height, age):
#     BMR_f = 447.593 + (9.247 * weight) + (3.098 * height) - (4.33 * age)
#     return BMR_f

# X["Metabolic Rate"] = [BMR_male(X["Weight"], X["Height"], X["Age"]) if sm == 1 else BMR_female(X["Weight"], X["Height"], X["Age"]) for sm in X["Sex_male"]]

#FE for Test


#Load Test and do the same binning to make things smoother
test_bin = pd.get_dummies(test, columns=['Sex'], drop_first=True, dtype=int)

# Creating a column BMI
test_bin["BMI"] = test_bin["Weight"]/(test_bin["Height"]/100)**2

#Create a Column Intensity
test_bin["Intensity"] = test_bin["Duration"] * test_bin["Heart_Rate"]

#Create a Column Weight-Based Intensity
test_bin["Weight-Based Intensity"] = test_bin["Duration"] * test_bin["Heart_Rate"] * test_bin["Weight"]

#Print
X.head()

test_bin.head()

y.head()

Scaling

While testing different methods, I discovered scaling has a negligible effect om decision tree regressors so I will not be doing that here.

However, when previously using the Multiple Linear Regression Model, Robust Scaling and Min-Max Scaling had the best effects.
The Model

During testing, decision tree regressors gave the lowest (positive) mean absolute error.

#Model Import

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

Fit the Model

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
model = DecisionTreeRegressor(max_leaf_nodes = 5000)
# Fit model
model.fit(train_X, train_y)

Test the Model

#Model Accuracy

predicted_calories = model.predict(X)
mean_absolute_error(y, predicted_calories)

# get predicted prices on validation data
val_predictions = model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

#Cross Validation

k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)


from sklearn.model_selection import cross_val_score 
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

average_r2 = np.mean(scores) 

print(f"R² Score for each fold: {[round(score, 4) for score in scores]}")
print(f"Average R² across {k} folds: {average_r2:.2f}")




scores2 = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_log_error')

average_rmsle = np.mean(scores2) 

print(f"RMSLE Score for each fold: {[round(score, 4) for score in scores2]}")
print(f"Average RMSLE across {k} folds: {average_rmsle:.2f}")

Test With test.csv

Submission

test['Calories'] = model.predict(test_bin)
test['Calories'].to_csv(f'FinalCalorieSubmission.csv')

