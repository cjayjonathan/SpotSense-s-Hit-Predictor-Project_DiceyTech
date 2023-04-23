#!/usr/bin/env python
# coding: utf-8

# As a data analyst at SpotSense, I am working on a project to develop a machine learning model that can accurately predict the popularity of a track on Spotify based on its audio features. By leveraging valuable data on users' listening habits, such as songs, artists, genres, and audio features, we aim to build a product that provides our clients in the music industry with data-driven insights for their music rollout.
# 
# At SpotSense, we specialize in using Spotify data to develop predictive models that can provide data products to various stakeholders in the music industry. Our team has extensive experience in machine learning and data analysis, making us confident in our ability to create a model that accurately predicts a song’s popularity on Spotify.
# 
# In addition to our primary goal, we are taking on the bonus challenge of building a second model that predicts the genre of a track based on its audio features. This will offer even more valuable insights for our clients and enable them to make more informed decisions about their music strategy.
# 
# Through this project, we hope to uncover important insights about the audio features that determine a user's preference for a given song on Spotify and how they vary across different genres. We are excited to explore different algorithm and model architectures, evaluate the performance of our machine learning model, and ensure that it is free of any bias or ethical concerns.
# 
# In this project, I am going to utilize a variety of different regression models to predict the popularity of songs on Spotify based on their audio features. By comparing the performance of these models and evaluating their strengths and weaknesses, I aim to determine the optimal algorithm and model architecture for this task. This approach will enable us to build a machine learning model that accurately predicts the popularity of a track on Spotify and provides valuable insights for our clients in the music industry.

# In[1]:


# Loading the needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import linear_model, tree
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import random
import time
from statsmodels.stats.outliers_influence import variance_inflation_factor    
from joblib import Parallel, delayed
pd.set_option('display.max_columns', None)
sns.set(rc={'figure.figsize':(11.7,8.27)})


# In[2]:


# Loading the Spotify dataset
df = pd.read_csv("C:\\Users\\owner\\Downloads\\SpotSense\\dataset.csv")


# In[3]:


# Familiarizing myself with the dataset
num_duplicates = df.duplicated().sum()
num_null_values = df.isna().sum()

print(f"Number of duplicate rows: {num_duplicates}\nNumber of null values:\n{num_null_values}")


# This code drops a list of columns specified by columns_to_drop from a pandas DataFrame df. The code explicitly specifies the columns parameter to improve readability, and stores the list of columns to drop in a separate variable for clarity. The inplace parameter is set to True to modify the original DataFrame instead of returning a new one.

# In[4]:


columns_to_drop = ['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name']
df.drop(columns=columns_to_drop, axis=1, inplace=True)


# In[5]:


# Getting to know the shape of the data
df.shape


# This code displays the first num_rows_to_show rows of a pandas DataFrame df. The code stores the number of rows to display in a separate variable for clarity, and explicitly specifies it in the head() function call. This approach makes it easier to adjust the number of rows to display if needed.

# In[6]:


num_rows_to_show = 10
df.head(num_rows_to_show)


# This code generates summary statistics of a pandas DataFrame df using the describe() function and stores the result in a new DataFrame df_summary. The modified code then prints the summary statistics to the console.
# 
# By storing the summary statistics in a separate DataFrame, the code allows for further manipulation and analysis of the results if needed. Additionally, explicitly printing the DataFrame with print() provides more control over the output formatting compared to simply calling df.describe().

# In[7]:


df_summary = df.describe()
print(df_summary)


# This code displays a summary of a pandas DataFrame df using the info() function and prints the result to the console.
# 
# The code simply adds the print() function to explicitly output the summary to the console. This approach can help with controlling the output format and making it easier to read, especially if the DataFrame has a large number of columns or rows.

# In[8]:


print(df.info())


# This code calculates the number of duplicate rows and null values in a pandas DataFrame df, stores the counts in separate variables, and prints the results to the console.
# 
# The code separates the calculations from the printing for improved readability and flexibility. Additionally, storing the counts in separate variables makes it easier to reuse the counts in subsequent analysis or reporting.

# In[9]:


num_duplicates = df.duplicated().sum()
num_null_values = df.isna().sum()

print(f"Number of duplicate rows: {num_duplicates}\nNumber of null values:\n{num_null_values}")


# # Pre-Processing Phase
# In the preprocessing step, one of the common tasks is to handle categorical variables in a dataset. Categorical variables are variables that take on a limited number of possible values, such as colors, genres, or categories. Machine learning algorithms often require numeric inputs, so it is necessary to transform categorical variables into numeric variables to use them in these algorithms.
# 
# There are various methods to encode categorical variables, but two common ones are one-hot encoding and label encoding.
# 
# One-hot encoding creates a binary column for each category and indicates whether or not that category is present for a particular row. For example, suppose we have a genre column in our dataset with values rock, pop, and jazz. One-hot encoding would create three new binary columns, genre_rock, genre_pop, and genre_jazz, and set the value to 1 if the corresponding genre is present and 0 otherwise.
# 
# Label encoding assigns a unique integer value to each category. For example, in the same genre column, rock could be assigned 0, pop could be assigned 1, and jazz could be assigned 2. Label encoding is simpler than one-hot encoding but may introduce an implicit ordinal relationship between the categories, which may not always be appropriate.
# 
# In both cases, it is essential to apply the same encoding method to the training and testing datasets consistently. Otherwise, the algorithm may not generalize well to new data. Additionally, it is important to handle missing values in categorical variables, either by imputing them or by treating them as a separate category.

# This function applies label encoding to all categorical columns in a pandas DataFrame df. It first checks whether a column is categorical by checking its datatype. Then, it fills any missing values with the string 'N'. Next, it applies label encoding to the column using the LabelEncoder() class from scikit-learn. Finally, it replaces the original column in the DataFrame with the encoded values.
# 
# The code makes a few changes to improve readability and maintainability. First, it imports LabelEncoder from scikit-learn, which is used to encode categorical variables. Second, it uses df[c] directly in the fit() and transform() methods of LabelEncoder, instead of calling the values attribute separately. Finally, it follows PEP 8 guidelines for function and variable naming by using underscores to separate words in the function name (label_encoder instead of labelencoder).

# In[10]:


def label_encoder(df):
    for c in df.columns:
        if df[c].dtype == 'object': 
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(df[c])
            df[c] = lbl.transform(df[c])
    return df


# This code applies the label_encoder() function to the DataFrame df to encode its categorical columns, and then prints the first few rows of the encoded DataFrame using print(df.head()).
# 
# The code also follows PEP 8 guidelines for function and variable naming by using underscores to separate words in the function name (label_encoder instead of labelencoder).

# In[11]:


df = label_encoder(df)
print(df.head())


# # Exploratory Data Analysis (EDA)
# 
# EDA is a crucial step in the data analysis process where the analyst examines and summarizes the main characteristics of the dataset. The goal of EDA is to gain insights and understanding of the data, to identify patterns and relationships between variables, and to check for anomalies and outliers.
# 
# During EDA, the analyst typically uses various statistical and visualization techniques to explore the data. Some common EDA techniques include:
# 
# Summary statistics: Compute descriptive statistics such as mean, median, standard deviation, and quartiles to gain a general understanding of the data.
# 
# Histograms and density plots: These plots are useful to visualize the distribution of continuous variables in the dataset.
# 
# Box plots: Box plots are useful to visualize the distribution of a variable and identify outliers.
# 
# Scatter plots: Scatter plots are useful to visualize the relationship between two continuous variables in the dataset.
# 
# Heatmaps and correlation matrices: These visualizations are useful to identify patterns and relationships between variables in the dataset.
# 
# Crosstabulations: Crosstabulations are useful to explore the relationship between two categorical variables in the dataset.
# 
# The insights and findings from EDA can guide subsequent analysis and modeling decisions. EDA can also help to identify potential issues or problems in the dataset, such as missing data, outliers, or data entry errors. Addressing these issues during EDA can improve the accuracy and reliability of subsequent analysis and modeling.

# This code creates a pairplot using seaborn library to visualize the relationship between different pairs of continuous variables in the DataFrame df.
# 
# The select_dtypes() method is used to select only the columns with int64 and float64 data types. The palette argument is used to specify the colors for the plot.
# 
# The code also follows PEP 8 guidelines by using a space after the comma in the include argument and using consistent quotation marks.

# In[12]:


sns.pairplot(df.select_dtypes(include=['int64', 'float64']), palette=["#8000ff", "#da8829"])


# This code creates a heatmap using seaborn library to visualize the correlation between different pairs of continuous variables in the DataFrame df.
# 
# The annot argument is used to display the correlation coefficient values on the heatmap. The fmt argument is used to format the values to one decimal place.
# 
# The code uses the 'Blues' color palette, which is a sequential color map with shades of blue that are suitable for visualizing correlations.

# In[13]:


sns.heatmap(df.corr(), annot=True, fmt=".1f", cmap = 'Blues')


# # Modelling and Selection of a Model
# 
# In the modeling part of the machine learning process, it's important to split the dataset into two separate datasets - a training set and a test set. The purpose of splitting the dataset is to have a portion of the data that the model has not seen before, which is important to check how well the model is able to generalize to new data.
# 
# The training set is used to fit the model, while the test set is used to evaluate the model's performance. Typically, the dataset is split into a 70:30 or 80:20 ratio for training and testing, respectively.
# 
# To split the dataset, we can use the train_test_split function from the scikit-learn library. This function randomly splits the data into two sets based on the specified test size.

# The first line of code selects all columns in the dataframe df except for the popularity column, which is the target variable we want to predict. The values attribute converts the selected columns to a NumPy array.
# 
# The second line of code selects the popularity column in the dataframe df and converts it to a NumPy array.
# 
# The third line of code splits the data into a training set and a test set. The test_size parameter specifies the proportion of the dataset that should be included in the test set (in this case, 20%). The random_state parameter sets the random seed for reproducibility. The function returns four arrays: X_train (the feature values for the training set), X_test (the feature values for the test set), y_train (the target values for the training set), and y_test (the target values for the test set).

# In[14]:


X = df.loc[:,df.columns != 'popularity'].values
y = df['popularity'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)


# The models defined are:
# 
# XGBoost Regressor (XGBreg): a gradient boosting framework that uses tree-based learning algorithms.
# Ridge regression (ridge_model): a linear regression model that applies L2 regularization.
# Lasso regression (lasso_model): a linear regression model that applies L1 regularization.
# Decision Tree Regressor (tr): a decision tree-based regression model.
# Bayesian Ridge Regression (bayridge_model): a linear regression model that applies Bayesian methods for regularization.
# Linear Regression (lm): a simple linear regression model.
# Polynomial regression models (prm_list): two polynomial regression models with degrees 2 and 3.
# Polynomial features (poly_list): two sets of polynomial features with degrees 2 and 3 to be used with the polynomial regression models.

# In[15]:


XGBreg = XGBRegressor(scale_pos_weight=1,
                      learning_rate=0.01,  
                      colsample_bytree = 0.3,
                      subsample = 0.8,
                      n_estimators=1000, 
                      reg_alpha = 0.3,
                      max_depth=10, 
                      gamma=1)
ridge_model = linear_model.Ridge(alpha=.5)
lasso_model = linear_model.Lasso(alpha =.1)
tr = tree.DecisionTreeRegressor()
bayridge_model = linear_model.BayesianRidge()
lm = linear_model.LinearRegression()
poly_list = [PolynomialFeatures(degree=i, include_bias=False) for i in range(2,4)]
prm_list = [linear_model.LinearRegression() for i in range(2,4)]


# This code defines a function named model_metrics that takes a machine learning model, test data and actual test labels as inputs. The function then fits the model to the training data, makes predictions on the test data, and calculates the mean squared error and R-squared scores between the predicted and actual labels. Finally, the function returns a dictionary containing the calculated scores and the time taken to fit and predict. Optionally, the function can also take the training data and labels as inputs.

# In[16]:


def model_metrics(model, X_test, y_test, decimals = 5, X_train = X_train, y_train = y_train):
    start  = datetime.now()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = np.round(mean_squared_error(y_test, y_pred),decimals)
    r2 = np.round(r2_score(y_test, y_pred),decimals)
    return {'mean_squared_error': mse, 'R-Squared': r2, 'time': (datetime.now() - start).seconds}


# This function takes a list of models, a list of PolynomialFeatures objects, and the training and test sets, and returns a list of dictionaries containing the mean squared error, R-squared, and the time taken to fit and predict for each model in the list, after transforming the features with the corresponding polynomial object. For each model, the function fits the model to the transformed training set, uses the model to predict the target variable for the transformed test set, and calculates the mean squared error and R-squared between the predicted and actual values. It then records the results in a dictionary and adds the dictionary to the list of metrics for each model. The function then returns this list of metrics.

# In[17]:


def poly_model_metrics(models, poly_list, X_test, y_test, decimals = 5, X_train = X_train, y_train = y_train):
    metrics_list = []
    for i in range(len(models)):
        start = datetime.now()
        poly_features = poly_list[i].fit_transform(X_train)
        models[i].fit(poly_features, y_train)
        y_pred = models[i].predict(poly_list[i].fit_transform(X_test))
        mse = np.round(mean_squared_error(y_test, y_pred), decimals)
        r2 = np.round(r2_score(y_test, y_pred), decimals)
        metrics_list.append(
            {'mean_squared_error': mse, 'R-Squared': r2, 'time': (datetime.now() - start).seconds})
    return metrics_list


# The code is used to evaluate multiple polynomial regression models using the poly_model_metrics function and create a dataframe to store the results of each model.

# In[18]:


poly_models = poly_model_metrics(prm_list, poly_list, X_test, y_test)
poly_results = pd.DataFrame(
    [
        {'mean_squared_error': metrics['mean_squared_error'], 'R-Squared': metrics['R-Squared'], 'time': metrics['time']} 
        for metrics in poly_models
    ],
    index = [f'PolynomiyalRegression_{i+2}_degrees' for i in range(len(poly_models))]) \
.reset_index() \
.rename(columns = {'index': 'model'})


# 

# In[19]:


results = pd.DataFrame(
    [
        model_metrics(XGBreg, X_test, y_test),
        model_metrics(ridge_model, X_test, y_test),
        model_metrics(lasso_model, X_test, y_test),
        model_metrics(tr, X_test, y_test),
        model_metrics(bayridge_model, X_test, y_test),
        model_metrics(lm, X_test, y_test)
    ], 
    index = ['XGBRegressor', 'Ridge', 'Lasso', 'DecisionTreeRegressor', 'BayesianRidge', 'LinearRegression']) \
.reset_index() \
.rename(columns={'index':'model'})


# This code appends the rows from poly_results DataFrame to results DataFrame.
# 
# Assuming both DataFrames have the same column names and structure, the append method concatenates the rows of the poly_results DataFrame to the bottom of the results DataFrame. The resulting DataFrame will have all the rows from both DataFrames combined.

# In[22]:


results = pd.concat([results, poly_results], axis=0, ignore_index=True)


# # Model Comparison Phase
# 
# In the model comparison phase, we evaluate the performance of different models that we built during the modeling phase. The goal of this phase is to select the best model for our problem.
# 
# To compare the models, we use the metrics calculated during the modeling phase, such as mean squared error and R-squared. We can visualize the results using tables or charts to easily compare the performance of the models.
# 
# The model comparison phase also involves analyzing the strengths and weaknesses of each model. We may consider factors such as model complexity, interpretability, and computational efficiency in addition to the performance metrics.
# 
# Based on the evaluation results and considerations, we can select the best model for our problem. It is important to note that the best model may not necessarily be the one with the best performance on the test dataset, but the one that balances the performance and other factors such as interpretability and ease of implementation.

# This code sorts the dataframe results in place based on the specified columns, in descending order of mean_squared_error and then in ascending order of time. If there are ties in mean_squared_error, it sorts the tied rows in descending order of R-Squared.
# 
# The results table shows the mean squared error (MSE), R-squared (R2), and time taken by each model to fit and predict the test data.
# 
# The table is sorted in ascending order by the mean squared error (MSE), which is a measure of how well the model fits the data. A lower MSE indicates a better fit.
# 
# The XGBRegressor model has the lowest MSE, the highest R2 score, and took the longest time to run. This model is the best-performing model in terms of both accuracy and speed.
# 
# The Polynomial Regression model with 2 degrees and 3 degrees come in the next places, followed by the DecisionTreeRegressor, LinearRegression, Ridge, BayesianRidge, and Lasso models.

# In[24]:


results.sort_values(by=['mean_squared_error', 'R-Squared', 'time'], ascending=[True, False, True], inplace=True)
print(results.head(10))


# In[25]:


#  Filtering the rows where the 'R-Squared' column is greater than or equal to 0
results = results[results['R-Squared'] >= 0]


# # Visualizing Performance of Different Models

# This code generates a bar plot that visualizes the performance of different models based on their mean squared error (MSE). The results DataFrame is first sorted in ascending order based on the MSE. Then, a bar plot is created using the Seaborn library where the x-axis represents the model names and the y-axis represents their corresponding MSE values. Finally, a title is added to the plot. The plt.xticks(rotation=45) command rotates the x-axis labels by 45 degrees for better readability if the model names are too long.

# In[27]:


# sort the results by mean_squared_error in ascending order
results = results.sort_values('mean_squared_error', ascending = True)

# create the barplot
plt.figure(figsize=(12,6))
sns.barplot(x=results['model'], y=results['mean_squared_error'])
plt.title('Model Performance based on Mean Squared Error', fontsize=16)

# set labels and ticks
plt.xlabel('Model', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# add grid lines
plt.grid(axis='y', linestyle='--')

# add text annotations
for i, row in results.iterrows():
    plt.text(i, row['mean_squared_error']+20, f'{row["mean_squared_error"]:.1f}', ha='center', fontsize=12)

plt.show()


# In[ ]:





# In[28]:


# sort the results by R-Squared in descending order
results = results.sort_values('R-Squared', ascending = False)

# set plot style and figure size
sns.set_style('darkgrid')
plt.figure(figsize=(10, 6))

# create bar plot
ax = sns.barplot(x=results['model'], y=results['R-Squared'], palette='Blues_r')
ax.set(xlabel='Model', ylabel='R-Squared', title='Model Performance based on R-Squared')

# add values to bars
for i, v in enumerate(results['R-Squared'].round(3)):
    ax.text(i, v+.01, str(v), fontsize=10, ha='center')

# rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# display plot
plt.show()


# This code uses the trained XGBreg model to make predictions on the test set X_test and stores the predicted values in y_pred. The predict() function takes the test data as input and returns predicted values for the target variable based on the model's learned parameters.

# In[29]:


y_pred = XGBreg.predict(X_test)


# This code will display a scatter plot with a red regression line and an annotation in the top-right corner indicating the R-squared value. The plot will have a white grid background and a default matplotlib style.
# 
# R-squared (R²) is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. In other words, R-squared tells you how well the data fits the regression model.
# 
# The value of R-squared ranges from 0 to 1. A higher value of R-squared indicates a better fit of the regression model to the data, meaning that the independent variables explain a larger proportion of the variance in the dependent variable. However, it is important to note that a high R-squared does not necessarily mean that the regression model is a good model or that the independent variables are good predictors of the dependent variable.

# In[30]:


from sklearn.metrics import r2_score

y_pred = XGBreg.predict(X_test)
r_squared = r2_score(y_test, y_pred)

sns.set_style("whitegrid")
fig, ax = plt.subplots()
ax = sns.regplot(x=y_test, y=y_pred, line_kws={'color': 'red'})
ax.set(xlabel='Actual test values', ylabel='Predicted values by XGBoost model')
ax.text(0.8, 0.1, f'R-Squared = {r_squared:.2f}', ha='center', va='center', transform=ax.transAxes)
plt.show()


# This code creates a histogram plot of the residuals with a kernel density estimate line and properly labeled axes and title. The figsize parameter sets the size of the plot, and the fontsize parameters adjust the size of the title and axis labels for better readability.

# In[31]:


residuals = y_pred - y_test

fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(residuals, kde=True, ax=ax)
ax.set_title('Residual Distribution', fontsize=16)
ax.set_xlabel('Residuals', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
plt.show()


# # Variable Selection / Feature Importance
# In machine learning, feature importance refers to determining the relevance of each feature or input variable in a predictive model to predict the target variable. It helps us to identify which features are contributing the most in determining the target variable, and which ones can be ignored or removed to increase the efficiency of the model.
# 
# In this particular context, the feature importance analysis is being performed to identify which features in the dataset are most responsible for determining the popularity of a track. This information can be used to refine the dataset or to inform future data collection efforts. By understanding the relative importance of different features, we can focus our efforts on collecting the most relevant data and improve the accuracy of our predictive models.

# In[32]:


# Fitting XGBoost Model to training data
xgb_model = XGBreg.fit(X_train, y_train)


# In[33]:


xgb_feature_importances = xgb_model.feature_importances_


# This code will create a variable top_ftrs that contains the names of the N most important features in descending order.

# In[40]:


feature_importances_df = pd.DataFrame(xgb_feature_importances, index = df.loc[:,df.columns != 'popularity'].columns) .reset_index() .rename(columns = {0: 'importance', 'index': 'feature'}) .sort_values('importance', ascending = False)


# This will create a horizontal bar plot with the feature names on the y-axis and the importance values on the x-axis, sorted in descending order. Each bar will have a label showing the relative importance of the feature, and the plot will have a title and axis labels for clarity. The sns.despine() function is used to remove the spines on the left and bottom of the plot for a cleaner look.

# In[41]:


plt.figure(figsize=(8,6))
sns.barplot(x=feature_importances_df['importance'], y=feature_importances_df['feature'], color='blue')
plt.title('Importance of Features for Predicting Popularity using XGBoost', fontsize=16)
plt.xlabel('Relative Importance', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().invert_yaxis()
for i, v in enumerate(feature_importances_df['importance']):
    plt.text(v + 0.01, i + .15, str(round(v,3)), color='blue', fontsize=12)
sns.despine(left=True, bottom=True)
plt.show()


# Hurrah! The results are out!
# 
# Above are the relative importance scores for each feature in predicting the target variable, popularity, using the XGBoost model. The feature with the highest importance score is "track_genre" with a score of 0.444, which means it plays a significant role in determining the popularity of a song in this model. Other important features include explicitness, acousticness, valence, loudness, tempo, and energy. On the other hand, features like key, mode, time_signature, and duration_ms have relatively low importance scores, indicating that they have less impact on predicting the popularity of a song in this model.
