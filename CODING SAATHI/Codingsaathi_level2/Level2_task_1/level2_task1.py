# -*- coding: utf-8 -*-
"""Level2_task1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1p0l8k-ApIvqv59z-uFAtdGqvX_vMazBF

#Import necessary libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

"""# Load the terrorism dataset"""

data = pd.read_csv('terrorism_data11.csv')

"""# Data preprocessing tasks"""

# Check the shape of the dataset
print('Shape of dataset:', data.shape)

print(data.info())

# Check the data types of the columns
print('Data types:', data.dtypes)

# Check for missing values
print('Missing values:', data.isnull().sum())

# Check for duplicates
print('Duplicate rows:', data.duplicated().sum())

# Check summary statistics
print('Summary statistics:', data.describe())

"""#Data Visualization"""

# Create box plot of number of terrorist incidents by region
print(data['country_txt'].unique())
sns.boxplot(x='region', y='nkill', data=data)
plt.title('Number of Terrorist Incidents by Region')
plt.xlabel('Region')
plt.ylabel('Number of Incidents')
plt.show()

# Create scatterplot of number of terrorist incidents by year
sns.scatterplot(x=data.iloc[:10000,2], y='nkill', data=data)
plt.title('Number of Terrorist Incidents by Year')
plt.xlabel('Year')
plt.ylabel('Number of Incidents')
plt.show()

"""#Creating a new feature 'total attacks'"""

# Create new feature representing total number of attacks in a country

data['total_attacks'] = data.groupby(['country'])['country'].transform('count')
print(data.total_attacks)

# Create bar chart of total attacks by country
fig, ax = plt.subplots(figsize=(12,6))
data.groupby(['country_txt'])['total_attacks'].sum().sort_values(ascending=False)[:20].plot(kind='bar', ax=ax)

ax.set_title('Top 20 Countries by Total Terrorist Attacks')
ax.set_xlabel('Country')
ax.set_ylabel('Number of Attacks')
plt.show()

# Create choropleth map of number of terrorist incidents by country
fig = px.choropleth(data, locations='country_txt', locationmode='country names', color='nkill',
                    hover_name='country_txt', range_color=[0, 500],
                    title='Number of Terrorist Incidents by Country')
fig.show()

"""#Creating a new attribute 'hotzone'"""

#Identify hot zones of terrorism
hot_zones = data.groupby(['region'])['nkill'].sum().sort_values(ascending=False) # Group the data by region and sum the nkill values to get the total number of deaths by region
print('Hot zones of terrorism:', hot_zones)

# Create a bubble chart of the hot zones
fig = px.scatter(hot_zones, x=hot_zones.index, y=hot_zones.values, size=hot_zones.values,
                 color=hot_zones.index, hover_name=hot_zones.index,
                 title='Hot Zones of Terrorism')
fig.update_layout(xaxis_title='Region', yaxis_title='Total Number of Deaths')
fig.show()

print(data['country_txt'].unique())