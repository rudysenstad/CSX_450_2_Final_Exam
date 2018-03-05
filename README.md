# CSX_450_2_Final_Exam


## Domain

This problem uses the canonical Ames, Iowa housing data set that is often used in teaching data science. In 2011, Dean De Cock cleaned and modified the data provided by the Ames' Assessor's office. It included housing sales in the city from 2006 to 2010 in an excel sheet with 113 columns/variables.


## Problem Statement

The goal is to predict the sale price based, which can be classified as a regression modeling problem because the target prediction is a continuous value. In our case, we will approach the data set from a different angle by clustering the data.


## Dataset

The Ames Iowa House Prices dataset (available [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data))

- Number of samples: 1,451
- Number of attributes: 80 (79 attributes + 1 target attribute (sale price))
- Expected Dataframe Dimensions: 1451 rows x 80 columns
- Column Data Types: category: 56, int64: 21, float64: 3
- Target: SalePrice, continuous value


## Solution Statement

A solution to the problem is to apply a regression model such as linear model, additive model or regression tree. 


## Benchmark Model

For regression models, a good naive benchmark is to use the mean.


## Performance/Evaluation Metric

Given that this is a regression task, we can measure the accuracy of the model with the R2 score or MSE (mean square error). Our goal would be to aim for the best possible R2 score of 1.0 or MSE value of 0.0.

