# HW3 - AFT Models for Survival Analysis - Short Overview

## Overview
This project applies **Accelerated Failure Time (AFT) models** to predict customer **churn** using the **telco.csv** dataset. The goal is to model the time to churn and understand which factors influence churn risk. Various AFT models were implemented and compared, including:
- **Weibull AFT**
- **Log-Logistic AFT**
- **Log-Normal AFT**
- **Generalized Gamma Regression** - Noted as one of the models available in the scope of AFT models

## Features
- **Data Preprocessing**: Handling missing values, converting categorical columns to numerical codes, and one-hot encoding.
- **Model Fitting**: The project compares several AFT models to predict customer churn.
- **Model Comparison via Plot and Metrics**: The models are evaluated using survival curves and AIC and other metrics.
- **Customer Lifetime Value (CLV) by Segments and "at-risk" explanation**: CLV is calculated for each customer based on the survival functions predicted by the AFT model and "at-risk" explained.
