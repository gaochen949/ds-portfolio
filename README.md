# Data-Science portfolio
# Project 1: Give Me Some Credit

## Problem Statement
Banks play a crucial role in market economies. Credit scoring algorithms, which make a guess at the probability of default, are the method banks use to determine whether or not a loan should be granted. This project aims to build a model that predicts the likelihood of default using customer financial data.

## Objectives
- Perform EDA to understand the structure, distribution, and quality of the data.
- Build a predictive model to classify the likelihood of default
- Identify key drivers of credit risk
- Provide actionable insights for risk management teams.

## Data Dictionary
[Link to Full Data Dictionary](./data_dictionary.md)  
[Link to Original Dataset](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)

## SageMaker Local Mode/Quickstart
```bash
conda env create -f environment.yml
conda activate sagemaker-local
cd ~/ds-portfolio/projects/GiveMeSomeCredit-local
python run_local.py


