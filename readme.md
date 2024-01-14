# Credit Risk Evaluation
## Improving business decisions with machine learning

This repository contains the data wrangling and exploratory analysis of the Home Credit default loans datasets. Also there are two files for models that are build to complete the plan presented below. There is a separate python file for custom made functions that are used in the notebooks.

## Summary

This project main idea is to provide risk evaluation as service for retail banks. After investigating the Home Credit Group dataset we have determined the steps that we will need to take to create this proof of concept product. The idea was to divide the data into cash and revolving loans and then build and train separate models to predict the TARGET variable which is if the loan has a high risk of default or not.
- The first step is to overview, clean and join the datasets and then perform thorough exploratory analysis to gain insight about our target and other feature relations.
- The second step is to build and train those mentioned models.
- The third step is to deploy the final models to google cloud.

## Final models

- The cash loan default classification model has met the requirements that were set. Its performance was acceptable because after some financial calculations it was clear that it would be profitable for the bank to use it. The model was deployed to google cloud and could be accessed [HERE](https://cash-loan-service-ahskxl2vlq-lm.a.run.app/docs)
- The revolving loan default classification model has not met the requirements that were set. Using it would lose money for the bank and not save it.