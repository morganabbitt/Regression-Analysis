# Regression Case Study

### Contents

1. [Goals](#goals)
2. [Data](#data)
3. [Github Workflow](#git)
4. [Regression](#reg)
5. [Conclusions](#conclusion)

The goal of the contest is to predict the sale price of a particular piece of heavy equipment at auction based on its usage, equipment type, and configuration. The data is sourced from auction result postings and includes information on usage and equipment configurations.

<a name="goals"></a>
### Goals
---
In this project we were given a team and equiptment data and we were asked to use different types of regression to predict the most accurate price while minimizing Root Mean Squared Log Error. 

We worked with Linear Regression, Logisitic Regression, and even attempted Regularized Regression to minimize our RMSLE. 

This README will go over some of the challenges we faced, along with our Regression results. 

<a name="data"></a>
### Data
---

The data for this case study are in `./data`. There you will find both training and testing data sets. We trained on the training set and evaluated our final model only on the testing dataset. For model selection and model comparison, we used Cross Fold Validation to evaluate each model and to avoid overfitting. 

We read in the data using pandas `pd.read_csv('data/Train.zip')`. 

This training data had 401,000 rows and 53 features. 27 of the 53 features contained over 300,000 null values. 

The biggest challenges we had was feature selection. The sale date of some of the equiptment was dated all the way back to 1919, which would scew results drascially. Besides feature selection we chose to impute missing values for features that had only a few null values, we used different metrics depending on the feature like the mode or mean. 

This dataset required a lot of preprocessing and data cleaning. 

<a name="git"></a>
### Github Workflow
---

Our Workflow is as follows:

As the team lead on this Regression Project, I started our master repo and my teammates cloned my repository on Github and created their separate branches. 

We all began EDA and feature exploration to come up with the best way to deal with our dataset. 

I monitored each team members progress by utilizing `git fetch <your feature branch>` and through this we were able to maintain a steady pace throughout our project. 

Everytime we completed an atomic piece of work: `git add -p` `git commit -m` `git push <your feature branch>`

After our changes to our separate branches were made, I merged all of our production code from each branch into the master branch. 

<a name="reg"></a>
### Regression
---
The evaluation of our model will be based on Root Mean Squared Log Error.
Which is computed as follows:

![Root Mean Squared Logarithmic Error](images/rmsle.png)

where *p<sub>i</sub>* are the predicted values and *a<sub>i</sub>* are the
target values.

This loss function is sensitive to the *ratio* of predicted values to
the actual values, a prediction of 200 for an actual value of 100 contributes
approximately the same amount to the loss as a prediction of 2000 for an actual
value of 1000.

When learning a predictive model, we would like you to use only *regression*
methods for this case study.  The following techniques are legal

Important Notes
---

1. This data is messy. We initially overestimated our time limits to clean the entire dataset and defaulted to cleaning and utilizing only features that we knew would be predictive in out model.
2. Because of the restriction to linear models, we were forced to consider how to transform continuous predictors in our model.
5. It's possible some columns in the test data will take on values not seen in
the training data. Plan accordingly.
6. Use your intuition to *think about where the strongest signal about a price
is likely to come from*. If you weren't fitting a model, but were asked to use
this data to predict a price what would you do? Can you combine the model with
your intuitive instincts?  This is important because it can be done *without
looking at the data*; thinking about the problem has no risk of overfitting.
8. We remembered that we were evaluated on a loss function that is only sensitive to
the *ratios* of predicted to actual values, so we used different model scoring to observe this ratio. 
