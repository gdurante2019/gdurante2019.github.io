---
layout: post
title:      "Data Science Toolbox:  Function to Create Top "n" Values"
date:       2020-07-22 20:39:35 -0400
permalink:  predicting_water_well_status_in_tanzania_with_machine_learning_models
---


## Overview
For my Flatiron School Data Science Boot Camp Module 3 (supervised machine learning) project, I selected a dataset containing data on almost 60,000 water supply projects (‘wells’) in Tanzania.  This was a categorical classification problem with 3 imbalanced target classes and a total of 39 features, most of which are categorical, and several of which are colinear.  I used data visualization fairly extensively during the exploration phase, and did a deep dive into several supervised machine learning approaches to predict water well status.  Along the way, I developed some useful functions to create visualizations, prepare data for modeling, running models, and displaying results.  

For this blog installment, I want to describe a function I developed to help tackle the challenge of selecting feature values in a way that reduced noise in the modeling and decreased computing resource intensity—specifically, selecting subsets of feature values to convert to dummy variables for modeling.  But first, a bit of background on the competition and the size and scope of this dataset...

## Background
Tanzania is a developing country that is home to over 57,000,000 people.  The government seeks to provide clean water to its citizens, but a significant percentage of people (~43%, or 25,000,000) do not have access to clean water (1).  Over 75,000 water supply projects have been constructed, ranging from larger wells for hundreds or thousands of people, to shallow wells or hand pumps serving small numbers.  Only about 54% of all water supply projects in the country are functional, while 38% are non-functional (needing replacement) and the remaining ~8% are functional, but in need of repair.  (More information about water infrastructure in Tanzania can be found at the Ministry of Water website (2).)

The government of Tanzania has collaborated with Taarifa, a non-profit organization, to create a database of all of the water supply projects in the country (3).  Data for each water supply project includes information about the project’s geographic location, local water abundance and quality, technical information (e.g., type of well), funder, installer, project administration, and more.  

DrivenData, a social enterprise that works with mission-driven organizations, is hosting a competition to predict water well status using machine learning algorithms (4).  DrivenData provides data to participants in the competition in the form of a .csv file and *very* brief descriptors for each field.  The 'training' dataset provided by DrivenData contains data and functional status for 59,400 water supply projects.  The 'test' dataset provides data, but not functional status, for 14,850 wells.  Participants develop machine learning models and submit their predictions--‘functional’, ‘non-functional’, or ‘functional, needing repair’--for these wells.  DrivenData compares each participant’s predictions against the actual status of the wells and returns an accuracy score for the participant’s prediction dataset.
## Data preparation for modeling
### The problem
*All* of the features I used in this project were categorical (non-ordinal), and so required encoding to be read by the model.  However, because I was using scikit-learn classifiers, label encoding was not appropriate for features.  Thus, it was necessary to convert features to dummy variables.  Considering that some features literally have hundreds or thousands of values each, creating dummy variables for all of these values would result in thousands of dummy variables!  

### A potential solution
To reduce the ‘noise’ of the dataset, I created a function that sorts the top ‘n’ feature values by number of projects, then groups the remaining values (total-minus-n feature values) into a category called ‘other’.  Instead of converting hundreds / thousands of feature values into dummy variables, we now have n+1 dummy variables for this feature.  

For example, the feature 'funder' has 1,898 unique values (names of funders).  Converting all of the values of this feature into dummy variables will result in 1,898 dummy variables, one for each name in the 'funder' feature!  However, if we select n=50, the function will keep the top 50 names (sorted from greatest number of projects associated with each name to the least) and replace the remaining 1,848 values with ‘other’, resulting in 51 dummy variables—a big difference from 1,898 dummy variables for one feature alone!

This approach is legitimate because for these features with thousands of values, there is an almost exponential drop off in the number of projects per unique feature name as one looks further down the list of, say, funders, sorted by number of projects per funder.  The bottom half of the list--over 900 funders--have financed anywhere from one to 10 projects.  While there almost certainly are some differences among these small-scale funders, the fact that each one is responsible for just a few projects suggests that any one particular small-scale funder won't have a significant influence on the correlation between functional status and funder.  Actually, it is more likely that small-scale funders actually behave fairly similar to each other, making it more appropriate to lump them together in one group, as one dummy variable.  

While the value one selects for 'n' is arbitrary, running different values of 'n' through the model can yield clues about  where the ‘sweet spot’ lies for the number of feature values to include.  I've noticed that setting ‘n’ in the range of 100 to 150 seems to give good results. 

### How does it work?
Here are the steps that the function performs to produce a subset of dummy variables for a given feature:

1. The function takes in the dataframe, an ‘n’ value, and a list of the feature(s) in question
2. For each feature, the function performs the following tasks:
    * Duplicates the feature column in question
    * Sorts the values (e.g., installers) from greatest number of projects to the least
    * Converts that feature column to Categorical in pandas, keeping the dataframe sorted by number of projects for that feature for the remaining steps
    * Creates dummy variables for the top ‘n’ values in the feature duplicate column and converts the remaining values (n+1 to the last value) to ‘other’
3. After completing these tasks for the first feature in the list, the revised dataframe is then fed back into step 2 above for the next feature, and so on
4. The resulting dataframe is then modified to contain only those columns desired for that particular model run
5. This dataframe is then split into features and target and encoding is performed (one hot encoding for features, label encoding for the target)
6. The function returns this modified dataframe, plus X and y

Below are the functions required to set the top ‘n’ values for dummy variable creation.  

### First function
The first is a helper function that takes a single feature and groups the dataframe by number of projects for that feature, creates a duplicate feature column, renames any feature value not in the top ‘n’ list as ‘other’, and returns the modified dataframe.

```
def name_replace_less_than_n(df, col, n=125, p=126, new_name='other'):

    dup_col = '{}_duplicate'.format(col)
    
    df[dup_col] = df[col]
    
    # First:  create sorted list of all col by project count
    sorted_list_all = df[dup_col].value_counts().index.tolist()
    
    # Second:  create top 'n' list and not-top-n list
    sorted_list_top_n = sorted_list_all[:n]
    sorted_list_not_top_n = sorted_list_all[p:]

    # Third:  change name of items on not-top-n list
    df.loc[df[dup_col].isin(sorted_list_not_top_n), dup_col] = new_name
    
    # Fourth:  
    return df

``` 

### Second function
The second function iterates through this process for each feature in the feature list provided to the function, then specifies features to use in modeling and performs encoding of the features and target of the resulting dataframe.  

```
def top_n_encode(df, features, features_top_n, all_model_features, n=125, p=126):

    df = df[features]
    
    # First:  create duplicate columns for features of interest, then replace those values NOT
    #         in the top 'n' list with value = 'other' 
    for col in features_top_n:
        name_replace_less_than_n(df, col, n=n, p=p, new_name='other')
        
    # Second:  set features to include
    df_n = df[all_model_features]
    
    # label encoding for target
    le = LabelEncoder()
    y = le.fit_transform(df_n['status_group'].astype(str))
    
    # get feature dummy variables 
    data = df_n.drop('status_group', axis=1)
    X = pd.get_dummies(data, drop_first=True)
        
    return df_n, X, y

```

From this point, the modeling functions I developed will take in df_n, X, and y and perform train-test-split, perform SMOTE resampling (where applicable), fit the model, and display results—including accuracy scores, feature importance tables (except for support vector machine models using rbf kernel), and confusion matrices.

## How much difference does this process make?
If we had used the original feature set presented above, 4,639 dummy variables (dv’s) would have been produced!  The bulk of these dv’s are from two features—‘funder’ and ‘scheme_name’ (1,898 and 2,560, respectively).

By setting n=125, we’ve reduced the number of dv’s to 431, with very little sacrifice in the accuracy score (76.40% for all 4639 dv’s vs. 76.01% for 431 dv’s).  Further, reducing the number of dv’s reduces the possibility of overfitting, and the amount of time it takes to run models, especially the more complex ones.  

While the amount of time required to run a decision tree model for the case of 4,639 dv’s wasn’t much more than for 431 dv’s, we can expect that more complex models will require much greater amounts of time with substantially larger numbers of dv’s.  Time and computing resources are limiting factors in any data science project, so trade-offs must be made regarding how many dummy variables to include in the case of a categorical classification model utilizing categorical features with hundreds or thousands of values.

In this project, I created a number of other handy functions, including plots of visualizations of well status by feature and consolidating feature importances into an easy-to-understand summary roll-up table.  These functions may be the subject for future blog postings.  


## References: 

1.	Water.org website on its work in Tanzania:  https://water.org/our-impact/where-we-work/tanzania/
2.	Tanzania Ministry of Water website:  https://www.maji.go.tz/
3.	Taarifa map of water supply projects in Tanzania (note that the Taarifa dashboard on the website was not working as of June 30, 2020, but the map is functional):  http://dashboard.taarifa.org/#/?max_results=100&reports_only=0
4.	DrivenData webpage for Tanzania water well prediction competition:  https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/


