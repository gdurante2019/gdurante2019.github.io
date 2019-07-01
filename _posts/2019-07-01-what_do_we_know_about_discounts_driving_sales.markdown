---
layout: post
title:      "What do we know about discounts driving sales??"
date:       2019-07-01 07:00:43 +0000
permalink:  what_do_we_know_about_discounts_driving_sales
---


One of the great things about data analysis is that it helps make clear what we really "know" versus what we may not know as well as we thought.  Data analysis can help confirm what we already "know"... or it challenges our preconceived notions about how something works and invites further investigation.  An important method for accomplishing this is hypothesis testing.  For Flatiron's end-of-Module 2 project, each of us was tasked with obtaining data from a SQL database for the purpose of answering at least four questions that can be answered using hypothesis testing.  

In a nutshell, hypothesis testing asks us to state our question of interest in two parts:  the Null Hypothesis and the Alternative Hypothesis.  
* At the most basic level, the Null Hypothesis can be thought of as saying, "there is no relationship between two (or more) phenomena".  A trivial example of a Null Hypothesis could be "there is no relationship between the sky being full of clouds on a particular day and rain occurring.  A more realistic example could be, "there is no relationship between the number of stock market IPOs in a given quarter and a subsequent 5% increase in the NASDAQ the following quarter.
* The Alternative Hypothesis, on the other hand, encapsulated the possible relationship we are trying to discern.  In the stock market example, the Alternative Hypothesis could be stated as “an increase of 5% in the NASDAQ stock exchange in the quarter after X number of IPOs in the previous quarter is unlikely to be due to chance.”  The important nuance here is the “unlikely to be due to chance” part. *The key thing to remember is that hypothesis testing does not “prove” anything; it merely gives a certain level of confidence that a relationship is likely to be, or not likely to be, due to chance.  *

With this as background, I set out to answer at least four questions about discounts and sales for Northwind, a fictitious company whose data is the subject of this analysis.  The database is a SQL database developed by Microsoft to help people learning Structured Query Language, or SQL.  It is a useful tool in that it is relatively small and has a circumscribed set of variables related to customers, orders, sales staff, suppliers, regions, and shipping companies.  

The four questions I set out to answer were:  
1. Do discounts result in increased order quantities?  (Required by Flatiron for this project)
2. Do discounts in general result in increased revenues per order?  
3. Do different levels of discount (e.g., 5%, 10%, 15%...) result in different levels of average order quantities, and 
4. Do different levels of discount (e.g., 5%, 10%, 15%...) result in different levels of average order revenues?

I also explored the distributions of order revenues across the eight product categories as an optional 5th question, and began looking into a few other questions around how discounts are employed within product categories.  For the purposes of this blog, however, I'll focus on the four listed above.

Focusing on these four questions allowed me to focus on a well-defined data pull from the Northwind database.  My SQL query focused on the following tables:  Customers, Orders, Order Detail, Products, and Product Detail.  I joined the tables using the relevant ID fields, and was able to import this data into a pandas dataframe.  

Next, I reviewed the data and performed some exploratory data analysis.  I looked at the table headers and the datatypes of the variables using the .info method in pandas.  I also wrote several functions to perform various tasks, to make it easier for me to perform various workflow steps and revisit aspects of the analysis as needed.  Examples of functions I created include  a function for reloading data from the SQL database and recreating certain sub-dataframes if needed, a resampling function, and functions for creating various kinds of plots.  (As an aside, I learned quite a bit about both writing functions and about matplotlib in the course of this project!)

The next step was to create dataframes that I could use for the various analyses I had set out.  Before setting up sub-dataframes of certain slices of data (discounted products versus non-discounted products, for example), I created a revenue column, which would be needed for hypothesis testing for the questions pertaining to revenues.  Having created the Revenue column, I then set up the sub-dataframes.  These included sales of non-discounted products, sales of all discounted products (regardless of discount size) and then dataframes of sales of products in the following discount bands:  1-4.9%, 5-9.9%, 10-14.9%, 15-19.9% and 20% and up.  

With these dataframe created, I could then begin visualizing the data.  For the first quesiton, I plotted histograms of order quantities for discounted products and non-discounted products.  In doing this, I saw that the distribution of order quantities were not normally distributed, but were heavily skewed, with smaller order quantities representing the vast majority of orders in both discounted and non-discounted products, then tapering off fairly quickly as order quantities increased.  The same was also true for order revenues, which followed a similar pattern.  

Because the distributions were non-normal, and it is best to utilize normal distributions whenever possible, I utilized my newfound knowledge of the Central Limit Theorem to create multiple samples from the dataframes, calculating the means for each, and then utilizing the resulting distributions of means for each dataframe in my statistical tests.  Having these normal distributions of order quantity means allowed me to perform two-sample, two-sided t-tests to determine that order quantities were statistically significantly higher for discounted products than for non-discounted products.  Formally, I felt confident in rejecting the null hypothesis (that differences in the average order quantities for discounted products vs. non-discounted products were likely due to chance).  From both the EDA and the hypothesis test, it appeared quite clear that discounts on the whole help drive increased order quantities.

But was this true for revenues as well?  This was the subject of my second hypothesis test (Question 2).  Using the workflow process set up in Question 1, I evaluated plots of the data, performed resampling, and ran the two-sample, two-sided t-test, finding statistically significantly higher order revenues for discounted products than for non-discounted products.  

Having the basic workflow established, I then sought to answer questions 3 and 4 regarding order quantities and order revenues for different discount levels.  Since the order quantity and order revenue distributions were non-normal for the sub-dataframes, I performed resampling to create a distribution of means for each discount level.  Since there were now multiple pair-wise comparisons to be performed, I knew that I needed to look to statistical methods that could perform the necessarily analyses without propogating errors.  I started with ANOVA, which established in each case that at least one of the pairs (e.g., discount of 5-9.9% with discoutn of 10-14.9%) was statistically significantly different enough so as to be very unlikely to be due to chance.  

Because the ANOVA test only tells us that at least one pair is statistically significant, but doesn't tell us which one (or if there are other pairs that are also statistically significantly different), it is necessary to employ another test to handle these pair-wise comparisons.  I chose Tukey's test to do this.  I employed pairwise_tukeyhsd from statsmodels.  In the case of order quantities, I found that all but one of the pairs had statistically significant differences between their distributions.  For revenues, the only non-significant pair was Discounts 15-19.9% and no discounts at all.  It seems that whatever increases in unit volume these discounts drive are offset by the reduction in revenue due to the lower cost.  

Interestingly, both for order quantities and order revenues, the 5-9.9% discount range came out on top, with both the highest revenues per order and the highest order quantities.  This is a particularly intriguing result for quantities, since you might expect that discounts higher than 5% would drive additional unit sales, so that higher discounts would result in uniformly higher order quantities.   This is not what we see, however.  



