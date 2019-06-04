---
layout: post
title:      "King County housing price analysis—Surprises, musings, and further analysis"
date:       2019-06-04 09:13:48 +0000
permalink:  king_county_housing_price_analysis_surprises_musings_and_further_analysis
---


Whether we’re consciously aware of it or not, we humans (and most other life forms on the planet) are wired to notice and seek out novelty.  Whether searching for food or avoiding predators, our ancestors had to be good at observing and respond to novel events in the environment or get kicked out of the gene pool.  

Surprises are a kind of novel event.  One reason why the field of data science is so interesting is that one can find novel or even surprising results in virtually any data set and data analysis.  Surprises can be both fascinating and perplexing, presenting us with puzzles to ponder and riddles to solve.  We might expect to see certain results, but seeing something unusual compels us to look further—first, to figure out whether or not we made an error; second (and more exciting), to find a potentially novel relationship in the data.  

Making a conscious effort to remain open to a range of possibilities (read:  “surprises”) can expand our insights and deepen our understanding of a phenomenon.  At the same time, it is important to focus one’s scope in a project and not attempt to “boil the ocean” by exploring every possible avenue and combination of effects.  A data scientist’s skillset must include the ability to balance open exploration of potentially new effects with the ability to constrain scope given the allotted time and resources.

***The stage is set…***

With this as a backdrop, I began work on the first big project in the Flatiron School Data Science Career Track curriculum, an analysis of a data set of over 20,000 houses in King County, Washington to identify the features that have the largest impact on home values.  The data set included information on each home’s physical structure (square footage of living space, number of bedrooms and bathrooms, square footage of lot, square footage of basement, number of floors, etc.) and other attributes (e.g., ZIP Code, latitude/longitude, waterfront location, year built, year renovated, and views of home by prospective buyers).  

Using the **OSEMiN** framework, we were tasked with **obtaining** (importing) the data set, **scrubbing** (cleaning) the data, **exploring** the data (exploratory data analysis, or EDA), **modeling** with multi-linear regression (ordinary least squares, or OLS, using Statsmodels for python) to answer specific questions we had generated, and **interpreting** results.  In my initial review, cleaning, and exploration of the data set, I found myself considering several questions:

* Bedrooms and bathrooms:  Common knowledge and life experience tell us that home price goes up with the number of bedrooms and bathrooms, all other things being equal.  But what does that relationship look like, and are there other variables that might be as important—or even more important?  
* Square footage of the home:  Clearly, square footage is important, but how important?  Are other features (such as number of bedrooms and bathrooms) relatively more or less important than overall square footage?
* Square footage of the lot on which the house is situated:  Does a small lot (or the absence of a lot) result in a reduction in a home’s market value?  
* Presence of a basement:  Does the presence of a basement—and potentially its square footage—increase home values?  
* Waterfront:  Does a waterfront location have a significant effect on the home value?
* Year built:  What sort of effect does the age of the property, represented by the year the home was built, have on price?
* Year renovated:  how much does renovation affect prices (if at all)?  
* Grade:  How does the house “grade” (as described by King County’s 13-point “grade” system characterizing home quality, features, and amenities) affect prices?  
   * A pretty clear relationship can be seen when graphing grade against price in the original data set (before removing outliers).  This relationship appears to be non-linear across the entire 13-point scale, but is this true for the majority of homes (those that are not at the high or low end of the price scale)?
   *	Note that “grade” is different than “condition”, which is another scale used in the region to describe the basic condition of homes, from 1 to 5.  

After initial data scrubbing to eliminate null values and ensure that the data points within each feature were of the same data type, I ran a scatter matrix and noted the following:

* Strong correlation with price:  sqft_living, sqft_above (square footage of living space above basement level), sqft_living15 (square footage of living space of 15 nearest neighbors), number of bedrooms, number of bathrooms, condition, and grade
* Little or no correlation with price:  sqft_lot, sqft_lot15 (lot size of 15 nearest neighbors), view, and floors
* Unclear correlation with price:  yr_built, yr_renovated, lat/long, zipcode, waterfront, and sqft_basement

***“We interrupt this program for a special announcement…”***

Before continuing with the findings and thoughts about surprises, I thought it might be helpful to talk a bit about my experience during this first project.  The scatter matrix visualizations provide a useful example:  they provide very useful guidance regarding where to focus one’s efforts, but as a student in learning mode, I wasn’t entirely sure how to use them.  I ended up spending a LOT of time going through multiple iterations of the scrubbing, EDA, and modeling processes, because I was afraid of “missing” something or “losing” an important variable.  I also experimented with cutoff points for outliers, creating lots of dummy variables (then deciding not to use most of them), etc.  Finally, I ran quite a number of model iterations, first with individual variables, then with combinations of variables.  

While there was some value for me in going through this process, I think I went just about every iteration there was and exhausted myself in the process.  Having gone through this, I now understand the value of more targeted approaches, and look forward to gaining the tools to balance open exploration and discovery with efficient workflow processes that will allow me to approach the process in a more systematic fashion.  

***“Now back to our regularly scheduled program…”***

Getting back to results…as I proceeded through the project, I eventually developed a model that achieved a very high R-squared score (98-99%, depending on which 3 or 4 variables I chose) with a small number of features.  In addition to the high R-squared score, the model shows a low skew (0.372), a kurtosis value of 3.072, a Jarque-Bera (JB) score of 399, and a Condition Number of 50.1.  I gained a high degree of confidence in this model’s ability to capture the key factors involved in home values in King County.  

And yet…there were a few surprises that prompted more questions.  Without going into great detail here, I found that certain features that are commonly assumed to influence home values appeared to have little or no impact in my model for this data set.  Life experience and “common knowledge” strongly suggest that these features actually do have an impact on home price.  I wondered if the scale of the geographic area might be obscuring these relationships.  For example, running the model with a small subset (just homes in one zip code, for example) might yield a more pronounced effect in the model of features such as waterfront or lot size.  

On the topic of ZIP Codes—I would also be interested in dividing zip codes into a small number of bins based on average home price, to see whether a particular zip code pushes price up or down.  If the ZIP Code turns out to be a significant factor in determining home price, then it would be intriguing to see whether an additional bedroom or bathroom adds more to a home price in certain ZIP Codes and less in others, even for homes that are very similar in most other respects.

Finally, I am intrigued by the idea that there may be certain ideal ratios or “rules of thumb” that implicitly tend to guide prices (e.g., square footage of living space relative to square footage of lot—perhaps only detectable at a local level, such as at the level of a ZIP Code or even neighborhood or district within a ZIP Code).  This is something that I would like to explore further if time permits.

