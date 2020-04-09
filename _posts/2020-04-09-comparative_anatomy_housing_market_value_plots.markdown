---
layout: post
title:      "Comparative Anatomy:  Housing Market Value Plots"
date:       2020-04-09 02:24:37 -0400
permalink:  comparative_anatomy_housing_market_value_plots
---

## Overview
The Module 4 Project in the Flatiron School Data Science Version 2 Curriculum provides students with experience in working with time series data—in this case, monthly housing values from 04/01/1996 to 03/01/2018 for over 14,000 ZIP Codes in the U.S.   Analyzing this data requires data science tools capable of handling large data sets.  I created a Jupyter notebook using python, numpy, pandas, matplotlib, seaborn, and used Internet search engines to get information about how to accomplish various technical tasks.

The project asks students to identify the 5 "best" ZIP codes for investment (as defined by the student) and develop support for these recommendations to a hypothetical client (e.g., a real estate investment trust).  The student must develop a methodology for evaluating ZIP codes as potential investment targets and support these findings through analysis using Jupyter notebook.  The student must also prepare a non-technical PowerPoint presentation and a blog post. 

Analyzing time series data requires a number of decisions around data manipulation and modeling tool selection.  In the case of this housing market value data, I used an ARIMA model.  Because seasonality effects on home values turned out to be negligible, I did not include a seasonality parameter in the modeling.  

I used the entire historical timeframe (going back to 1996) as a starting point for the modeling.  My understanding is that, even with the market bubble and crash, this approach is generally acceptable for at least two reasons:  1) the autoregressive and moving average components are heavily dependent on recent history and much less so on older data, and 2) more data is generally beneficial to model accuracy.  With more time, I may revisit the data to see how the predictions and confidence intervals change with data only looking at the past 6 or 8 years.

## Deep dive:  focus on visualizations
While my Module 4 project officially focuses on ZIP codes in the Sacramento metro area, I also explored average values for 16 metro areas and used ARIMA modeling and forecasting tools to predict the values 24 months into the future. 
While I did not have time to analyze ZIP codes in these major metropolitan areas, I wanted to see if the plots of average home market values at the metro area level showed some shape features similar to some of those seen at the city or ZIP code level.  Further, I wanted to find out whether some of my hunches about curve shape features and investment quality might have validity.  

I created visualizations of the top 30 metro areas, then compared these against my intuition and personal knowledge as a homeowner of the housing market collapse and how it affected various regions of the US and the country as a whole.  I selected several metro areas that I thought looked promising, as well as some others that I thought would perform less well or would experience greater volatility, then performed ARIMA modeling and forecasting on these areas.

### Evaluating visualizations of time series for trends
Having looked at many visualizations at the metro, city, and ZIP code level, I began noticing certain characteristics in the housing value curves that I thought might correlate to investment quality.  I noticed that across different zip codes, cities, and even metro areas, there were some discernable pattern ‘types’ within the general broader patterns shared by most (though not all) visualizations.

Before exploration shape variations, let’s define what a “typical” pattern looks like:  
* A slow but gradually accelerating ascent in house values until about 2003
* Starting in 2003, prices began increasing and then rapidly accelerated upwards
* Peak of bubble generally shows up as the top of a pointy hill around 2006-07
* Once the market peaked, the plots show a steep descent in values through 2009 or so
* Continued but more gradual declines in market value until around 2012
* The values of the post-crash bottom in 2012 timeframe lower than 2003 values
* Most markets showed appreciable gains from 2012 to 2014 and rapid growth starting in 2013
* Starting in the 2014-15 timeframe, some markets saw the growth rate slow, while others continued to experience robust growth until the end of the data set on April 1, 2018.  
 
*A “typical” housing market value plot:  Tampa, FL metro area*

![Tampa FL](https://raw.githubusercontent.com/gdurante2019/dsc-mod-4-project-online-ds-sp-000/master/images/blog_typical.png)

*NOTE:  while the housing bubble peaked around 2006 for most markets, it is worth noting that the Case-Shiller National Home Price Index noted a record drop (over 18% on average) in home values in the last 3 months of 2008.  This is the reason for the red line in the graph above at the beginning of 2009.  The bubble had burst, but the housing market continued to show weakness, and any remaining talk of a relatively shallow market decline and relatively quick recovery was replaced by much more negative sentiment, which lasted at least a couple of years afterwards.*  

#### Variations on the “typical” pattern

There were several variations that I noted when looking at market value plots at the metro, city, and ZIP code level.  Though I have not yet had time to perform analyses investigating each of these—and I doubt that they would all be predictive of future outcomes—I did observe some tantalizing correlations and I think further analysis could yield additional useful insights into visualization assessment.

Below are some key shape variations that I noticed across metro areas.  These shape variations can also be seen at the city or ZIP code level.

* A shallower market bubble peak and market bust trough compared to the “typical” profile:

![shallow market bubble peak](https://raw.githubusercontent.com/gdurante2019/dsc-mod-4-project-online-ds-sp-000/master/images/blog_shallow_peak.png)


* A broader market bubble peak vs. a sharp, pointy market bubble peak:

![broader market bubble peak](https://raw.githubusercontent.com/gdurante2019/dsc-mod-4-project-online-ds-sp-000/master/images/blog_broader_peak.png)


* An earlier peak vs. a later peak:

![earlier market peak](https://raw.githubusercontent.com/gdurante2019/dsc-mod-4-project-online-ds-sp-000/master/images/blog_earlier_peak.png)      


* An earlier trough vs. a later trough:

![earlier market trough](https://raw.githubusercontent.com/gdurante2019/dsc-mod-4-project-online-ds-sp-000/master/images/blog_earlier_trough.png)     


* A more wobbly curve (e.g., greater fluctuations in values from one time period to the next) than the “typical” profile:

![wobbly curve](https://raw.githubusercontent.com/gdurante2019/dsc-mod-4-project-online-ds-sp-000/master/images/blog_wobbly_curve.png)


* A convex curve during the buildup years (2003 or even earlier, up to 2007-2008), as opposed to the concave curve for the “typical” profile during the same period: 

![convex buildup curve](https://raw.githubusercontent.com/gdurante2019/dsc-mod-4-project-online-ds-sp-000/master/images/blog_convex_buildup.png)      


* How much, if at all, did market values dip below 2003 values during the crash?  Some ZIP codes had a big drop during the crash relative to 2003 values, some had small dips below 2003 values during the crash, and some (though fewer) never reached as low as 2003 values, even during the crash:

![crash dip below 2003 vals](https://raw.githubusercontent.com/gdurante2019/dsc-mod-4-project-online-ds-sp-000/master/images/blog_dip_below_2003_vals.png)


* Housing market values by 2018 that exceeded the market bubble heights of 2006-2007:

![2018 values exceed bubble peak](https://raw.githubusercontent.com/gdurante2019/dsc-mod-4-project-online-ds-sp-000/master/images/blog_2018_exceeds_bubble_vals.png)


* Housing market values by 2018 that had not yet reached the market bubble heights of 2006-2007:

![2018 values below bubble peak](https://raw.githubusercontent.com/gdurante2019/dsc-mod-4-project-online-ds-sp-000/master/images/blog_2018_below_bubble_vals.png)


* Precipitous/spiky dips in housing values post-crash, vs. more gradual declines and then increases in home values:

![spiky dips](https://raw.githubusercontent.com/gdurante2019/dsc-mod-4-project-online-ds-sp-000/master/images/blog_spiky_dips.png)


* Slope of curve during the recovery (post 2012)—is it pretty consistently linear in an upward direction, or are there inflection points indicating times of more rapid growth vs. less rapid increases:

![recovery curve shape](https://raw.githubusercontent.com/gdurante2019/dsc-mod-4-project-online-ds-sp-000/master/images/blog_recovery_curve.png)        


* Slope of curve in the 6- to 12-month period prior to the last data point on 4/1/2018:

![final 6-12 months of data](https://raw.githubusercontent.com/gdurante2019/dsc-mod-4-project-online-ds-sp-000/master/images/blog_final_6-12_months_curve1.png)

### Some initial findings regarding a couple of shape features

The findings below are preliminary, and some are based on the more detailed analysis I performed on ZIP codes in the Sacramento metro area.  Others also draw upon what I observed in the metro area analyses.

* ***Broader peak during height of the bubble*** — What, if any, relationship was there between the width of the peak of market values and subsequent performance?  
  * I could not find an apparent relationship between shape of peak and market performance
  * This surprised me; for example, I thought that a broader peak might correspond to a stronger recovery and forecasted growth, but this was not consistently true
* ***Earlier peak*** — How do markets with an earlier peak during the bubble fare versus those that peaked later?  
  * Initially, I had thought that an earlier peak might be correlated with worse predicted values, but in looking at the semi-finalists in the Sac metro region, there isn’t enough evidence to support this
   * While two of the worst performers (Sacramento-DelRios and Arden-Arcade) peaked particularly early...
    ![earlier bubble worse](https://raw.githubusercontent.com/gdurante2019/dsc-mod-4-project-online-ds-sp-000/master/images/blog_early_peak_arden_delrios.png)
   * ... other ZIPs with early peaks did better (e.g., Granite Bay, Rescue—two of the top 5 ZIPs):
    ![earlier bubble better](https://raw.githubusercontent.com/gdurante2019/dsc-mod-4-project-online-ds-sp-000/master/images/blog_early_peak_gb_rescue.png)


* ***2018 values exceeding market heights during the bubble***—For markets that saw values increase to new highs in 2018 that were even higher than the highest point of the housing market bubble in 2007
  * **Questions:**
    * Would their results differ from those markets that have rebounded, but have not yet reached the highs seen during the housing bubble? 
    * Are the markets at new highs experiencing a new ‘bubble’, or are the new highs merely a reflection of a strong local economy or other factors (e.g., gentrification of an area or longer-term population trends in the area) that have led to sustainable valuations?  
    * Likewise, for the former set of markets, are they due for stronger growth relative to those markets that have already exceeded their previous high valuations during the bubble, or are the former markets simply lagging behind in terms of recovery and growth?
    * What correlation, if any, was there between the difference between market peaks and troughs, versus the subsequent higher recovery values in 2018?  (In other words, how did markets with a more ‘typical’ plot fare against markets that did not experience dramatic highs and lows during the market bubble?)
  * **Initial observations:**
    * Of the semi-finalists in the Sac metro region, ZIPs with 2018 values that were higher than the 2006-2007 peak values (e.g., Sacramento-Del Rios, Arden-Arcade), tended to have greater representation in the mediocre and lower ratings groups…  
  ![higher than 2018 worse performance](https://raw.githubusercontent.com/gdurante2019/dsc-mod-4-project-online-ds-sp-000/master/images/blog_2018_higher_arden_delrios.png)
   * …whereas the top 5 ZIP codes all had 2018 valuations at or slightly lower than the 2006-2007 peak:
  ![lower than 2018 better performance](https://raw.githubusercontent.com/gdurante2019/dsc-mod-4-project-online-ds-sp-000/master/images/blog_2018_lower_auburn_slaketahoe1.png)
   
   * It is worth noting that there were also a number of cases where ZIPs with higher 2018 values still had good investment ratings, and also ZIPs where the 2018 values had not reached values at the height of the bubble, and yet they had mediocre or poor ratings
   * On the other hand, a quick review of a few other metro areas that did not experience a large drop in values during the recession, but which experienced significant market value gains during the recovery, suggests that these areas have the potential for significant continued market value gains
 
 ***Recommendation:***  Markets that did not see a big bubble, but which have experienced 2018 values higher than values during the market bubble, warrant further investigation
 
### Further analyses to consider
* Given that ARIMA analyses gives greater weight to more recent observations, it would be interesting to see how the slope of the curve in the most recent 6-12 months (2017-2018) of actual housing market values correlates with forecast
* Another area of potential interest is the correlation, if any, between post-crash dip relative to 2003 values and predicted values

## Conclusion
In this blog, I’ve sought to highlight some of the variations in market value plots for a variety of geographic areas, and suggest possible relationships to explore further.  Because the original Zillow dataset is so large, my broader intent is to highlight how exploratory data analysis and visualizations can help guide our decisions regarding which data to analyze and the best methods to choose.  

