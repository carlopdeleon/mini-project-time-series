# The Weather outside is Weather*

# Project Descriptions

This project is to discover weather patterns in Rangoon, Burma. This project is so vast that the data retrieved goes back all the way to the year 1796. We have embarked on a journey back to the days gone by in order to predict weather patterns for the days that have not gone by. This is a time series project to predict the future. 

# Goal
- Explore historic data and discover weather patterns for Rangoon, Burma
- Utilize machine learning models to predict future weather patterns


# Acquire

- Acquired from Kaggle: 
    
    - Link: https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data
    - Used Global Land Temperatures by Major City file.
- 239177 rows. Each row represents a date of when the weather was observed.
- 7 columns. Each column are values the represent temperature, city, grid coordinates, and country

# Prepare
- Set date as the index.
- Normalized column names to be more python friendly.
- Dropped nulls.
    - 81 observations.
- Subset observatios for Rangoon, Burma.
    - remaining rows: 2613
- Convert from celsius to farenheit for readability.
- Split data into three sets (55/25/20): Train, Validate, and Test


# Data Dictionary

| Feature | Definition |
| :-- | :-- |
| avg_temp | average temperature  |
| avg_temp_certainty | average amount of degrees difference from aveage temperature |
| city | city name|
| country | country name | 
| lat | latitude coordinate |
| lon | longitude coordinate |


# Steps to Reproduce

1. Clone this Repo
2. Download data from Kaggle: https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data
3. Use functions in prepare.py to clean and prep data.
4. Use function in the explore.py to explore data.
5. Use functions in visual.py to plot the charts.
6. Use same configurations for models.

# Conclusion

**Explore:**
- Rangoon's seems to have a temperate climate with average temperatures ranging from low 70s to low 80s. 
- Records go back all the way to 1796, which brings question to how accurate those measurements are. 

**Model Performance:**
- Rolling Average model on test performed 0.16 worse when compared to the validate data set. Although, it was still able to perform 2.12 better than Previous Cycle and 0.95 better than Holts Seasonal. 




