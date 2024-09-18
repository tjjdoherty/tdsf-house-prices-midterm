# Data Science Midterm Project

## Project/Goals

In this project we took a dataset full of housing information from housing stock across different US State capital cities and built a regression model to predict prices. The goal was to build a high performing machine learning model that could accurately predict price, performing well when being tested on unseen data.

**Notes to Readers** - We are refactoring to move some of the data cleaning and wrangling, which was extensive, into the notebook 0.5 - Data Cleaning, which is currently empty. All data cleaning, wrangling and EDA is done in 1 - EDA.ipynb at this time

## Process
### Step 1: Importing from .json
- In EDA.ipynb we accessed the directory of .json data files from the data folder to create a raw dataframe (Pandas/Python) full of the housing information.
- Collaborating on the project on two different OSs/machines, we simply ran different lines of opening the csv into a dataframe once we had saved the json as a local .csv file in our respective local directories
  
### Step 2: Cleaning of null columns/rows, duplicates, and fillna
- Initially we tried to trim down the 67 columns of data to reduce the complexity of the model we would build. To do this we saw that 17 of the columns were entirely null across all records, so these were immediately removed
- It also became clear when we did summary statistics that there were some instances where the same property_id appeared 5 times, so many dupes were present. Once removing duplicates by copies of the property_id, we went from 8100 records to 1700 in the database.
- Many instances where data was missing from columns coincided with records without latitude/longitude fields being completed, so these records were also removed.
- Many nulls in columns that were clearly important to house price were filled by using either a 0 in the case of number of beds/baths/garage, mean or median values for things like the lot square footage (lot_sqft) and home square fottage (sqft)
- There were several columns referring to the different type of bathrooms (full, 3/4, half and 'baths'). Baths had the strongest correlation coefficient to the sold price, so to avoid multicollinearity this description.baths was the only column bathroom related that was kept, the others were dropped.
- Many columns had only one or two completed records out of the 1700 non-duplicate rows remaining, so these columns were dropped too.

### Step 3: Feature Engineering
- There were many tags in a single column for each property listing amenities like fireplace, dining room, air_conditioning etc. Many of which seemed important and they were One Hot Encoded (OHE'd) into the dataframe. To include all tags would be extremely complex when building the model and performing EDA so only those which frequently appeared were included. Two columns were then made - n_amenities, which was the sum of the number of these amenities each property included, and n_high_amenities, the sum of the number of amenities each property had, specifically the amenities which had a correlation coefficient of 0.1 or above with respect to the sold_price
- Other feature engineering was the sum of beds and baths combined into n_rooms, the number of days the property listed for, and the age of the building (the difference between year_built and the current year)

### Step 4: EDA
- We uses a Seaborn heat map of the correlation matrix to identify which features were the strongest correlators to the sold_price. The weakest correlators were dropped from the dataframe when it came to build the model.
- One of the issues frequently seen was the extremely weak correlation coefficients between some features because so few records actually had them. These were dropped as explained above.

### Step 5: Model_selection
- As we are predicting a numeric value in sold_price, we tried vanilla linear regression, XGBoost and Ridge Regression in the model selection, testing for R^2 score. We also enhanced the Ridge Regression with kfold cross-validation.

## Results
 Using XGBoost we are returning an R^2 of 0.76 in training and 0.56 in testing - clearly overfitting is happening but our overall strongest test performance
- Using Ridge Regression we are returning an R^2 of 0.51 for both training and testing.
- Vanilla linear regression was returning 0.33 in training and 0.64 in testing although this is highly volatile - changing the random state gave very different results
- We added k-fold (n = 10) cross-validation to the Ridge Regression
  
## Challenges 
- The number of features in the data is truly vast. We had to leave out the cities/state data entirely, and focus on the features describing a home individually. This is because encoding a yes/no value for state when there are potentially 50 different states would create an enormous solution space for the model to explore, finding combinations of all 50 states with all other features selected to estimate the sold_price, let alone multiple cities in the same state. This is despite the high likelihood that which state the property in does affect value. We may need to feature engineer something which can bring some ordinality to the states, e.g. median income in each state (widely considered an indicator of home value as higher incomes in the area --> higher home values)
- There is a little bit of overfitting in most of our modelling, but XGBoost is looking promising and once we have the hyperparameters tuned we may have a well performing model.
  
## Future Goals
- XGBoost is the optimal model from what we have explored
- We will tune the pipeline and create the presentation showing the results. We will likely lead with XGBoost model as it is our best model, with gridsearch for hyperparameter tuning
- In future, we can also look at PCA as a means to reduce the dimensionality of our data set.
