
# Home Rent Prediction 

Welcome to Python-based Machine Learning project! In this repository, we delve into the world of predictive modeling, leveraging various techniques such as Data Exploration, Data Preprocessing, Feature Engineering, Linear Regression Model, and Random Forest Model.


##  Project Overview

The primary goal of this project is to compare the performance of two popular regression models, Linear Regression and Random Forest, on a given dataset. By analyzing and showcasing their respective results, we aim to demonstrate how Random Forest outperforms Linear Regression in predictive accuracy.
## Data Source

The dataset used in this project is sourced from Kaggle, a free-to-access website renowned for its extensive collection of diverse datasets. We acknowledge and express our gratitude to the data provider for making this dataset publicly available, enabling researchers and enthusiasts to explore and innovate in the field of Machine Learning.
## Variables

The dataset has 10 variables. They are:

longitude, latituded, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value, ocean_proximity


## Data Exploration

Here all the variables have 20640 non-null values but "total_bedrooms" have 207 Missing values. So after droping those missing values we get 20433 non-null values for each variables.
## Importance of Data Preprocessing

One of the key takeaways from this project is the significance of data preprocessing in achieving superior model performance. Through proper handling of missing data and feature transformation, I witness a substantial boost in the Random Forest model's accuracy compared to Linear Regression.

I invite you to explore our code, datasets, and detailed documentation to gain valuable insights into the techniques employed and the outcomes observed.
## Features

- #### Data Exploration: 
    We begin by thoroughly exploring the dataset, gaining valuable insights into its structure, distribution, and patterns. Understanding the data is crucial for successful model building.

- #### Data Preprocessing
    Data preprocessing plays a pivotal role in enhancing the model's performance. We handle missing values and apply necessary transformations to ensure the data is suitable for modeling.

- #### Feature Engineering: 
    In this phase, we identify and create relevant features that contribute to the predictive power of our models. This step can significantly impact the model's overall performance.
- #### Linear Regression Model: 
    We implement the classic Linear Regression algorithm to establish a baseline for our predictive modeling. The R-squared value achieved by this model is approximately 0.65 means that approximately 65% of the variability in the dependent variable can be explained by the linear relationship with the independent variables in the model.
- #### Random Forest Model: 
    Next, we introduce the powerful Random Forest model, which offers higher accuracy compared to Linear Regression. The Random Forest model yields an impressive R-squared value of around 0.96 means that the model correctly predicted around 96% of the cases in the test data.


## Dependencies:

- Python 3.x
- Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn
## Data Before Preprocessing

I split the dataset into two parts. 90% of the data used of training purpose and 10% were used for testing. For sure the test data is intact and it is not touched before training.

![](https://github.com/Wasif-Allvi/Home-Rent-Prediction/blob/4c7988447c7e14d9b54f0fabbec870e13d0e8290/results/exploratory_data_analysis/before%20preprocessing/data_exploration_before_preprocessing.png)



The variable "ocean_proximity" is not present here because it doesn't contain any numerical value. I has only 5 non-numerical values - <1H Ocean, INLAND, NEAR BAY, ISLAND, NEAR OCEAN.

Here we can clearly observe, the data is not having a Normal Distribution. The varibales "total_rooms, total_bedrooms, population, households" are right-skewed distribution. It is also known as positively skewed distribution. 
#### Coorelation Matrix of Training Data using Heatmap


![](https://github.com/Wasif-Allvi/Home-Rent-Prediction/assets/45194832/e5aa508e-5d22-4b16-a72e-de5a626c739f)

Here we can see the coorelation of each variable with all other. We can clearly see the "median_house_value" is strongly coorelated with "median_income" with the value of 0.69. So, it is a very important variable to work with.
## Data After Preprocessing

As the data has Right-skewed distribution, so it needs to be preprocess to work further. So I use Logarithmic Transformation on "total_rooms", "total_bedrooms", "population", "households" to make the distribution more symmetrical. 


![](https://github.com/Wasif-Allvi/Home-Rent-Prediction/assets/45194832/aa69ef2c-3a1e-428b-aaee-11b625c5e9a7)




#### Coorelation Matrix of Training Data using Heatmap

I feel the variable "ocean_proximity" will be a key factor assuming the house value. So I make all the classes "ocean_proximity" have into boolean features. That means, if "<1H Ocean" present in the feature it will be 1 otherwise 0 and so on.

![](https://user-images.githubusercontent.com/45194832/256948661-3829f875-808c-4170-99b1-9044e2592fbf.png)

We can see here, the median_house_value is negatively coorelated with the feature INLAND. That means, If someone stays in INLAND he will pay less for the house rather than someone stays near ocean. In other words, if the house is very near from the ocean the price will be high.




## Feature Engineering

Adding a new feature "bedroom_ratio" will be a interesting thing. That means, how many rooms are bedroom among all the rooms.

One more feature "household_rooms" is added here that reflects how many rooms per households. 

These two feature might influence the price of the house.

Lets see the heatmap again.

![](https://github.com/Wasif-Allvi/Home-Rent-Prediction/assets/45194832/592209fd-1ce4-4981-8be1-b10f1a83b4c2)

Now the interesting part is, "bedroom_ratio" has a negative coorelation with the "median_house_value" with the value of -0.2. Where the total number of bedrooms is also very weakly coorelated with our target variable "median_house_value" with the value of 0.051.


The "households" is also very weakly coorelated with the "median_house_value" with the value of 0.071. But the new feature "household_rooms" has much more strong coorelation with the "median_house_value" with the value of 0.11 and that looks interesting!

## Linear Regression Result


Now Lets fit the Linear Regression model and evaluate our test data. 

![](https://github.com/Wasif-Allvi/Home-Rent-Prediction/assets/45194832/6594da01-0315-4431-887a-5eaebfc1f51b)

In this graph, the data points are scattered. That means, the error rate(y-f(x)) is high. This reflects linear regression model did not provide a good fit to the data.

Here the regression score is 0.654 that means the model can predict correctly only 65% of the data. This is not a very good score, neither a bad score.


## Random Forest Result

![](https://github.com/Wasif-Allvi/Home-Rent-Prediction/assets/45194832/e323edf9-ea2f-471f-ad2a-7e6e2cbf9f6c)

In this graph, the data points appear less scattered compared to the linear regression model. The reduced scatter suggests that the error rate (y - f(x)) is lower when using the random forest model, indicating a better fit to the data.

With a regression score of 0.96, the random forest model demonstrates higher accuracy in predicting the data compared to the linear regression model. The score of 0.96 means that the random forest model can correctly predict approximately 96% of the data points. This indicates a significantly better performance than the linear regression model, making the random forest model highly effective in this context.
## Contributions

I welcome contributions, feedback, and suggestions to improve this project. If you have ideas for enhancements or bug fixes, feel free to open an issue or submit a pull request.

I hope this project inspires you to explore the endless possibilities of Machine Learning and predictive modeling. Happy coding!
