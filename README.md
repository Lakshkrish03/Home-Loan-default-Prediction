# 💰 Home Loan Default Prediction - Using Machine Learning 

<center>

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) 
[![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

</center>


![Alt Text](https://github.com/Lakshkrish03/House-Loan-default-Prediction/blob/main/Home_Loan_Defaults_pic.jpg?raw=true)

# 🚀 Introduction

🏠 Navigating the Risks of Home Loans: A Common man's Perspective 🏠

Imagine you're setting off on a journey to buy your dream home. You've saved up, found the perfect place, and secured a loan to make it all happen. But what if, somewhere down the road, you hit a financial bump and struggle to keep up with your mortgage payments? This is where the challenge of home loan defaults comes into play, affecting millions of people around the world.

Understanding Home Loan Defaults:

Home loan defaults happen when folks, like you, who've taken out a mortgage, find themselves unable to keep up with their repayments. It's like getting stuck in a financial pothole on the road to homeownership. When this happens, it's not just the borrowers who face tough times. Lenders, the banks or companies that loaned the money, also get into hot water because they're counting on those repayments to keep their own finances healthy.

Why Predictive Models Matter:

Now, imagine if there was a way to predict these bumps in the road before they even happen. That's where predictive models come in. They're like financial forecasters, analyzing loads of data to spot the warning signs of potential loan defaults. By looking at things like your income, credit history, and even broader economic trends, these models can give lenders a heads-up when someone might be at risk of falling behind on their payments.

______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

Let's delve into the core aspects of our project: the problem area, the Users involved, the main concept, and the potential impact. By examining these elements in detail, we can gain a better understanding of how our machine learning models aim to address these issues effectively and cater to the needs of the users.

We'll start by thoroughly exploring the problem area we're addressing, followed by an examination of the Users who will benefit from our project. Then, we'll discuss the overarching concept behind our machine learning models and how they are designed to provide valuable insights for the users.

The dataset we have isn't made up of current data but can still be really useful for building better machine learning models.

#  🏠 Problem Area

Default occurs when a borrower fails to make scheduled mortgage payments for say more than 30 or 90 days, leading to potential financial losses for the lender.
The problem area revolves around accurately predicting the probability of default for loan applicants, which is essential for lenders to manage credit risk effectively, optimize loan approval processes, and maintain a healthy loan portfolio. 

Challenges in home loan default prediction include: 

* analyzing complex and high-dimensional data,
* identifying relevant features that impact default risk,
* handling imbalanced datasets, and
* ensuring model interpretability and transparency in lending decisions.

By addressing these challenges, this project aims to developpredictive models that can assist lenders in making more informed and prudent lending decisions, ultimately reducing the incidence of loan defaults and improving overall financial stability in the housing finance market.

# 👥 Users 

The users of home loan default prediction models primarily include 

* 💼 financial institutions,
* 🏦 banks,
* 🏡 mortgage lenders, and 
* 👥 credit unions 

# 💡 Big Idea 

The concept of home loan default prediction involves using advanced machine learning algorithms to accurately predict the likelihood of borrowers defaulting on their mortgage loans. 

The key innovation behind this concept is the use of sophisticated algorithms to process large and complex datasets and generate predictions, such as:

* Logistic Regression,
* Decision Trees,
* Random Forest
* XGBoost
* Gradient boosting, 
* Neural networks, and
* Support Vector Machine

Some of these models may not work on all Datasets, but I am planning to explore all the above models if possible. 
  
# 🌍 Impact 

Home loan default prediction models have a multifaceted impact, spanning the financial industry, the broader economy, and society. They enhance risk management practices for lenders, reduce financial losses due to loan defaults, and optimize loan portfolios. By promoting fair and transparent lending practices, they increase access to credit, promote homeownership and wealth accumulation, and contribute to the stability and sustainability of the housing market. Additionally, they foster confidence and trust in the lending industry, promote consumer protection and financial inclusion, and reduce systemic risks, contributing to economic stability and growth.

📉 Some Real Facts in actual Mortgage default figures around the world are as follows (Referenced from https://www.statista.com) :

* As of December 2022, Australian owner-occupiers' debt outstanding amounted to over 1.4 trillion Australian dollars (around 920 trillian US Dollars).
* In the fiscal year 2021, the value of outstanding housing loans granted to individuals by private financial institutions in Japan amounted to around 191.57 trillion Japanese yen (around 1232 trillian Us Dollars).
* In the second quarter of the year, the UK had close to 1.9 billion euros (around 2 trillian US Dollars) worth of mortgages outstanding. Other countries with large mortgage markets included the Netherlands, Spain, Sweden, and Italy - all exceeding 400 billion euros (around 429 US Dollars).
* Despite a short period of decrease after the burst of the U.S. housing bubble and the global financial crisis, the total amount of mortgage debt in the United States has been on the rise in recent years. In 2023, the mortgage debt amounted to 20.2 trillion U.S. dollars, up from 19.3 trillion U.S. dollars in 2023.

With the integration of machine learning models, there's a huge possibility in determining whether an individual is highly likely to default on their payments before approvong their loan applications. In my view, effective utilization of these models could potentially reduce overall defaults by up to 5% to 10% approximately saving at the least 260 trillian US Dollars worldwide.

# 📊 Data 

The main challenge in working with home loan default prediction datasets lies in ensuring data quality, completeness, and consistency. Data preprocessing techniques, such as cleaning, imputation, and Exploratory Data Analysis, are essential for addressing missing values, outliers, and inconsistencies in the data. Moreover, handling imbalanced datasets, where the number of defaulted loans may be significantly lower than the number of non-defaulted loans, requires careful consideration to prevent bias and ensure model fairness.
Overall, the availability of high-quality, comprehensive datasets is essential for building accurate and reliable home loan default prediction models. The chosen dataset URL has been included at the end of this section. This dataset is highly imbalanced and includes a lot of features that make the approach of the project more challenging. 

Please find the following reference sources for my datasets: 
Main Data set: (https://www.kaggle.com/competitions/ai511-homeloan-2022/data?select=train_data.csv)(kaggle.com) – The version of the data is train_data.csv. 

# 🗂️ Data Dictionary

The Raw Dataset that will be used for my modeling has 122 columns and 184506 rows. The following table describes what each and every column in the Dataset is. 

The Table is outlined as follows:

| Column Name                   | Description                                                                                                     |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------|
| SK_ID_CURR                    | ID of loan in our sample                                                                                        |
| TARGET                        | Target variable (1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample, 0 - all other cases) |
| NAME_CONTRACT_TYPE            | Identification if loan is cash or revolving                                                                     |
| CODE_GENDER                   | Gender of the client                                                                                            |
| FLAG_OWN_CAR                  | Flag if the client owns a car                                                                                   |
| FLAG_OWN_REALTY               | Flag if client owns a house or flat                                                                             |
| CNT_CHILDREN                  | Number of children the client has                                                                               |
| AMT_INCOME_TOTAL              | Income of the client                                                                                            |
| AMT_CREDIT                    | Credit amount of the loan                                                                                       |
| AMT_ANNUITY                   | Loan annuity                                                                                                    |
| AMT_GOODS_PRICE               | For consumer loans it is the price of the goods for which the loan is given (In our case this will be the home on which the loan was issued)                                     |
| NAME_TYPE_SUITE               | Who was accompanying client when he was applying for the loan                                                   |
| NAME_INCOME_TYPE              | Clients income type (businessman, working, maternity leave, ...)                                                 |
| NAME_EDUCATION_TYPE           | Level of highest education the client achieved                                                                  |
| NAME_FAMILY_STATUS            | Family status of the client                                                                                     |
| NAME_HOUSING_TYPE             | What is the housing situation of the client (renting, living with parents, ...)                                  |
| REGION_POPULATION_RELATIVE    | Normalized population of region where client lives (higher number means the client lives in more populated region) |
| DAYS_BIRTH                    | Client's age in days at the time of application                                                                 |
| DAYS_EMPLOYED                 | How many days before the application the person started current employment                                       |
| DAYS_REGISTRATION             | How many days before the application did client change his registration                                          |
| DAYS_ID_PUBLISH               | How many days before the application did client change the identity document with which he applied for the loan  |
| OWN_CAR_AGE                   | Age of client's car                                                                                             |
| FLAG_MOBIL                    | Did client provide mobile phone (1=YES, 0=NO)                                                                  |
| FLAG_EMP_PHONE                | Did client provide work phone (1=YES, 0=NO)                                                                    |
| FLAG_WORK_PHONE               | Did client provide home phone (1=YES, 0=NO)                                                                    |
| FLAG_CONT_MOBILE              | Was mobile phone reachable (1=YES, 0=NO)                                                                       |
| FLAG_PHONE                    | Did client provide home phone (1=YES, 0=NO)                                                                    |
| FLAG_EMAIL                    | Did client provide email (1=YES, 0=NO)                                                                         |
| OCCUPATION_TYPE               | What kind of occupation does the client have                                                                   |
| CNT_FAM_MEMBERS               | How many family members does client have                                                                       |
| REGION_RATING_CLIENT          | Our rating of the region where client lives (1,2,3)                                                            |
| REGION_RATING_CLIENT_W_CITY   | Our rating of the region where client lives with taking city into account (1,2,3)                              |
| WEEKDAY_APPR_PROCESS_START    | On which day of the week did the client apply for the loan                                                     |
| HOUR_APPR_PROCESS_START       | Approximately at what hour did the client apply for the loan                                                   |
| REG_REGION_NOT_LIVE_REGION    | Flag if client's permanent address does not match contact address (1=different, 0=same, at region level)       |
| REG_REGION_NOT_WORK_REGION    | Flag if client's permanent address does not match work address (1=different, 0=same, at region level)          |
| LIVE_REGION_NOT_WORK_REGION   | Flag if client's contact address does not match work address (1=different, 0=same, at region level)            |
| REG_CITY_NOT_LIVE_CITY        | Flag if client's permanent address does not match contact address (1=different, 0=same, at city level)         |
| REG_CITY_NOT_WORK_CITY        | Flag if client's permanent address does not match work address (1=different, 0=same, at city level)            |
| LIVE_CITY_NOT_WORK_CITY       | Flag if client's contact address does not match work address (1=different, 0=same, at city level)              |
| ORGANIZATION_TYPE             | Type of organization where client works                                                                        |
| EXT_SOURCE_1                  | Normalized score from external data source                                                                      |
| EXT_SOURCE_2                  | Normalized score from external data source                                                                      |
| EXT_SOURCE_3                  | Normalized score from external data source                                                                      |
| APARTMENTS_AVG                | Normalized information about building where the client lives                                                    |
| BASEMENTAREA_AVG              | Normalized information about building where the client lives                                                    |
| YEARS_BEGINEXPLUATATION_AVG   | Normalized information about building where the client lives                                                    |
| YEARS_BUILD_AVG               | Normalized information about building where the client lives                                                    |
| COMMONAREA_AVG                | Normalized information about building where the client lives                                                    |
| ELEVATORS_AVG                 | Normalized information about building where the client lives                                                    |
| ENTRANCES_AVG                 | Normalized information about building where the client lives                                                    |
| FLOORSMAX_AVG                 | Normalized information about building where the client lives                                                    |
| FLOORSMIN_AVG                 | Normalized information about building where the client lives                                                    |
| LANDAREA_AVG                  | Normalized information about building where the client lives                                                    |
| LIVINGAPARTMENTS_AVG          | Normalized information about building where the client lives                                                    |
| LIVINGAREA_AVG                | Normalized information about building where the client lives                                                    |
| NONLIVINGAPARTMENTS_AVG       | Normalized information about building where the client lives                                                    |
| NONLIVINGAREA_AVG             | Normalized information about building where the client lives                                                    |
| APARTMENTS_MODE               | Normalized information about building where the client lives                                                    |
| BASEMENTAREA_MODE             | Normalized information about building where the client lives                                                    |
| YEARS_BEGINEXPLUATATION_MODE  | Normalized information about building where the client lives                                                    |
| YEARS_BUILD_MODE              | Normalized information about building where the client lives                                                    |
| COMMONAREA_MODE               | Normalized information about building where the client lives                                                    |
| ELEVATORS_MODE                | Normalized information about building where the client lives                                                    |
| ENTRANCES_MODE                | Normalized information about building where the client lives                                                    |
| FLOORSMAX_MODE                | Normalized information about building where the client lives                                                    |
| FLOORSMIN_MODE                | Normalized information about building where the client lives                                                    |
| LANDAREA_MODE                 | Normalized information about building where the client lives                                                    |
| LIVINGAPARTMENTS_MODE         | Normalized information about building where the client lives                                                    |
| LIVINGAREA_MODE               | Normalized information about building where the client lives                                                    |
| NONLIVINGAPARTMENTS_MODE      | Normalized information about building where the client lives                                                    |
| NONLIVINGAREA_MODE            | Normalized information about building where the client lives                                                    |
| APARTMENTS_MEDI               | Normalized information about building where the client lives                                                    |
| BASEMENTAREA_MEDI             | Normalized information about building where the client lives                                                    |
| YEARS_BEGINEXPLUATATION_MEDI  | Normalized information about building where the client lives                                                    |
| YEARS_BUILD_MEDI              | Normalized information about building where the client lives                                                    |
| COMMONAREA_MEDI               | Normalized information about building where the client lives                                                    |
| ELEVATORS_MEDI                | Normalized information about building where the client lives                                                    |
| ENTRANCES_MEDI                | Normalized information about building where the client lives                                                    |
| FLOORSMAX_MEDI                | Normalized information about building where the client lives                                                    |
| FLOORSMIN_MEDI                | Normalized information about building where the client lives                                                    |
| LANDAREA_MEDI                 | Normalized information about building where the client lives                                                    |
| LIVINGAPARTMENTS_MEDI         | Normalized information about building where the client lives                                                    |
| LIVINGAREA_MEDI               | Normalized information about building where the client lives                                                    |
| NONLIVINGAPARTMENTS_MEDI      | Normalized information about building where the client lives                                                    |
| NONLIVINGAREA_MEDI            | Normalized information about building where the client lives                                                    |
| FONDKAPREMONT_MODE            | Normalized information about building where the client lives                                                    |
| HOUSETYPE_MODE                | Normalized information about building where the client lives                                                    |
| TOTALAREA_MODE                | Normalized information about building where the client lives                                                    |
| WALLSMATERIAL_MODE            | Normalized information about building where the client lives                                                    |
| EMERGENCYSTATE_MODE           | Normalized information about building where the client lives                                                    |
| OBS_30_CNT_SOCIAL_CIRCLE      | How many observation of client's social surroundings with observable 30 DPD (days past due) default               |
| DEF_30_CNT_SOCIAL_CIRCLE      | How many observation of client's social surroundings defaulted on 30 DPD (days past due)                          |
| OBS_60_CNT_SOCIAL_CIRCLE      | How many observation of client's social surroundings with observable 60 DPD (days past due) default               |
| DEF_60_CNT_SOCIAL_CIRCLE      | How many observation of client's social surroundings defaulted on 60 (days past due) DPD                         |
| DAYS_LAST_PHONE_CHANGE        | How many days before application did client change phone                                                        |
| FLAG_DOCUMENT_2               | Did client provide document 2                                                                                   |
| FLAG_DOCUMENT_3               | Did client provide document 3                                                                                   |
| FLAG_DOCUMENT_4               | Did client provide document 4                                                                                   |
| FLAG_DOCUMENT_5               | Did client provide document 5                                                                                   |
| FLAG_DOCUMENT_6               | Did client provide document 6                                                                                   |
| FLAG_DOCUMENT_7               | Did client provide document 7                                                                                   |
| FLAG_DOCUMENT_8               | Did client provide document 8                                                                                   |
| FLAG_DOCUMENT_9               | Did client provide document 9                                                                                   |
| FLAG_DOCUMENT_10              | Did client provide document 10                                                                                  |
| FLAG_DOCUMENT_11              | Did client provide document 11                                                                                  |
| FLAG_DOCUMENT_12              | Did client provide document 12                                                                                  |
| FLAG_DOCUMENT_13              | Did client provide document 13                                                                                  |
| FLAG_DOCUMENT_14              | Did client provide document 14                                                                                  |
| FLAG_DOCUMENT_15              | Did client provide document 15                                                                                  |
| FLAG_DOCUMENT_16              | Did client provide document 16                                                                                  |
| FLAG_DOCUMENT_17              | Did client provide document 17                                                                                  |
| FLAG_DOCUMENT_18              | Did client provide document 18                                                                                  |
| FLAG_DOCUMENT_19              | Did client provide document 19                                                                                  |
| FLAG_DOCUMENT_20              | Did client provide document 20                                                                                  |
| FLAG_DOCUMENT_21              | Did client provide document 21                                                                                  |
| AMT_REQ_CREDIT_BUREAU_HOUR   | Number of enquiries to Credit Bureau about the client one hour before application                                |
| AMT_REQ_CREDIT_BUREAU_DAY    | Number of enquiries to Credit Bureau about the client one day before application (excluding one hour before application) |
| AMT_REQ_CREDIT_BUREAU_WEEK   | Number of enquiries to Credit Bureau about the client one week before application (excluding one day before application) |
| AMT_REQ_CREDIT_BUREAU_MON    | Number of enquiries to Credit Bureau about the client one month before application (excluding one week before application) |
| AMT_REQ_CREDIT_BUREAU_QRT    | Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application) |
| AMT_REQ_CREDIT_BUREAU_YEAR   | Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application)   |

# 📋 Project Workflow

This project will be structured into three segments to ensure a systematic and accurate analysis.

###  🏃‍♂️ Sprint 1: 

This phase will involve a preliminary study of the dataset, basic data cleaning, exploratory data analysis (EDA), and some initial feature engineering.

###  🚀 Sprint 2: 

In this stage, we will conduct advanced exploratory data analysis, comprehensive feature engineering, preprocessing, and build baseline models to establish a foundation for advanced modeling. Additionally, we will develop a model evaluation framework that accurately reflects the practical use case of our models.

###  🏁 Sprint 3: 

Building upon the baseline models and evaluation framework developed in Sprint 2, we will delve into advanced model building, optimization, evaluation, and interpretation.

# 🤔 Key Questions to be Answered (which are subject to change as the project advances)

* What is the distribution of home loan defaulters across different demographic groups (e.g., gender, age, number of children)? <br>
* Are there any observable patterns or correlations between the borrower's income level and the likelihood of defaulting on their home loan? <br>
* Do individuals who own cars or real estate properties have a lower probability of defaulting compared to those who don't? <br>
* How does the loan amount (AMT_CREDIT) and the associated annuity (AMT_ANNUITY) affect the likelihood of default? <br>
* Is there a relationship between the value of the goods purchased with the loan (AMT_GOODS_PRICE) and the probability of default? <br>
* Can we identify any significant predictors of default risk through exploratory data analysis (EDA) and feature engineering? <br>
* How accurately can machine learning models predict the likelihood of default based on the available features in the dataset? <br>
* Which machine learning algorithms perform best for predicting home loan defaults, and what features contribute most to their predictive power? <br>
* Are there any potential biases in the dataset that could affect the performance and fairness of the predictive models? <br>
* Can patterns in the applicant's employment history predict the likelihood of defaulting on a home loan?<br>
* Do certain types of housing situations or family statuses affect the probability of default?<br>
* Are there any correlations between the condition or characteristics of the property and the risk of default?<br>


