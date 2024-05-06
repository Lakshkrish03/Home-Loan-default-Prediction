# 💰 Predicting Home Loan Default - Using Machine Learning 

<center>
  
[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

</center>

![Alt Text](https://github.com/Lakshkrish03/House-Loan-default-Prediction/blob/main/Home_Loan_Defaults_pic.jpg?raw=true)

# 🚀 Introduction

🏠 Navigating the Risks of Home Loans: A Common man's Perspective 🏠

Imagine you're setting off on a journey to buy your dream home. You've saved up, found the perfect place, and secured a loan to make it all happen. But what if, somewhere down the road, you hit a financial bump and struggle to keep up with your mortgage payments? This is where the challenge of home loan defaults comes into play, affecting millions of people around the world.

Understanding Home Loan Defaults:

Home loan defaults happen when folks, like you, who've taken out a mortgage, find themselves unable to keep up with their repayments. It's like getting stuck in a financial pothole on the road to homeownership. When this happens, it's not just the borrowers who face tough times. Lenders, the banks or companies that loaned the money, also get into hot water because they're counting on those repayments to keep their own finances healthy.

Why Predictive Models Matter:

Now, imagine if there was a way to predict these bumps in the road before they even happen. That's where predictive models come in. They're like financial forecasters, analyzing loads of data to spot the warning signs of potential loan defaults. By looking at things like your income, credit history, and even broader economic trends, these models can give lenders a heads-up when someone might be at risk of falling behind on their payments.

### 📉 Some Real-World facts on Home Loan defaults around the world:

Here are some real-world facts on home loan defaults around the world, along with references:

United States: During the 2008 financial crisis, the U.S. experienced a significant increase in home loan defaults, leading to widespread foreclosures and economic turmoil. According to the Federal Reserve Bank of St. Louis, the mortgage delinquency rate in the U.S. peaked at 11.5% in 2010, affecting millions of homeowners (source : https://www.stlouisfed.org/on-the-economy/2021/may/mortgage-distress-great-recession).

Australia: In recent years, Australia has seen a rise in mortgage stress, with a significant portion of households struggling to meet their home loan repayments. According to the Roy Morgan Mortgage Stress research, around 30.3% of Australian households were experiencing mortgage stress in in the later half of 2023 (source : https://www.roymorgan.com/findings/mortgage-stress-risk-october-2023).

United Kingdom: The UK housing market has also faced challenges with home loan defaults. According to data from UK Finance, mortgage arrears and possessions remained low in 2020 due to government support measures during the COVID-19 pandemic. However, concerns about future defaults persist as support measures are phased out (source : https://www.ukfinance.org.uk/sites/default/files/uploads/Data%20%28XLS%20and%20PDF%29/Household-Finance-Review-2020-Q4-FINAL.pdf).

Canada: Canada runs the highest risk of mortgage defaults among advanced economies, the International Monetary Fund warns, while other reports show Canadians are increasingly struggling with debt. According to Financial Post, Posthaste: More borrowers struggle as IMF warns Canada at highest risk of mortgage defaults (source : https://financialpost.com/news/imf-warns-canada-highest-risk-mortgage-defaults)

______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

Let's delve into the core aspects of our project: the problem area, the Users involved, the main concept, and the potential impact. By examining these elements in detail, we can gain a better understanding of how our machine learning models aim to address these issues effectively and cater to the needs of the users.

We'll start by thoroughly exploring the problem area we're addressing, followed by an examination of the Users who will benefit from our project. Then, we'll discuss the overarching concept behind our machine learning models and how they are designed to provide valuable insights for the users.

The dataset we have isn't made up of current data but can still be really useful for building better machine learning models.

#  🏠 Problem Area

Home loan default prediction involves the task of assessing the likelihood that a borrower will default on their mortgage loan based on various factors and historical data. When individuals apply for home loans, lenders evaluate their creditworthiness to determine the risk associated with lending them money. Default occurs when a borrower fails to make scheduled mortgage payments for say more than 30 or 90 days, leading to potential financial losses for the lender. The problem area revolves around accurately predicting the probability of default for loan applicants, which is essential for lenders to manage credit risk effectively, optimize loan approval processes, and maintain a healthy loan portfolio. Challenges in home loan default prediction include analyzing complex and high-dimensional data, identifying relevant features that impact default risk, handling imbalanced datasets, and ensuring model interpretability and transparency in lending decisions. By addressing these challenges, this project aims to developpredictive models that can assist lenders in making more informed and prudent lending decisions, ultimately reducing the incidence of loan defaults and improving overall financial stability in the housing finance market.

# 👥 Users 

The users of home loan default prediction models primarily include financial institutions, banks, mortgage lenders, and credit unions involved in the lending process. Additionally, borrowers seeking home loans are indirectly impacted by the outcomes of these models. Lenders rely on these predictive models to assess the creditworthiness of loan applicants and determine the risk associated with extending mortgage loans. By accurately predicting the likelihood of loan default, lenders can make more informed and prudent lending decisions, optimizing their loan approval processes and minimizing potential losses due to defaults. Borrowers, on the other hand, benefit from fair and transparent lending practices that consider their creditworthiness and financial stability when evaluating loan applications. Through, the use of home loan default prediction models, borrowers may have increased access to credit, as lenders can more accurately assess risk and offer competitive loan terms to eligible applicants. Overall, the users of home loan default prediction models are integral to the lending ecosystem, shaping the accessibility and affordability of homeownership while ensuring the financial stability of lending institutions.

# 💡 Big Idea 

The concept of home loan default prediction involves using advanced machine learning algorithms to accurately predict the likelihood of borrowers defaulting on their mortgage loans. This is important for financial institutions, banks, and mortgage lenders as it helps them manage credit risk, optimize lending decisions, and maintain a healthy loan portfolio. By analyzing historical data of loan applicants such as their demographics, credit scores, income levels, employment history, and loan characteristics, machine learning models can identify patterns and trends associated with loan defaults. The aim is to develop predictive models that can effectively assess credit risk, enabling lenders to differentiate between low-risk and high-risk borrowers and make informed lending decisions.

The key innovation behind this concept is the use of sophisticated algorithms, such as logistic regression, decision trees, random forests, gradient boosting, and neural networks, to process large and complex datasets and generate predictions. Lenders can then use these models to evaluate the creditworthiness of loan applicants, optimize loan approval processes, and mitigate the risk of default in their loan portfolios.Machine learning models can handle large volumes of data and can adapt and improve over time as new data becomes available. This adaptability is particularly valuable in dynamic and evolving lending environments where borrower behaviors and economic conditions may change over time. Additionally, machine learning models can incorporate a wide range of features and data sources, including unstructured data such as text and images, further enhancing their predictive capabilities.

# 🌍 Impact 

Home loan default prediction models have a multifaceted impact, spanning the financial industry, the broader economy, and society. They enhance risk management practices for lenders, reduce financial losses due to loan defaults, and optimize loan portfolios. By promoting fair and transparent lending practices, they increase access to credit, promote homeownership and wealth accumulation, and contribute to the stability and sustainability of the housing market. Additionally, they foster confidence and trust in the lending industry, promote consumer protection and financial inclusion, and reduce systemic risks, contributing to economic stability and growth.

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
| AMT_GOODS_PRICE               | For consumer loans it is the price of the goods for which the loan is given                                      |
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
