
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from scipy import stats



img = ("images123456.png")



st.subheader("The goal of this project is to predict sales and demand accurately for Walmart. To achieve this, we will follow these steps:")

st.text("""

1. **Understanding the dataset:**
   - Examine the available features, such as weekly sales, whether the week is a holiday or not, temperature, fuel price, consumer price index (CPI), and unemployment rate.
   - Check for missing values or inconsistent data that need to be cleaned.

2. **Exploratory Data Analysis (EDA):**
   - Visualize the distributions of variables and look for relationships between them.
   - Identify patterns, trends, and potential outliers.

3. **Data preparation:**
   - Encode the "IsHoliday" variable (1 for holiday weeks and 0 for non-holiday weeks).
   - Consider normalizing or standardizing numerical features if necessary.

4. **Regression modeling:**
   - Build regression models to predict weekly sales.
   - You can start with a simple linear regression model using a single feature (e.g., temperature) and then progress to more complex models.

5. **Model evaluation:**
   - Use metrics like R2 (coefficient of determination) and RMSE (root mean square error) to evaluate model performance.
   - Compare results from different models to select the best one.
        """)


st.html("""
<h3>Here are the steps for project execution and planning:</h3>
""")

st.text("""
        1. **Define the project scope and objectives.**
2. **Plan resource management.**
3. **Organize task management.**
4. **Coordinate tasks and resources.**
5. **Schedule time management and improve predictability.**
6. **Evaluate and enhance the project.**
        """)

# Loading Dataset
df_store = pd.read_csv('stores.csv') #store data
df_train = pd.read_csv('train.csv') # train set
features = pd.read_csv("features.csv") #feature data
sample   = pd.read_csv("sampleSubmission.csv") #sample data
test     = pd.read_csv("test.csv") #test data


st.sidebar.image(img)

# Data Exploration (EDA)
# Sidebar Stadistics
##################Store#####################
if st.sidebar.checkbox("Store"):
       st.header("Store")
       st.dataframe(df_store.head())

       st.text("""
        1. **Store**: This column represents the store identifier or number. 
               Each store has a unique value in this column.
        2. **Type**: The "Type" column indicates the type of store.
        3. **Size**: The "Size" column generally refers to the physical size of the store.
        """)
       
       st.write(f"Number of records: {len(features)}")
       st.write(f"Number of columns: {len(features.columns)}")
       
       st.html("<h3>Check for null values</h3>")
       # Check for null values
       st.write(df_store.isnull().sum())

       col1, col2 = st.columns(2)
       with col1:
          st.html("<h3>Describe Store</h3>")
          st.write(df_store.describe())

          plt.figure(figsize=(8, 6))
          sns.countplot(data=df_store, x='Type')
          plt.xlabel('Type Store')
          plt.ylabel('Cantidad')
          plt.title('Distribución de tipos de tienda')
          st.pyplot(plt)

       with col2:
            st.html("<h3>Data types of the columns</h3>")
            # Data types of the columns
            st.write(df_store.dtypes)
            

            df_store['Type'].nunique() # number of distinct categories present in that column

            fig               = plt.figure(figsize=(5, 2))
            sns.histplot(data = df_store, x="Store", hue="Type", multiple="stack")
            st.subheader("Distinct types of Store")
            st.pyplot(plt)
            
            numeric_columns    = df_store[['Store', 'Size']] #This means that the columns containing numerical data are selected.
            correlation_matrix = numeric_columns.corr() #The correlation matrix between the ‘Store’ and ‘Size’ columns is calculated.
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Matriz de correlación (Store y Size)')
            st.pyplot(plt)


          



################Features#####################
if st.sidebar.checkbox("Features"):
       st.header("Features")
       st.dataframe(features.head())
       
       col3, col4 = st.columns(2)



       st.text("""   

- **Store**: This column represents the identifier or store number for Walmart. 
               Each store would have a unique value in this column.
- **Date**: The "Date" column indicates the date to which the data entry refers. 
               It can be useful for analyzing trends over time.
- **Temperature**: The "Temperature" column represents the temperature at the store's location.
                It can impact sales, as purchasing behavior may vary based on the weather.
- **Fuel_Price**: The "Fuel_Price" column refers to the fuel price in the region. This can also influence customer buying behavior.
- **MarkDown1, MarkDown2, MarkDown3, MarkDown4, MarkDown5**: These columns likely represent specific discounts or promotions applied in the store. 
               The values in these columns indicate the amount of the discounts.
- **CPI (Consumer Price Index)**: The Consumer Price Index is a measure of inflation.
                It can affect consumers' purchasing power and, consequently, sales.
- **Unemployment**: The "Unemployment" column represents the unemployment rate in the region. 
               This can also impact consumer spending.
- **IsHoliday**: The "IsHoliday" column is a binary value indicating whether the day corresponds to a holiday or not.
                It can affect sales due to changes in buying behavior during holidays¹.
               """)
       
       st.write(f"Número de registros: {len(features)}")
       st.write(f"Número de columnas: {len(features.columns)}")


       st.header("Decriptive statistics")
       st.write(features.describe()) #generate descriptive statistics

       st.write("Missing Values")
       st.write(features.isnull().sum()) #missing values
        
       with col3:
             
           st.html("<h3>Data types of the column</h3>")
           st.write(features.dtypes) # Data types of the columns

       with col4:
           st.header("Column information:")
           st.write(f"Number of unique stores: {features['Store'].nunique()}")
           st.write(f"Date range: {features['Date'].min()} to {features['Date'].max()}")
           st.write(f"Average temperature: {features['Temperature'].mean():.2f}°C")
           st.write(f"Average fuel price: {features['Fuel_Price'].mean():.2f}")
           st.write(f"Average MarkDown1 discount: {features['MarkDown1'].mean():.2f}")

           feature_store = features.merge(df_store, how="inner", on = "Store").copy()
           
           st.write("Features Stores after merge")
           st.write(feature_store.head())



feature_store = features.merge(df_store, how="inner", on = "Store").copy()


#####DF Train#########
if st.sidebar.checkbox("Train"):
       st.header("Train")
       st.dataframe(df_train.head())
       
       col5, col6 = st.columns(2)
       train_df   = df_train.merge(feature_store, how="inner", on=['Store', 'Date', 'IsHoliday'])\
       .sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True).copy()

       
       st.html('<h3>DF Train after merge</h3>')
       st.write(train_df.head()) #Show the first rows of the dataframe

       st.html("<h3>Descriptive Statistics</h3>")
       train_df.describe() #generate descriptive statistics

       st.write(train_df.dtypes)


#These lines of code convert the date columns in the mentioned DataFrames 
#into datetime objects 
#for easier date analysis and manipulation. 
feature_store['Date'] = pd.to_datetime(feature_store['Date'])
df_train['Date']      = pd.to_datetime(df_train['Date'])
test['Date']          = pd.to_datetime(test['Date'])


#These lines of code enrich the feature_store DataFrame with information about the day, 
#week, month, and year of the dates present in the “Date” column.
feature_store['Day']    = feature_store['Date'].dt.isocalendar().day
feature_store['Week']   = feature_store['Date'].dt.isocalendar().week
feature_store['Month']  = feature_store['Date'].dt.month
feature_store['Year']   = feature_store['Date'].dt.isocalendar().year

#these lines of code combine the df_train and feature_store DataFrames through an inner join, 
#sort the result, and create independent copies for the training and test datasets.
train_df = df_train.merge(feature_store, how='inner', on = ['Store','Date','IsHoliday'])\
    .sort_values(by=['Store','Dept','Date']).reset_index(drop=True).copy()
test_df = test.merge(feature_store, how='inner', on = ['Store','Date','IsHoliday'])\
    .sort_values(by=['Store','Dept','Date']).reset_index(drop=True).copy()   

#df_weeks will contain the sum of the numeric columns for each week in the original 
#train_df DataFrame.
df_weeks = train_df.groupby('Week').sum(numeric_only=True)

##############Df Weeks##############
if st.sidebar.checkbox("Weeks"):
     st.html("<h3>Weeks Sales</h3>")
     st.write(df_weeks.head())  #Show the first rows of the dataframe

     st.subheader("Descriptive Statistics")
     st.write(df_weeks.describe()) #generate descriptive statistics


     st.subheader("Weekly Sales per Temperature ")
     st.write(df_weeks[["Weekly_Sales", "Temperature"]].describe())

     #To calculate the average of sales for holidays and non-holidays separately, 
     #you can use the following code: df_weeks.groupby("IsHoliday")["Weekly_Sales"].mean(). 
     st.subheader("Calculate the average of sales for holidays and non-holidays")
     st.write(df_weeks.groupby("IsHoliday")["Weekly_Sales"].mean()) 

     # Sum of markdowns
     markdown_totals = df_weeks[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].sum()

     # Create a pie chart
     
     plt.pie(markdown_totals, labels=markdown_totals.index, autopct='%1.1f%%')
     plt.title('Distribution of Markdowns')
     st.pyplot(plt)



     
      # Weekly Sales per Year with Week
      ################Sales################
if st.sidebar.checkbox("Sales"):
     weekly_sales_by_week      = train_df.groupby(by=["Year", "Week"], as_index=False).agg({"Weekly_Sales" : ["mean", "median"]}).copy()
     weekly_sales_by_week_2010 = weekly_sales_by_week.loc[weekly_sales_by_week["Year"] == 2010].copy()
     weekly_sales_by_week_2011 = weekly_sales_by_week.loc[weekly_sales_by_week["Year"] == 2011].copy()
     weekly_sales_by_week_2012 = weekly_sales_by_week.loc[weekly_sales_by_week["Year"] == 2012].copy()

     weekly_sales_by_week_2010 = weekly_sales_by_week_2010.reset_index(drop=True)
     weekly_sales_by_week_2011 = weekly_sales_by_week_2011.reset_index(drop=True)
     weekly_sales_by_week_2012 = weekly_sales_by_week_2012.reset_index(drop=True)

     weekly_sales_by_week = pd.concat([weekly_sales_by_week_2010, weekly_sales_by_week_2011, weekly_sales_by_week_2012], ignore_index=True)
     st.subheader("Sales per Week")
     st.write(weekly_sales_by_week.head())
      
     # Weekly Sales per Year

     weekly_sales_by_year = train_df.groupby(by=["Year"], as_index=False).agg({"Weekly_Sales" : ["mean", "median"]}).copy()

     weekly_sales_by_year_2010 = weekly_sales_by_year.loc[weekly_sales_by_year["Year"] == 2010].copy()
     weekly_sales_by_year_2011 = weekly_sales_by_year.loc[weekly_sales_by_year["Year"] == 2011].copy()
     weekly_sales_by_year_2012 = weekly_sales_by_year.loc[weekly_sales_by_year["Year"] == 2012].copy()

     weekly_sales_by_year_2010 = weekly_sales_by_year_2010.reset_index(drop=True)
     weekly_sales_by_year_2011 = weekly_sales_by_year_2011.reset_index(drop=True)
     weekly_sales_by_year_2012 = weekly_sales_by_year_2012.reset_index(drop=True)

     weekly_sales_by_year = pd.concat([weekly_sales_by_year_2010, weekly_sales_by_year_2011, weekly_sales_by_year_2012], axis=0)
     st.subheader("Sales Per Year")
     st.write(weekly_sales_by_year.head())


     #These lines of code calculate the average weekly sales for each month.
     st.subheader("Average weekly sales for each Month")
     monthly_sales_by_month = train_df.groupby(by=["Month"], as_index=False)["Weekly_Sales"].mean()
     st.write(monthly_sales_by_month.head())




     plt.figure(figsize=(4, 2))
     plt.plot(monthly_sales_by_month.index, monthly_sales_by_month.values, marker='o')
     plt.xlabel('Month')
     plt.ylabel('Sales average per weeks')
     plt.title('Sales average per weeks by month')
     plt.grid()

     st.pyplot(plt)

     train_df_numeric = train_df.select_dtypes(include='number')
     
     corr_matrix = train_df_numeric.corr()

     plt.figure(figsize=(16, 12))
     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

     st.pyplot(plt)
     plt.figure(figsize=(6, 2))
     sns.histplot(train_df['Weekly_Sales'], color='g', kde=True, bins=100)
     plt.title('Distribution of the target variable (Weekly_Sales)')
     plt.xlabel('Sales Weeks')
     plt.ylabel('Frecuency')
     plt.grid()
     st.pyplot(plt)

     
     grid = sns.catplot(data=train_df, x="Type", y="Weekly_Sales", hue="Year", kind="bar")
     plt.title('Weekly Sales per type and year')
     plt.xlabel('Type')
     plt.ylabel('Weekly Sales')
     plt.grid()

     st.pyplot(grid)


     
     grid = sns.catplot(data=train_df, x="Year", y="Weekly_Sales", hue="Month", kind="bar")
     plt.title('Weekly_Sales per month and year')
     plt.xlabel('Year')
     plt.ylabel('Weekly_Sales')
     plt.grid()

   
     st.pyplot(grid)

     
     plt.figure(figsize=(10, 5))

     sns.barplot(x=train_df.Store, y=train_df.Weekly_Sales)

     st.pyplot(plt)


     plt.figure(figsize=(11, 6))

   
     sns.lineplot(x='Date', y='Weekly_Sales', data=train_df, hue='IsHoliday')


     st.pyplot(plt)


    # Grouping Data by Year

     growth                          = train_df.copy()
     growth['Date']                  = pd.to_datetime(growth.Date,format='%d-%m-%Y')
     growth['Year'], growth['Month'] = growth['Date'].dt.year, growth['Date'].dt.month


#let's Group the data.

     hypothesis = growth.groupby('Store')[['Fuel_Price','Unemployment', 'CPI','Weekly_Sales', 'IsHoliday']]
     factors    = hypothesis.get_group(1)
     day_arr    = [1]
     for i in range (1,len(factors)):
        day_arr.append(i*7)
    
     factors['Day'] = day_arr.copy()



     plt.figure(figsize=(11, 6))

     st.header("Distribution  CPI")

     st.subheader("Scatter Plot")
     sns.scatterplot(x='CPI', y='Weekly_Sales', data=factors, hue='IsHoliday')
     st.pyplot(plt)

     st.subheader("Linear Regression Plot")
     sns.lmplot(x='CPI', y='Weekly_Sales', data=factors, hue='IsHoliday')
     st.pyplot(plt)




     plt.figure(figsize=(11, 6))

     st.header("Distribution  Fuel_Price")
     

     st.subheader("Scatter Plot")
     sns.scatterplot(x='Fuel_Price', y='Weekly_Sales', data=factors, hue='IsHoliday')
     st.pyplot(plt)

 
     st.subheader("Linear Regression Plot")
     sns.lmplot(x='Fuel_Price', y='Weekly_Sales', data=factors, hue='IsHoliday')
     st.pyplot(plt)

 



     plt.figure(figsize=(11, 6))

     st.header("Distribution  Unemployment")

     

     st.subheader("Scatter Plot")
     sns.scatterplot(x='Unemployment', y='Weekly_Sales', data=factors, hue='IsHoliday')
     st.pyplot(plt)

 
     st.subheader("Linear Regression Plot")
     sns.lmplot(x='Unemployment', y='Weekly_Sales', data=factors, hue='IsHoliday')
     st.pyplot(plt)

   

     plt.figure(figsize=(16, 9))

   
     sns.barplot(x='Day', y='Weekly_Sales', data=factors.head(50), hue='IsHoliday')


     st.pyplot(plt)





  







    

 




  


       

       


           
          

    



