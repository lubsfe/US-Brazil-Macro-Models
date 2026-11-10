import requests
import pandas as pd
from datetime import datetime
import statsmodels.api as sm

# Defining API Keys and URLs
API_Key = '4202887bc7d8cffbe8a266f705805656'
Series_ID = 'FEDFUNDS' #Federal Reserve Rates
URL = 'https://api.stlouisfed.org/fred/series/observations'

# Using Dictionary to Set Up the Request Parameters
params = {
    'series_id': Series_ID,
    'api_key': API_Key,
    'file_type' : 'json',
    'frequency' : 'm',
    'observation_start' : '2015-01-01'
}

response = requests.get(URL, params=params)
#print(response.status_code)

# Extracting the JSON data from response
JSON_data = response.json()

# Loading the observed lists into a pandas DataFrama
df = pd.DataFrame(JSON_data['observations'])
#print(df.head())

""" Cleaning the dataframe
Dropping extra columns, and converting text strings to proper pandas date and number format, 
and setting the date as the index.
"""

# Keeping only 'date' and 'value' columns
df = df[['date', 'value']]

# Converting dates to pandas datetime objects
df['date'] = pd.to_datetime(df['date'])

# Converting the interest rate values into numeric floats and adding 'coerce' to turn any missing/bad data into NaN
df['value'] = pd.to_numeric(df['value'], errors='coerce')

# Dropping any rows with missing values
df = df.dropna()

# Setting the date column as the index for time-series alignment
df = df.set_index('date')

# Printing the dateframe and its data types
#print("\nCleaned FRED Data:")
#print(df.head())
#print("\nData Types:")
#print(df.info())

# Calculating a dynamic 9-year rolling window
end_date = datetime.today()
start_date = end_date.replace(year=end_date.year - 9)

# Defining the BCB endpoint and parameters
url_bcb = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados'
params_bcb = {
    'dataInicial': start_date.strftime('%d/%m/%Y'),
    'dataFinal': end_date.strftime('%d/%m/%Y')
}

response = requests.get(url_bcb, params=params_bcb)
#print(response.status_code)

# Extracting the JSON data
data_bcb = response.json()
df_bcb = pd.DataFrame(data_bcb)

# Converting the Portuguese 'data' column to pandas datetime (parsing day/month/year)
df_bcb['data'] = pd.to_datetime(df_bcb['data'], format='%d/%m/%Y')

# Converting the 'valor' column from text to numeric floats
df_bcb['valor'] = pd.to_numeric(df_bcb['valor'])

# Renaming columns to English so they make sense alongside my FRED data
df_bcb = df_bcb.rename(columns={'data': 'date', 'valor': 'selic_rate'})

# Setting the date as the index
df_bcb = df_bcb.set_index('date')

#print("Cleaned BCB Selic Data:")
#print(df_bcb.head())
#print("\nData Types:")
#print(df_bcb.info())

# Resample the daily BCB data to a monthly start ('MS') average
df_bcb_monthly = df_bcb.resample('MS').mean()

# Merging the FRED (df) and BCB (df_bcb_monthly) dataframes on their date indexes
df_master = pd.merge(df, df_bcb_monthly, left_index=True, right_index=True, how='inner')

# Renaming the FRED 'value' column to be descriptive
df_master = df_master.rename(columns={'value': 'fed_funds_rate'})

#print("\n--- MASTER MACROECONOMIC DATASET ---")
#print(df_master.head())
#print("\nFinal Data Info:")
#print(df_master.info())

# Defining my independent variable (X - the US rate) and dependent variable (y - the Brazilian rate)
X = df_master['fed_funds_rate']
y = df_master['selic_rate']

# statsmodels requires me to manually add a constant (y-intercept) to your model
X = sm.add_constant(X)

# Fitting the Ordinary Least Squares (OLS) linear regression model
model = sm.OLS(y, X).fit()

# Printing the comprehensive statistical summary
print("\n--- LINEAR REGRESSION RESULTS ---")
print(model.summary())