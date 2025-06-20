import pandas as pd
import pandas_datareader.wb as wb
import yfinance as yf
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

import stockplayground as cntry
import numpy as np
import datetime
import time

from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.nn as nn
import torch.optim as optim


economic_or_fin = input("Global economic data or stock data? 'economic' or 'stock'")
if economic_or_fin == "stock":
    user_symbol = input("What stock symbol would you like to enter? (NASDAQ/NYSE) For example, type 'AAPL' or 'TSLA'. ")
    stock2_orno = input("Type another stock symbol if you like. If no, type no. If yes, type yes.")
    if stock2_orno == "yes":
        user_symbol2 =input("Type another stock ticker symbol. This will just be graphed alongside the primary stock.")
    else:
        pass
    user_time = input("Do you want a custom graph? If yes, type 'yes'.  "
                      "For the past year, type 'y'. "
                      "For the past month, type 'm'. "
                      "For the most recent quarter, type 'q'. ")
    try:
        if user_time == 'yes':
            user_startyear = int((input("Please enter the year you want the graph to start. ")))
            user_startmonth = int((input("Please enter the month (number) you want the graph to start. ")))
            user_startday = int((input("Please enter the day (number) you want the graph to start. ")))
            user_endyear = int((input("Please enter the year you want the graph to end. ")))
            user_endmonth = int((input("Please enter the month (number) you want the graph to end. ")))
            user_endday = int((input("Please enter the day (number) you want the graph to end. ")))
            start = datetime.datetime(user_startyear, user_startmonth, user_startday)
            end = datetime.datetime(user_endyear, user_endmonth, user_endday)
            df = yf.download(user_symbol, 'yahoo', start, end)
            df2 = yf.download(user_symbol2, 'yahoo', start, end)
            time.sleep(1)
        elif user_time == 'y':
            end = datetime.datetime.today()

            try:
                start = end.replace(year=end.year - 1)
            except ValueError:
                start = end.replace(year=end.year - 1, day=28)
            df = yf.download(user_symbol, start, end)
            df2 = yf.download(user_symbol2, start, end)
            time.sleep(1)
        elif user_time == 'm':
            end = datetime.datetime.today()
            start = end - relativedelta(months=1)
            df = yf.download(user_symbol, start, end)
            df2 = yf.download(user_symbol2, start, end)
            time.sleep(1)
        else:
            end = datetime.datetime.today()
            start = end - relativedelta(months=3)
            df = yf.download(user_symbol,start, end)
            df2 = yf.download(user_symbol2,start,end)
            time.sleep(1)

    except Exception as ratelimit:
        print(f"Sorry, the Pandas Datareader is having a slight issue.{ratelimit}")
        user_etf = input(
            "Would you like to see a financial ETF (XLF) or a tech ETF (QQQ)? Please enter the primary ticker symbol.")
        if user_etf == "QQQ":
            df = pd.read_csv('Download Data - FUND_US_XNAS_QQQ.csv', index_col=0, parse_dates=True)
            df2 = pd.read_csv('Download Data - FUND_US_ARCX_XLF.csv', index_col=0, parse_dates=True)
            user_symbol = 'QQQ'
        else:
            df = pd.read_csv('Download Data - FUND_US_ARCX_XLF.csv', index_col=0, parse_dates=True)
            df2 = pd.read_csv('Download Data - FUND_US_XNAS_QQQ.csv', index_col=0, parse_dates=True)
            user_symbol = 'XLF'


    class Stock:
        def __init__(self, symbol, data):
            self.symbol = symbol
            self.data = data

        def percent_change(self):
            return self.data["Close"].pct_change()

        def standard_deviation(self):
            return self.data["Close"].iloc[-10:].std()

    print(df['Close'].head(10))
    stock = Stock(user_symbol, df)
    print(f'Percent change: {Stock.percent_change(stock)}')
    print(f'Standard deviation: {Stock.standard_deviation(stock)}')
    latest_price = float(df['Close'].iloc[-1])

    range_high = float(df['High'].max())
    range_low = float(df['Low'].min())
    average_close = float(np.mean(df['Close']))

    print(f"The average price is ${average_close:.2f}")
    print(f"The low is ${range_low:.2f}")
    print(f"The high is ${range_high:.2f}")
    print(f"The current price of {user_symbol} is ${latest_price:.2f}")
    df.index = pd.to_datetime(df.index)
    print(df.head())
    plt.plot(df['Close'],label="Primary",color = 'red')
    plt.plot(df2['Close'],label = "Secondary",color = 'blue')
    plt.legend()
    plt.grid(True)
    plt.title("Stocks")
    plt.xticks(rotation=45)
    plt.show()

elif economic_or_fin == "economic":
    start = int(input("What year (####) do you want to start from?"))
    end = int(input("What year (####) do you want to end from?"))
    country = input("What country do you want??")
    country = country.strip().title()
    if country not in cntry.country_dict:
        print(f"Country '{country}' not recognized. Please check the name and try again.")
        exit()
    refined_country = cntry.country_dict[country]
    df_gdp = wb.download(indicator='NY.GDP.MKTP.CD', start=start, end=end, country=refined_country)
    df_gdp_capita = wb.download(indicator='NY.GDP.PCAP.CD', start=start, end=end, country=refined_country)
    df_gdp_growth = wb.download(indicator='NY.GDP.MKTP.KD.ZG', start=start, end=end, country=refined_country)
    df_inflation = wb.download(indicator='FP.CPI.TOTL.ZG', start=start, end=end, country=refined_country)
    df_unemploy = wb.download(indicator='SL.UEM.TOTL.ZS', start=start, end=end, country=refined_country)
    df_population = wb.download(indicator='SP.POP.TOTL', start=start, end=end, country=refined_country)
    df_poverty = wb.download(indicator='SI.POV.DDAY', start=start, end=end, country=refined_country)
    df_life = wb.download(indicator='SP.DYN.LE00.IN', start=start, end=end, country=refined_country)
    df_invest = wb.download(indicator='NE.GDI.TOTL.ZS', start=start, end=end, country=refined_country)
    df_residential = pd.read_csv('WS_SPP_csv_col.csv')

    print(df_residential.head())

    df_gdp_capita = df_gdp_capita.reset_index().drop('country', axis=1)
    df_poverty = df_poverty.reset_index()
    df_poverty = df_poverty.drop(['country'], axis=1)
    df_life = df_life.reset_index()
    df_life = df_life.drop(['country'], axis=1)
    df_gdp_growth = df_gdp_growth.reset_index()
    df_gdp_growth = df_gdp_growth.drop(['country'], axis=1)
    df_inflation = df_inflation.reset_index()
    df_inflation = df_inflation.drop(['country'], axis=1)
    df_population = df_population.reset_index()
    df_population = df_population.drop(['country'], axis=1)
    df_unemploy = df_unemploy.reset_index()
    df_unemploy = df_unemploy.drop(['country'], axis=1)
    df_gdp = df_gdp.reset_index().set_index('year')
    df_invest = df_invest.reset_index()
    df_invest = df_invest.drop(['country'], axis=1)

    df_gdp.rename(columns={'NY.GDP.MKTP.CD': 'GDP'}, inplace=True)
    df_gdp_capita.rename(columns={'NY.GDP.PCAP.CD': 'GDP Per Capita'}, inplace=True)
    df_gdp_growth.rename(columns={'NY.GDP.MKTP.KD.ZG': 'GDP Growth'}, inplace=True)
    df_inflation.rename(columns={'FP.CPI.TOTL.ZG': 'Inflation'}, inplace=True)
    df_unemploy.rename(columns={'SL.UEM.TOTL.ZS': 'Unemployment'}, inplace=True)
    df_population.rename(columns={'SP.POP.TOTL': 'Population'}, inplace=True)
    df_poverty.rename(columns={'SI.POV.DDAY': 'Poverty'}, inplace=True)
    df_invest.rename(columns={'NE.GDI.TOTL.ZS': 'Investment'}, inplace=True)
    df_life.rename(columns={'SP.DYN.LE00.IN': 'Life Expectancy'}, inplace=True)

    df_gdp.index = pd.to_datetime(df_gdp.index, format='%Y', errors='coerce')

    df_gdp_capita.index = pd.to_datetime(df_gdp_capita['year'], format='%Y', errors='coerce')
    df_gdp_capita.drop('year', axis=1, inplace=True)
    df_gdp_growth.index = pd.to_datetime(df_gdp_growth['year'], format='%Y', errors='coerce')
    df_gdp_growth.drop('year', axis=1, inplace=True)
    df_invest.index = pd.to_datetime(df_invest['year'], format='%Y', errors='coerce')
    df_invest.drop('year', axis=1, inplace=True)
    df_inflation.index = pd.to_datetime(df_inflation['year'], format='%Y', errors='coerce')
    df_inflation.drop('year', axis=1, inplace=True)
    df_unemploy.index = pd.to_datetime(df_unemploy['year'], format='%Y', errors='coerce')
    df_unemploy.drop('year', axis=1, inplace=True)
    df_population.index = pd.to_datetime(df_population['year'], format='%Y', errors='coerce')
    df_population.drop('year', axis=1, inplace=True)
    df_poverty.index = pd.to_datetime(df_poverty['year'], format='%Y', errors='coerce')
    df_poverty.drop('year', axis=1, inplace=True)
    df_life.index = pd.to_datetime(df_life['year'], format='%Y', errors='coerce')
    df_life.drop('year', axis=1, inplace=True)

    df_gdp = df_gdp.select_dtypes(exclude=['object', 'string'])

    df_economy = df_gdp.join(
        [df_gdp_capita, df_poverty, df_life, df_gdp_growth, df_inflation, df_population, df_unemploy, df_invest],
        how='outer')
    df_economy1 = df_economy[['GDP Per Capita', 'GDP Growth', 'Inflation', 'Unemployment']]
    df_economy2 = df_economy.drop(['GDP Per Capita', 'GDP Growth', 'Inflation', 'Unemployment'], axis=1)
    df_gdp.plot(xlabel='Time',ylabel='GDP',title='GDP Over Time',grid=True,legend=False,rot=45)
    df_population.plot(xlabel='Time',ylabel='Population In Millions/Billions',title='Population Over Time',grid=True,legend=False,rot=45)
    df_economy3 = df_economy.drop(['GDP','Population'], axis=1)
    df_economy3.plot(title='Economic Data',legend=True,grid=True,rot=45)

    df_houseprice = pd.read_csv('average_house_prices.csv')
    region_values = {"North": 1.0,
                     "South": 0.9,
                     "East": 1.1,
                     "West": 1.05,
                     "Central": 1.2}
    df_houseprice['region'] = df_houseprice['region'].map(region_values)
    X = df_houseprice[
        ['region', 'average_income', 'crime_rate', 'population_density', 'school_quality_index', 'unemployment_rate']]
    y = df_houseprice['avg_house_price']

    df_clean = pd.concat([X, y], axis=1).dropna()

    X = df_clean[
        ['average_income', 'unemployment_rate', 'school_quality_index', 'population_density', 'crime_rate', 'region']]
    y = df_clean['avg_house_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    polypolyrf = RandomForestRegressor(n_estimators=100)

    polypolyrf.fit(X_train, y_train)

    y_predict = polypolyrf.predict(X_test)
    avg_income_min = float(X['average_income'].min())
    avg_income_max = float(X['average_income'].max())
    X_df = pd.DataFrame({
        'average_income': np.linspace(avg_income_min, avg_income_max, 100),
        'unemployment_rate': [X['unemployment_rate'].mean().item()] * 100,
        'school_quality_index': [X['school_quality_index'].mean().item()] * 100,
        'population_density': [X['population_density'].mean().item()] * 100,
        'crime_rate': [X['crime_rate'].mean().item()] * 100,
        'region': [X['region'].mean().item()] * 100

    })

    X_plot = X_df['average_income']
    y = y.iloc[:100]

    y_df = polypolyrf.predict(X_df)
    y_plot = polypolyrf.predict(X_df)
    plt.plot(X_plot, y_plot, linewidth=2, color="green", label="Random Forest Regression")
    plt.scatter(X_plot, y)

    plt.xlabel('Avg Income')
    plt.ylabel('Average House Price')
    plt.show()
    mse = mean_squared_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    print(f"The mean squared error is: {mse:.4f}")
    print(f"The R^2 score is: {r2:.4f}")
    #X = pd.DataFrame([{
        #"average_income": float(input("Whats your average income? For example, type '12345'. ")),
        #"unemployment_rate": float(input("Whats the unemployment rate? 0-10. ")),
        #"school_quality_index": float(input("Whats the school quality index? 0-10. ")),
        #"population_density": float(input("Whats the population density? Per km^2. ")),
        #"crime_rate": float(input("Whats the crime rate? 0-10")),
        #"region": input("Enter the region. e.g. North, South, East, West, Central. ")
    #}])
    #X['region'] = X['region'].map(region_values)
    #answer=polypolyrf.predict(X)
    #print(f"The estimated price for your house would be {answer}. ")

    def read_fred_csv(filename, value_col_name):
        df = pd.read_csv(filename)
        df['DATE'] = pd.to_datetime(df['observation_date'])
        df.set_index('DATE', inplace=True)
        df.rename(columns={df.columns[0]: value_col_name}, inplace=True)
        return df


    def read_dupe_csv(filepath: str, value_name: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        df.set_index(df.columns[0], inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        df = df[numeric_cols]
        if len(df.columns) > 1:
            df = df[[df.columns[0]]]
        df.columns = [value_name]

        return df

    q_gdp_percapita = pd.read_csv('A939RC0Q052SBEA.csv')
    q_gdp_percapita.reset_index()
    q_gdp_percapita.set_index('observation_date', inplace=True)
    q_gdp_percapita.rename(columns={"A939RC0Q052SBEA":'GDP per Capita'}, inplace=True)
    print(q_gdp_percapita.head())
    q_gdp_percapita.index = pd.to_datetime(q_gdp_percapita.index)
    print(q_gdp_percapita.head())
    q_unrate = read_fred_csv('UNRATE.csv', 'Unemployment')
    q_invest = read_fred_csv('GPDI.csv', 'Investment')
    q_population = read_fred_csv('POP.csv', 'Population')
    q_poverty = read_dupe_csv('simulated_poverty.csv', 'Poverty')
    q_life = read_dupe_csv('simulated_life_expectancy.csv', 'Life Expectancy')
    q_gdp = read_dupe_csv('GDP.csv', 'GDP')
    q_gdp['GDP']=q_gdp['GDP'] * 1_000_000_000
    q_invest = q_invest['GPDI'] * 1_000_000_000
    print(q_gdp.head())
    print(q_gdp.columns)
    if isinstance(q_gdp.columns, pd.MultiIndex):
        q_gdp.columns = q_gdp.columns.get_level_values(-1)
    if 'GDP' in q_gdp.columns:
        q_gdp['GDP'] = pd.to_numeric(q_gdp['GDP'], errors='coerce')
    else:
        raise ValueError("Column 'GDP' not found in q_gdp")
    print(q_gdp.head())

    # Only keep numeric columns
    q_gdp = q_gdp.loc[:, q_gdp.dtypes.apply(pd.api.types.is_numeric_dtype)]
    print(q_gdp.head())

    # Now compute GDP Growth
    q_gdp['GDP Growth'] = q_gdp['GDP'].pct_change() * 100
    print(q_gdp.head())
    # Create a new DataFrame with only GDP Growth
    q_gdp_growth = q_gdp[['GDP Growth']]
    # CPI and Inflation calculation
    q_cpi = pd.read_csv('CPIAUCSL.csv', index_col='observation_date')
    q_cpi['CPIAUCSL'] = pd.to_numeric(q_cpi['CPIAUCSL'], errors='coerce')
    q_cpi.index = pd.to_datetime(q_cpi.index)
    q_cpi['Inflation_MoM'] = q_cpi['CPIAUCSL'].pct_change() * 100
    q_cpi.dropna(inplace=True)

    q_gdp_growth = q_gdp_growth.rename(columns={'GDP Growth': 'GDP Growth Rate'})

    q_gdp = q_gdp.resample('Q').ffill()
    q_population = q_population.resample('Q').ffill()
    q_poverty = q_poverty.resample('Q').ffill()
    q_life = q_life.resample('Q').ffill()
    q_invest = q_invest.resample('Q').ffill()
    q_gdp_percapita= q_gdp_percapita.resample('Q').ffill()
    print(q_gdp_percapita.head())

    for df in [q_gdp, q_gdp_growth, q_life, q_poverty, q_invest, q_population, q_unrate,q_gdp_percapita]:
        df.index = pd.to_datetime(df.index)
        df.index = df.index.to_period('Q').to_timestamp('Q')
    print(q_gdp_percapita.head())

    # Join into one economy DataFrame
    q_economy = q_gdp.join(
        [q_gdp_growth, q_invest, q_gdp_percapita],
        how='outer'
    )
    q_economy.drop(index=q_economy.index[0],axis=0,inplace=True)
    q_economy.drop(columns=('GDP Growth'),inplace=True)
    print(q_economy.shape)

    q_economy = pd.read_csv('made_up_weekly_economic_data.csv')
    q_economy.index = pd.to_datetime(q_economy.index)
    X_torch = q_economy.drop(columns=['GDP Growth Rate', 'Date'])
    Y_torch = q_economy['GDP Growth Rate']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_torch)
    Y_scaled = Y_torch.values.reshape(-1, 1)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=9, shuffle=True)

    deep_model = nn.Sequential(
        nn.Linear(9,32),
        nn.ReLU(),
        nn.Linear(32,64),
        nn.ReLU(),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32,16),
        nn.ReLU(),
        nn.Linear(16,1),
    )
    deep_model.load_state_dict(torch.load('gdp_model.pth'))
    deep_model.train()

    loss_l1 = nn.L1Loss()
    optimizer = optim.Adam(deep_model.parameters(), lr=0.001)

    num_epochs = 550
    for epoch in range(num_epochs):
        for batch_X, batch_Y in dataloader:
            prediction = deep_model(batch_X)
            loss = loss_l1(prediction,batch_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f"Epoch number {epoch+1}, loss = {loss:.2f}")
    with torch.no_grad():
        prediction = deep_model(X_tensor)
    prediction_np = prediction.numpy()
    y_np = Y_scaled

    prediction_unscaled = prediction_np * (Y_torch.max() - Y_torch.min()) + Y_torch.min()
    y_unscaled = y_np * (Y_torch.max() - Y_torch.min()) + Y_torch.min()

    for i in range(5):
        prediction_float = float(prediction_unscaled[i][0])
        y_float = float(y_unscaled[i][0])
        print(f"These are the first 5 predictions. Prediction: {prediction_float:.2f}%, Actual growth: {y_float:.2f}%.")
    #saving this thing
    torch.save(deep_model.state_dict(), 'gdp_model.pth')
    #loading this thing
