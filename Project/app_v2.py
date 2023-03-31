import os
import dash
import numpy as np
import pandas as pd
from threading import Timer
import webbrowser
import yfinance as yf
import opstrat as op
from datetime import datetime
from scipy.stats import norm
import dash_table as dt
from dash import Dash, html, dcc, dash_table
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

class ComputeOptionPrice():

    def __init__(self, ticker: str):
        self.ticker = ticker
        stock = yf.Ticker(ticker)
        self.data_3m = stock.history(period = '3mo')

    def current_price(self) -> float:
        """ Pull the current or latest stock price from Yahoo Finance

        Returns:
            float: Current stock price
        """        
        return self.data_3m['Close'][-1]
    
    def historical_vol(self) -> float:
        """ Compute the 3-month historical volatility

        Returns:
            float: Implied volatility
        """
        self.data_3m['Log Returns'] = np.log(self.data_3m['Close']/self.data_3m['Close'].shift())
        return self.data_3m['Log Returns'].std()*(3*252/12)**.5
    
    def strike_price(self, moneyness: float) -> float:
        """ Calculate the strike price

        Args:
            moneyness (float): Indicates the option's moneyness (S/K)

        Returns:
            float: Strike price
        """        
        self.moneyness = moneyness
        return 100.0/moneyness*self.current_price()
    
    def option_price_greeks(self, moneyness: float) -> tuple:
        """ Calculate the 3-month x% moneyness call option price and black scholes greeks

        Args:
            moneyness (float): Indicates the option's moneyness (S/K)

        Returns:
            tuple: (Option price, B-S greeks)
        """        
        bsm = op.black_scholes(K = self.strike_price(moneyness), St = self.current_price(),
                               r = 0, t = 90, v = self.historical_vol()*100, type = 'c')
        return bsm['value']['option value'], bsm['greeks']
    
    def greeks_compare(self, moneyness: float) -> pd.DataFrame:
        """ Create a dataframe consisting of both black scholes and dollar greeks

        Args:
            moneyness (float): Indicates the option's moneyness (S/K)

        Returns:
            pd.DataFrame: Table containing both B-S and dollar greeks
        """        
        dollargreeks = (self.option_price_greeks(moneyness)[1]).copy()
        dollargreeks.update({'delta':dollargreeks['delta']*self.current_price(),
                             'gamma':dollargreeks['gamma']*self.current_price()**2/100})
        greeks = pd.DataFrame.from_dict([self.option_price_greeks(moneyness)[1], dollargreeks]).T
        greeks = greeks.reset_index()
        greeks.columns = ['Greeks', 'B-S Greeks', 'Dollar Greeks']
        return greeks

class DeltaHedging(ComputeOptionPrice):
    
    def __init__(self, ticker: str, start: str, end: str):
        self.ticker, self.start, self.end = ticker, start, end
        stock = yf.Ticker(ticker)
        self.data = stock.history(start = start, end = end)
        self.data = self.data['Close'].reset_index()
        self.data['Date'] = self.data['Date'].dt.strftime('%Y-%m-%d')
    
    def black_scholes(self, t: float, stock: float, strike: float, sigma: float) -> tuple:
        """ Compute the call option price and option delta using black scholes formula

        Args:
            t (float): Time to maturity
            stock (float): Current stock price
            strike (float): Strike price
            sigma (float): Implied volatility

        Returns:
            tuple: (Option price, delta)
        """              
        d1 = np.log(stock/strike)/sigma/np.sqrt(t) + 0.5*sigma*np.sqrt(t)
        d2 = d1 - sigma*np.sqrt(t)
        cp = (stock*norm.cdf(d1) - strike*norm.cdf(d2))
        delta = norm.cdf(d1)
        return cp, delta
    
    def hedging_algo(self, moneyness: float) -> pd.DataFrame:
        """ Perform dynamic delta hedging for the option and calculate the option, hedging and total PnLs

        Args:
            moneyness (float): Indicates the option's moneyness (S/K)

        Returns:
            pd.DataFrame: Table containing delta hedging data
        """        
        self.moneyness = moneyness
        strike = ComputeOptionPrice(self.ticker).strike_price(moneyness)
        sigma = ComputeOptionPrice(self.ticker).historical_vol()
        
        bsm = [self.black_scholes((datetime.strptime(self.end, '%Y-%m-%d') - 
                              datetime.strptime(self.data['Date'][i], '%Y-%m-%d')).days/365, 
                              self.data['Close'][i], strike, sigma) for i in range(len(self.data))]

        self.data['Option Price'], self.data['B-S Delta'] = [x[0] for x in bsm], [x[1] for x in bsm]

        self.data['Option Value'] = -100*self.data['Option Price']
        self.data['Hedge Value'] = 100*self.data['B-S Delta']*self.data['Close']

        self.data['Option PnL'] = self.data['Option Value'] - self.data['Option Value'].shift()
        self.data['Hedge PnL'] = self.data['Hedge Value'] - self.data['Hedge Value'].shift()
        self.data['Cashflow'] = -100*(self.data['B-S Delta'] - self.data['B-S Delta'].shift())*self.data['Close']
        self.data['Total PnL'] = self.data['Option PnL'] + self.data['Hedge PnL'] + self.data['Cashflow']

        self.data = self.data.drop(columns = ['Option Value']).rename(columns = {'Close': 'Stock Price'})
        return self.data

def blank_figure():
    """ Create a simple function to hide the initial default graph / figure grid on dash
    """
    fig = go.Figure(go.Scatter(x = [], y = []))
    fig.update_layout(template = None)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline = False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline = False)
    
    return fig


app = dash.Dash(__name__, suppress_callback_exceptions = True)

app.layout = html.Div([
    html.H3('Kirkoswald Asset Management LLC Interview Project'),
    html.I('Please enter a stock symbol and moneyness (%) and then hit submit!'),
    dcc.Input(id = 'ticker', value = 'SPY', type = 'text', style = {"margin-left": "15px"}),
    dcc.Input(id = 'moneyness', value = 100, type = 'number', style = {"margin-left": "15px"}),
    html.Button(id = 'submit-button', type = 'submit', children = 'Submit', style = {"margin-left": "15px"}),
    html.P(id = 'err', style = {'color':'red'}),
    html.P(id = 'option-price', style = {'whiteSpace': 'normal'}),
    html.P(id = 'greeks-title'),
    html.P(id = 'data-table', style = {"margin-left": "10px", "margin-right": "800px"}),
    html.P(id = 'theta-sigma-op'),
    dcc.Graph(id = 'graph', style = {'width': '70vw', 'height': '40vh'}, figure = blank_figure()),
    html.P(id = 'table-title'),
    html.P(id = 'table')])

        
@app.callback([Output('err', 'children'),
               Output('option-price', 'children'),
               Output('greeks-title', 'children'),
               Output('data-table', 'children'),
               Output('theta-sigma-op', 'children'),
               Output('graph', 'figure'),
               Output('table-title', 'children'),
               Output('table', 'children')],
              [Input('submit-button', 'n_clicks')],
              [State('ticker', 'value'), State('moneyness', 'value')], prevent_initial_call = True)


def final_output(clicks: int, ticker_value: str, moneyness_value: float) -> tuple:
    """ Create a dash page that takes stock ticker and moneyness as inputs and produces the following outputs

    Args:
        clicks (int): Number of clicks to the submit button
        ticker_value (str): Stock symbol / ticker
        moneyness_value (float): Indicates the option's moneyness (S/K)

    Raises:
        dash.exceptions.PreventUpdate: To block the previous / initial callback

    Returns:
        tuple: Dash output containing option prices, greeks, sigma, PnL plots and delta hedging data
    """       
    if clicks is None:
        raise dash.exceptions.PreventUpdate
        
    ticker_value = ticker_value.strip()

    try:
        assert not yf.Ticker(ticker_value).history(period = "3mo").empty
    except AssertionError:
        return ("Either the stock symbol you have entered is incorrect or it may have been delisted!", [], [], [], [], blank_figure(), [], [])
    else:
        try:
            assert float(moneyness_value) >= 75 and float(moneyness_value) <= 125
        except AssertionError:
            return ("Please enter a moneyness (%) value between 75 and 125!", [], [], [], [], blank_figure(), [], [])
        else:
            opprice = ComputeOptionPrice(ticker_value).option_price_greeks(moneyness_value)
            greeks_table = ComputeOptionPrice(ticker_value).greeks_compare(moneyness_value)

            start, end = '2023-01-02', '2023-04-05'  # Given dates

            t = (datetime.strptime(end, '%Y-%m-%d') - datetime.today()).days/365
            stock_p = ComputeOptionPrice(ticker_value).current_price()
            strike_p = ComputeOptionPrice(ticker_value).strike_price(moneyness_value)
            sigma = ComputeOptionPrice(ticker_value).historical_vol()

            del_hed = DeltaHedging(ticker_value, start, end).hedging_algo(moneyness_value)
            
            
            option_price = f'{"The 3-month "}{str(moneyness_value)}{"% moneyness call option price is: $"}{str(opprice[0])}'
            greeks_data = dt.DataTable(greeks_table.to_dict('records'), [{"name": i, "id": i} for i in greeks_table.columns])
            sig_theta_oprice = f'''{"The 1-day theta is: "}{str(opprice[1]['theta'])}{" , "}
                    {"Implied Volatility of the Option is: "}{'{:.6f}%'.format(100*ComputeOptionPrice(ticker_value).historical_vol())}
                    {" and the "}{"price of an "}{str(moneyness_value)}{"% money call option assuming t0 = "}{str(start)}
                    {" & T = "}{str(end)}{" is: $"}{str(DeltaHedging(ticker_value, start, end).black_scholes(t, stock_p, strike_p, sigma)[0])}'''
            pnl_plot = px.line(del_hed, x = "Date", y = ["Option PnL", "Hedge PnL", "Total PnL"], 
                               title = 'Delta Hedging - Profit & Loss', labels = dict(value = 'PnL', variable = 'Position'), 
                               template = "plotly_dark")
            delta_hedge_data = dt.DataTable(data = del_hed.to_dict('records'), 
                                            columns = [{"name": i, "id": i} for i in del_hed.columns], 
                                            style_table = {'overflowY': 'auto'})
            
            
            return ("",
                    option_price,  # 3-month x% moneyness call option price
                    "Greeks Comparison - B-S Greeks vs Dollar Greeks:",
                    greeks_data,  # Black-Scholes greeks vs $ greeks 
                    sig_theta_oprice,  # 1-day theta, implied volatility and the price of an x% money call option with given t0, T
                    pnl_plot,  # Option PnL, Hedging PnL and Total PnL in a single plot
                    "Results of Daily Delta Hedging on the Option:",
                    delta_hedge_data  # Results of daily delta hedging
            )
    

# Dash browser is set to open automatically but you can simply disable it by commenting out the code below

def open_browser():
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new('http://127.0.0.1:1227/')

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run_server(debug = True, port = 1227)
    
# Please uncomment the following code in case you do not want the dash browser to open automatically

#if __name__ == '__main__':
#    app.run_server(debug = True)