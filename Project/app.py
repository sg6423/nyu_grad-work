#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:08:12 2023

@author: harshag
"""

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


# Part - 1 of the project is to price an option and then compute greeks

class computeOptionPrice:

    def __init__(self, ticker):
        self.ticker = ticker
        stock = yf.Ticker(ticker)
        self.data_3m = stock.history(period = '3mo')

    def current_price(self):
        return self.data_3m['Close'][-1]
    
    def historical_vol(self):
        self.data_3m['Log Returns'] = np.log(self.data_3m['Close']/self.data_3m['Close'].shift())
        return self.data_3m['Log Returns'].std()*(3*252/12)**.5
    
    def strike_price(self, moneyness):
        self.moneyness = moneyness
        return 100.0/int(moneyness)*self.current_price()
    
    def option_price_greeks(self, moneyness):
        bsm = op.black_scholes(K = self.strike_price(moneyness), St = self.current_price(),
                               r = 0, t = 90, v = self.historical_vol()*100, type = 'c')
        return bsm['value']['option value'], bsm['greeks']
    
    def greeks_compare(self, moneyness):  # this function calculates dollar greeks (delta & gamma)
        dollargreeks = (self.option_price_greeks(moneyness)[1]).copy()
        dollargreeks.update({'delta':dollargreeks['delta']*self.current_price(),
                             'gamma':dollargreeks['gamma']*self.current_price()**2/100})
        greeks = pd.DataFrame.from_dict([self.option_price_greeks(moneyness)[1], dollargreeks]).T
        greeks = greeks.reset_index()
        greeks.columns = ['Greeks', 'B-S Greeks', 'Dollar Greeks']
        return greeks
    

# Part - 2 of the project calculates the option price again (using the mathematical formulas instead) and then
# performs daily delta hedging

class deltaHedging:
    
    def __init__(self, ticker, start, end):
        self.ticker, self.start, self.end = ticker, start, end
        stock = yf.Ticker(ticker)
        self.data = stock.history(start = start, end = end)
        self.data = self.data['Close'].reset_index()
        self.data['Date'] = self.data['Date'].dt.strftime('%Y-%m-%d')
    
    def black_scholes(self, t, S, K, sigma):
        d1 = np.log(S/K)/sigma/np.sqrt(t) + 0.5*sigma*np.sqrt(t)
        d2 = d1 - sigma*np.sqrt(t)
        cp = (S*norm.cdf(d1) - K*norm.cdf(d2))
        delta = norm.cdf(d1)
        return cp, delta
    
    def hedging_algo(self, moneyness):  # this function calculates the hedge value, cashflows, and the PnLs
        self.moneyness = moneyness
        K = computeOptionPrice(self.ticker).strike_price(moneyness)
        sigma = computeOptionPrice(self.ticker).historical_vol()
        
        bsm = [self.black_scholes((datetime.strptime(self.end, '%Y-%m-%d') - 
                              datetime.strptime(self.data['Date'][i], '%Y-%m-%d')).days/365, 
                              self.data['Close'][i], K, sigma) for i in range(len(self.data))]

        self.data['Option Price'], self.data['B-S Delta'] = [x[0] for x in bsm], [x[1] for x in bsm]

        self.data['Option Value'] = -100*self.data['Option Price']
        self.data['Hedge Value'] = 100*self.data['B-S Delta']*self.data['Close']

        self.data['Option PnL'] = self.data['Option Value'] - self.data['Option Value'].shift()
        self.data['Hedge PnL'] = self.data['Hedge Value'] - self.data['Hedge Value'].shift()
        self.data['Cashflow'] = -100*(self.data['B-S Delta'] - self.data['B-S Delta'].shift())*self.data['Close']
        self.data['Total PnL'] = self.data['Option PnL'] + self.data['Hedge PnL'] + self.data['Cashflow']

        self.data = self.data.drop(columns = ['Option Value']).rename(columns = {'Close': 'Stock Price'})
        return self.data
    
    
def blank_figure():  # this function hides the default graph grid
    fig = go.Figure(go.Scatter(x = [], y = []))
    fig.update_layout(template = None)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline = False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline = False)
    
    return fig

# The final part is to put all the parts together on a webpage using dash & plotly

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3('Kirkoswald Asset Management LLC Interview Project'),
    html.I('Please enter a stock symbol and moneyness(%) and then hit submit!'),
    dcc.Input(id = 'ticker', value = 'SPY', type = 'text', style = {"margin-left": "15px"}),
    dcc.Input(id = 'moneyness', value = 100, type = 'number', style = {"margin-left": "15px"}),
    html.Button(id = 'submit-button', type = 'submit', children = 'Submit', style = {"margin-left": "15px"}),
    html.Div(html.P([])), html.Div(id = 'option-price', style = {'whiteSpace': 'normal'}),
    html.Div(html.P([])), html.Div(id = 'greeks-title'),
    html.Div(html.P([])), html.Div(id = 'data-table', style = {"margin-left": "10px", "margin-right": "800px"}),
    html.Div(html.P([])), html.Div(id = 'thetasigmaop'),
    html.Div(html.P([])), dcc.Graph(id = 'graph', 
                                    style = {'width': '70vw', 'height': '40vh'}, figure = blank_figure()),
    html.Div(html.P([])), html.Div(id = 'table-title'),
    html.Div(html.P([])), html.Div(id = 'table')])


@app.callback([Output('option-price', 'children'),
               Output('greeks-title', 'children'),
               Output('data-table', 'children'),
               Output('thetasigmaop', 'children'),
               Output('graph', 'figure'),
               Output('table-title', 'children'),
               Output('table', 'children')],
              [Input('submit-button', 'n_clicks')],
              [State('ticker', 'value'), State('moneyness', 'value')])


def final_output(clicks, ticker_value, moneyness_value):
    if clicks is None:
        raise dash.exceptions.PreventUpdate
    
    opprice = computeOptionPrice(ticker_value).option_price_greeks(moneyness_value)
    greeks_table = computeOptionPrice(ticker_value).greeks_compare(moneyness_value)
    
    t0, T = '2023-01-02', '2023-03-31'  # given dates

    t = (datetime.strptime(T, '%Y-%m-%d') - datetime.today()).days/365
    S = computeOptionPrice(ticker_value).current_price()
    K = computeOptionPrice(ticker_value).strike_price(moneyness_value)
    Sigma = computeOptionPrice(ticker_value).historical_vol()
    
    del_hed = deltaHedging(ticker_value, t0, T).hedging_algo(moneyness_value)
    
    return (# 3-month x% moneyness call option price
            "The 3-month " + str(moneyness_value) + "% moneyness call option price is: $" + str(opprice[0]),
            
            "Greeks Comparison - B-S Greeks vs Dollar Greeks:",
            # Black-Scholes greeks vs $ greeks
            dt.DataTable(greeks_table.to_dict('records'), [{"name": i, "id": i} for i in greeks_table.columns]),
            # 1-day theta, implied volatility and the price of an x% money call option with given t0, T
            "The 1-day theta is: " + str(opprice[1]['theta']) + " , " + "Implied Volatility of the Option is: " + 
            '{:.6f}%'.format(100*computeOptionPrice(ticker_value).historical_vol()) + " and the " + "price of an " + 
            str(moneyness_value) + "% money call option assuming t0 = " + str(t0) + " & T = " + 
            str(T) + " is: $" + str(deltaHedging(ticker_value, t0, T).black_scholes(t, S, K, Sigma)[0]),
            # Option PnL, Hedging PnL and Total PnL in a single plot
            px.line(del_hed, x = "Date", y = ["Option PnL", "Hedge PnL", "Total PnL"], 
                    title = 'Delta Hedging - Profit & Loss', labels = dict(value = 'PnL', variable = 'Position'), 
                    template = "plotly_dark"),
            
            "Results of Daily Delta Hedging on the Option:",
            # Results of daily delta hedging
            dt.DataTable(data = del_hed.to_dict('records'), 
                         columns = [{"name": i, "id": i} for i in del_hed.columns], 
                         style_table = {'overflowY': 'auto'}))


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