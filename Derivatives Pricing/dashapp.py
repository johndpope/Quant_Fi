# -*- coding: utf-8 -*-
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from option_pricing import *
import inspect

app = dash.Dash(static_folder='static')
app.title='Option Pricer - Nicolas B'
app.layout =html.Div([ 

    html.Div([html.H3("Monte-Carlo Option Pricer")], style= {'marginLeft': '20px', 'marginBottom': '-10px'}),


    html.Div([

        html.Div([

            html.Table(
            [
                html.Tr( [html.Th("Inputs"), html.Th("Values")] )
            ] +
            [
                html.Tr( [html.Td("Option Type"), html.Td(dcc.Dropdown(
                id='my-dropdown',
                options=[
                    {'label': 'Vanilla', 'value': 'Vanilla'},
                    {'label': 'Digital', 'value': 'Digital'},
                    {'label': 'Asian', 'value': 'Asian'},
                    {'label': 'Barrier', 'value': 'Barrier'}
                ],
                clearable=False,
                placeholder="Select an Option type"
                ))]),

                html.Tr([html.Td(""), html.Td(dcc.RadioItems(
                id='is_call',
                options=[
                    {'label': 'Call', 'value': '1'},
                    {'label': 'Put', 'value': '-1'}
                ],
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
                ))]),

                html.Tr( [html.Td("Underlying"), html.Td(dcc.Input(
                id='S0',
                placeholder='Underlying Price',
                value='',
                type='text'
                        ))] ),

                html.Tr( [html.Td("Strike"), html.Td(dcc.Input(
                id='K',
                placeholder='Strike Price',
                value='',
                type='text'
                        ))] ),

                html.Tr( [html.Td("Maturity (Y)"), html.Td(dcc.Input(
                id='T',
                placeholder='Maturity',
                value='',
                type='text'
                        ))] ),

                html.Tr( [html.Td("Interest rate %"), html.Td(dcc.Input(
                id='rate',
                placeholder='Risk-free rate',
                value='',
                type='text'
                        ))] ),

                html.Tr( [html.Td("Volatility %"), html.Td(dcc.Input(
                id='sigma',
                placeholder='Volatility',
                value='',
                type='text'
                        ))] ),

                html.Tr( [html.Td("Dividend Yield %"), html.Td(dcc.Input(
                id='div',
                placeholder='Dividend',
                value='',
                type='text'
                        ))] ),

                html.Tr( [html.Td("Paths Number"), html.Td(dcc.Input(
                id='paths',
                placeholder='Paths Number',
                value='',
                type='text'
                        ))] )
            ]
            ),
        ], style= {'text-align': 'right'}),

        html.Div([    
            html.Button('Price', id='button-2'),
            html.Div(id='output')
        ], style = {'marginTop': '10px'})


    ], style = {'display': 'block', 'width': '25%', 'marginLeft': '20px', 'marginRight': '15px', 'float' :'left'}),

    html.Div([



            html.Table(
            [
                html.Tr( [html.Th("Outputs"), html.Th("Values")] )
            ] +
            [
                html.Tr( [html.Td("Price"), html.Td(id='output')] ),

                html.Tr( [html.Td("Delta"), html.Td(id='output1')] ),

                html.Tr( [html.Td("Gamma"), html.Td(id='output2')] ),

                html.Tr( [html.Td("Theta"), html.Td(id='output3')] ),

                html.Tr( [html.Td("Vega"), html.Td(id='output4')] ),

                html.Tr( [html.Td("Rho"), html.Td(id='output4')] )
            ]),

    ], style = {'display': 'block', 'width': '25%', 'marginLeft': '20px', 'float' :'left'})

])

@app.callback(
    Output('output', 'children'),
    [Input('button-2', 'n_clicks')],
    state=[State('my-dropdown', 'value'),
     State('is_call', 'value'),
     State('S0', 'value'),
     State('K', 'value'),
     State('rate', 'value'),
     State('T', 'value'),
     State('sigma', 'value'),
     State('div', 'value'),
     State('paths', 'value')])

def compute(n_clicks, type, iscall, S0, K, rate, T, sigma, div, paths):
    param = locals()
    for keys in param:
        try:
            param[keys] = float(param[keys])
        except:
            param[keys]

    if param['type'] == 'Vanilla':
        Option = MonteCarloVanilla(param['S0'], param['K'], param['rate'], param['T'], param['sigma'], 0, param['iscall'], int(param['paths']))
        return '{:.5f}'.format(Option.value())
    else: 
        return ''

app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
app.css.append_css({'external_url': 'https://rawgit.com/lwileczek/Dash/master/undo_redo5.css'})

if __name__ == '__main__':
    app.run_server(debug=True)