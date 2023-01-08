import base64

from dash import Dash, dash_table, dcc, html, ALL, MATCH, ALLSMALLER
from dash.dependencies import Input, Output, State
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import pandas as pd
import json
import os

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
# 'https://codepen.io/chriddyp/pen/bWLwgP.css',
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = DashProxy(transforms=[MultiplexerTransform()], suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.DARKLY, dbc_css])
# app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc_css])
# external_stylesheets = [ dbc.themes.DARKLY]

# app = Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

par_table = dash_table.DataTable(
    id='parameter-table',
    columns=[{"id": "Parameter Name", "name": "Parameter Name"},
             {"id": "Value", "name": "Value", "type": "numeric"},
             {"id": "Min", "name": "Min Value", "type": "numeric"},
             {"id": "Max", "name": "Max Value", "type": "numeric"}],
    style_cell={'textAlign': 'left'},
    data=[{"Parameter Name": "oxide SLD",
           "Value": "3.47",
           "Min": "3.45",
           "Max": "3.5"
           }],
    selected_rows=[],
    editable=True,
    row_deletable=True,
    row_selectable='multi',
    page_size=10,
    # style_as_list_view=True,
    style_data={
        # 'color': 'black',
        # 'backgroundColor': 'white'
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(20, 20, 20)',
        }
    ],
    style_header={
        'backgroundColor': 'rgb(210, 210, 210)',
        'color': 'black',
        'fontWeight': 'bold'
    }
)

layer_table = dash_table.DataTable(
    id='layer-table',
    columns=[{"id": "Layer Name", "name": "Layer Name"},
             {"id": "Thickness", "name": "Thickness", "presentation": "dropdown"},
             {"id": "SLD", "name": "SLD", "presentation": "dropdown"},
             {"id": "Roughness", "name": "Roughness", "presentation": "dropdown"}, ],
    style_cell={'textAlign': 'left'},
    data=[{"Layer Name": "oxide",
           "Thickness": "3.47",
           "SLD": "3.45",
           "Roughness": "3.5"
           }],

    editable=True,
    row_deletable=True,
    dropdown={'Thickness': {'options': [{'label': 'Default Value', 'value': 'Default Value'}]},
              'SLD': {'options': [{'label': 'Default Value', 'value': 'Default Value'}]},
              'Roughness': {'options': [{'label': 'Default Value', 'value': 'Default Value'}]}},

    style_data={
        # 'color': 'black',
        # 'backgroundColor': 'white'
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(20, 20, 20)',
        }
    ],
    style_header={
        'backgroundColor': 'rgb(210, 210, 210)',
        'color': 'black',
        'fontWeight': 'bold'
    }
)


@app.callback(
    Output("accordion-item", "children"),
    [Input("btn-add-contrast", "n_clicks")],
    [State("accordion-item", "children")],
)
def add_contrast(n, div_children):
    acc_item = dbc.AccordionItem(
        [dbc.FormFloating(
            [
                dbc.Input(type="input", placeholder=""),
                dbc.Label("File name"),
            ]
        ),
            dbc.Button('Add Row', n_clicks=0,
                       id={"type": "dynamic-button", "index": n}, ),

            dash_table.DataTable(
                id={"type": "contrast-table", "index": n},
                columns=[{"id": "Layer Name", "name": "Layer Name"},
                         {"id": "Layer", "name": "Layer", "presentation": "dropdown"}, ],
                style_cell={'textAlign': 'left'},
                data=[{"Layer Name": "Layer",
                       "Layer": ""
                       }],

                editable=True,
                row_deletable=True,
                dropdown={'Layer': {'options': [{'label': 'Default Value', 'value': 'Default Value'}]}},
                style_data={
                    # 'color': 'black',
                    # 'backgroundColor': 'white'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(20, 20, 20)',
                    }
                ],
                style_header={
                    'backgroundColor': 'rgb(210, 210, 210)',
                    'color': 'black',
                    'fontWeight': 'bold'
                }
            )

            # html.Div(children=[contrast_table], id="contrast-container"),
        ],
        title="Contrast " + str(n + 1),
        id={"type": "dynamic-acc-item", "index": n},
    )
    div_children.append(acc_item)
    return div_children


col1 = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Model File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    dbc.Accordion(
        id='accordion-item',
        children=[
            dbc.AccordionItem(
                [dbc.Button('Add Row', id='editing-par-rows-button', n_clicks=0),
                 par_table],
                title="Model Parameters",
            ),
            dbc.AccordionItem(
                [dbc.Button('Add Row', id='editing-layer-rows-button', n_clicks=0),
                 layer_table],
                title="Layers",
            ),
            # html.Div(id='accordion-item', children=[]),
        ],
        always_open=True,
    ),

    # html.Div(id='some-output'),

    dbc.Button("Add contrast", id="btn-add-contrast", n_clicks=0),

    dbc.Button("Save Model...", id="btn-download-model", style={"margin-left": "15px"}, n_clicks=0),
    dcc.Download(id="download-model")

    # dcc.Graph(id='adding-rows-graph')

], style={'width': '95%', 'display': 'inline-block', 'vertical-align': 'middle'}, className="dbc")

app.layout = html.Div(
    [
        dbc.Row([dbc.Col(col1),
                 dbc.Col(html.Div(html.Div(id='some-output'),
                                  style={'width': '95%', 'display': 'inline-block', 'vertical-align': 'middle'},
                                  className="dbc"), )]),
    ]
)


def generate_code(par_data, selected_rows, layer_data, contrast_data):
    par_data = par_data.reset_index()  # make sure indexes pair with number of rows
    layer_data = layer_data.reset_index()  # make sure indexes pair with number of rows

    outstring = '''    '''

    # for index, row in par_data.iterrows():
    #     outstring += '''{} = Parameter({}, "{}", bounds=({}, {}), vary={})
    #     '''.format(row['Parameter Name'].replace(' ', '_'),
    #                row['Value'], row['Parameter Name'].replace(' ', '_'),
    #                row['Min'], row['Max'],
    #                index in selected_rows)
    #
    # outstring += '''
    #     '''
    #
    # for index, row in layer_data.iterrows():
    #     outstring += '''{}_l = Slab({}, {}, {}, name='{}', vfsolv=0, interface=None))
    #     '''.format(row['Layer Name'].replace(' ', '_'),
    #                row['Thickness'].replace(' ', '_'), row['SLD'].replace(' ', '_'), row['Roughness'].replace(' ', '_'),
    #                row['Layer Name'].replace(' ', '_'))
    #
    # outstring += '''
    # '''
    #
    # outstring += '''    s_contrast = '''
    #
    # for index, row in contrast_data.iterrows():
    #     outstring += '''{} | '''.format(row['Layer'].replace(' ', '_'))
    # # s_d2o_sub = si_sld | oxide_l | d2o(0, solv_roughness)

    code = dcc.Markdown('''
    ```python
    {}
    ```'''.format(outstring))

    return code


@app.callback(Output('parameter-table', 'data'),
              Output('layer-table', 'data'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def load_model(model_dict, filename, list_of_dates):
    if model_dict is not None:
        content_type, content_string = model_dict.split(",")
        decoded = base64.b64decode(content_string)

        content_dict = json.loads(decoded)
        params = content_dict['pars']
        layers = content_dict['layers']

        return params, layers
    else:
        default_pars = [{"Parameter Name": "oxide SLD",
                         "Value": "3.47",
                         "Min": "3.45",
                         "Max": "3.5"
                         }]
        default_layers = [{"Layer Name": "oxide",
                           "Thickness": "3.47",
                           "SLD": "3.45",
                           "Roughness": "3.5"
                           }]
        return default_pars, default_layers


@app.callback(  # Output('some-output', 'children'),
    Output({"type": "contrast-table", "index": MATCH}, 'dropdown'),
    Input('layer-table', 'data'),
    State({"type": "contrast-table", "index": MATCH}, 'data')
)
def on_contrast_table_change(data, contrast_data):
    df = pd.DataFrame(data)

    opts = {'Layer': {'options': [{'label': v, 'value': v} for v in df.loc[:, 'Layer Name']]}}
    return opts


@app.callback(  # Output('some-output', 'children'),
    Output('layer-table', 'dropdown'),
    Input('parameter-table', 'selected_rows'),
    Input('parameter-table', 'data'),
    State('layer-table', 'data'),
)
def on_table_change(selected_rows, par_data, layer_data):
    p_data = pd.DataFrame(par_data)
    l_data = pd.DataFrame(layer_data)
    # c_data = pd.DataFrame(contrast_data)

    # code = generate_code(p_data, selected_rows, l_data, c_data)

    opts = {'Thickness': {'options': [{'label': v, 'value': v} for v in p_data.loc[:, 'Parameter Name']]},
            'SLD': {'options': [{'label': v, 'value': v} for v in p_data.loc[:, 'Parameter Name']]},
            'Roughness': {'options': [{'label': v, 'value': v} for v in p_data.loc[:, 'Parameter Name']]}}
    return opts


@app.callback(
    Output({"type": "contrast-table", "index": MATCH}, 'data'),
    Input({"type": "dynamic-button", "index": MATCH}, 'n_clicks'),
    State({"type": "contrast-table", "index": MATCH}, 'data'),
    State({"type": "contrast-table", "index": MATCH}, 'columns')
)
def add_contrast_table_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    return rows


@app.callback(
    Output({"type": "dynamic-button", "index": MATCH}, 'n_clicks'),
    Input({"type": "dynamic-button", "index": MATCH}, 'n_clicks'),
)
def upon_click(n_clicks):
    if n_clicks == 1:
        raise PreventUpdate
    print(n_clicks)
    return 1


@app.callback(
    Output('parameter-table', 'data'),
    Input('editing-par-rows-button', 'n_clicks'),
    State('parameter-table', 'data'),
    State('parameter-table', 'columns'))
def add_par_table_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    return rows


@app.callback(
    Output('editing-par-rows-button', 'n_clicks'),
    Input('editing-par-rows-button', 'n_clicks')
)
def upon_click(n_clicks):
    if n_clicks == 1:
        raise PreventUpdate
    print(n_clicks)
    return 1


@app.callback(
    Output('layer-table', 'data'),
    Input('editing-layer-rows-button', 'n_clicks'),
    State('layer-table', 'data'),
    State('layer-table', 'columns'))
def add_layer_table_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    return rows


@app.callback(
    Output('editing-layer-rows-button', 'n_clicks'),
    Input('editing-layer-rows-button', 'n_clicks')
)
def upon_click(n_clicks):
    if n_clicks == 1:
        raise PreventUpdate
    print(n_clicks)
    return 1


@app.callback(Output("download-model", "data"),
              State("parameter-table", "data"),
              State("layer-table", "data"),
              Input("btn-download-model", "n_clicks"),
              prevent_initial_call=True, )
def save_model(par_data, layer_data, n_clicks):
    model_dict = {'pars': par_data, 'layers': layer_data}
    return dict(content=json.dumps(model_dict, indent=4), filename="hello.txt")


if __name__ == '__main__':
    app.run_server(debug=True)
