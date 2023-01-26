import base64
import pprint

from dataclasses import dataclass, asdict
from enum import Enum, auto

from dash import Dash, dash_table, dcc, html, ctx, no_update, ALL, MATCH, ALLSMALLER
from dash.dependencies import Input, Output, State
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import pandas as pd
import json
import os

## General interactions:
# Load model

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
# dbc_css = 'https://codepen.io/chriddyp/pen/bWLwgP.css',
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = DashProxy(transforms=[MultiplexerTransform()], suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.SLATE, dbc_css])
# app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc_css])
# external_stylesheets = [ dbc.themes.DARKLY]

# app = Dash(__name__, external_stylesheets=external_stylesheets)

par_table = dash_table.DataTable(
    id='parameter-table',
    # id={"type": 'parameter-table', "index": 0},
    columns=[{"id": "Parameter Name", "name": "Parameter Name"},
             {"id": "Value", "name": "Value", "type": "numeric"},
             {"id": "Min", "name": "Min Value", "type": "numeric"},
             {"id": "Max", "name": "Max Value", "type": "numeric"}],
    style_cell={'textAlign': 'left', 'font-size': 14, 'padding': '0px'},
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
             {"id": "Roughness", "name": "Roughness", "presentation": "dropdown"},
             {"id": "Hydration", "name": "Hydration", "presentation": "dropdown"},
             ],
    style_cell={'textAlign': 'left', 'font-size': 14, 'padding': '0px'},
    data=[{"Layer Name": "oxide",
           "Thickness": "3.47",
           "SLD": "3.45",
           "Roughness": "3.5",
           "Hydration": "0.0"
           }],

    editable=True,
    row_deletable=True,
    dropdown={'Thickness': {'options': [{'label': 'Default Value', 'value': 'Default Value'}]},
              'SLD': {'options': [{'label': 'Default Value', 'value': 'Default Value'}]},
              'Roughness': {'options': [{'label': 'Default Value', 'value': 'Default Value'}]},
              'Hydration': {'options': [{'label': 'Default Value', 'value': 'Default Value'}]}
              },

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


def add_contrast(n, div_children):
    acc_item = dbc.AccordionItem(
        [dbc.FormFloating(
            [
                dbc.Input(type="input", placeholder=""),
                dbc.Label("File name"),
            ]
        ),
            dbc.Button('Add Row', n_clicks=0,
                       id={"type": "dynamic-button", "index": n},
                       style={"padding": '5px'}, ),

            dash_table.DataTable(
                id={"type": "contrast-table", "index": n},
                columns=[{"id": "Layer Name", "name": "Layer Name"},
                         {"id": "Layer", "name": "Layer", "presentation": "dropdown"}, ],
                style_cell={'textAlign': 'left', 'font-size': 14, 'padding': '0px'},
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
    return div_children, n


def delete_contrast(n, div_children, ncontrast, which):
    if n > 0:
        div_children.pop(which + 1)
        return div_children, ncontrast - 1
    else:
        return div_children, ncontrast


@app.callback(
    Output({"type": "dynamic-acc-item", "index": ALL}, "title"),
    Input("input_contrast_range", "max"),
    State("accordion-item", "children"),
    State({"type": "dynamic-acc-item", "index": ALL}, "title")
)
def update_contrast_titles(ncontrast, div_children, titles):
    for i, title in enumerate(titles):
        titles[i] = "Contrast " + str(i + 1)

    return titles


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
                [dbc.Button('Add Row', id='editing-par-rows-button', n_clicks=0,
                            style={"padding": '5px'}, ),
                 par_table],
                title="Model Parameters",
            ),
            dbc.AccordionItem(
                [dbc.Button('Add Row', id='editing-layer-rows-button', n_clicks=0,
                            style={"padding": '5px'}, ),
                 layer_table],
                title="Layers",
            ),
        ],
        always_open=True,
    ),

    # html.Div(id='some-output'),

    dbc.Button("Add contrast", id="btn-add-contrast", n_clicks=0),
    dbc.Button("Delete contrast", id="btn-delete-contrast", n_clicks=0),
    dcc.Input(id="input_contrast_range", type="number", placeholder="input with range",
              value=1, min=1, max=1, step=1, ),

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

    for index, row in par_data.iterrows():
        outstring += '''{} = Parameter({}, "{}", bounds=({}, {}), vary={})
        '''.format(row['Parameter Name'].replace(' ', '_'),
                   row['Value'], row['Parameter Name'].replace(' ', '_'),
                   row['Min'], row['Max'],
                   index in selected_rows)

    outstring += '''
        '''

    for index, row in layer_data.iterrows():
        outstring += '''{}_l = Slab({}, {}, {}, name='{}', vfsolv=0, interface=None))
        '''.format(row['Layer Name'].replace(' ', '_'),
                   row['Thickness'].replace(' ', '_'), row['SLD'].replace(' ', '_'), row['Roughness'].replace(' ', '_'),
                   row['Layer Name'].replace(' ', '_'))

    outstring += '''
    '''

    # outstring += '''    s_contrast = '''
    #
    for index, row in contrast_data.iterrows():
        outstring += '''    
        s_contrast_{} ='''.format(index + 1)
        for lay in row:
            if lay is not None:
                outstring += ''' {} |'''.format(lay['Layer'].replace(' ', '_'))

    # # s_d2o_sub = si_sld | oxide_l | d2o(0, solv_roughness)

    code = dcc.Markdown('''
    ```python
    {}
    ```'''.format(outstring))

    return code


def load_model(model_dict, filename, list_of_dates, acc_item):
    if model_dict is not None:
        content_type, content_string = model_dict.split(",")
        decoded = base64.b64decode(content_string)

        content_dict = json.loads(decoded)
        params = content_dict['Pars']
        layers = content_dict['Layers']
        contrasts = content_dict['Contrasts']
        print(contrasts)
        # for i in range(2, len(acc_item)):
        #     print(i)
        # acc_item.pop(-1)

        return params, layers, acc_item  # , contrasts
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
        return default_pars, default_layers, acc_item


@app.callback(  # Output('some-output', 'children'),
    Output({"type": "contrast-table", "index": MATCH}, 'dropdown'),
    Input('layer-table', 'data'),
    State({"type": "contrast-table", "index": MATCH}, 'data')
)
def on_contrast_table_change(data, contrast_data):
    df = pd.DataFrame(data)

    opts = {'Layer': {'options': [{'label': v, 'value': v} for v in df.loc[:, 'Layer Name']]}}
    return opts


@app.callback(Output('some-output', 'children'),
              Output('layer-table', 'dropdown'),
              Input('parameter-table', 'selected_rows'),
              Input('parameter-table', 'data'),
              State('layer-table', 'data'),
              # State({"type": "dynamic-acc-item", "index": ALL}, )
              State({"type": "contrast-table", "index": ALL}, 'data')
              )
def on_table_change(selected_rows, par_data, layer_data, contrast_data):
    p_data = pd.DataFrame(par_data)
    l_data = pd.DataFrame(layer_data)
    c_data = pd.DataFrame(contrast_data)

    code = generate_code(p_data, selected_rows, l_data, c_data)

    opts = {'Thickness': {'options': [{'label': v, 'value': v} for v in p_data.loc[:, 'Parameter Name']]},
            'SLD': {'options': [{'label': v, 'value': v} for v in p_data.loc[:, 'Parameter Name']]},
            'Roughness': {'options': [{'label': v, 'value': v} for v in p_data.loc[:, 'Parameter Name']]},
            'Hydration': {'options': [{'label': v, 'value': v} for v in p_data.loc[:, 'Parameter Name']]}
            }
    return code, opts


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

    return 1


@app.callback(Output("download-model", "data"),
              State("parameter-table", "data"),
              State("layer-table", "data"),
              State({"type": "contrast-table", "index": ALL}, "data"),
              State("btn-add-contrast", "n_clicks"),
              Input("btn-download-model", "n_clicks"),
              prevent_initial_call=True, )
def save_model(par_data, layer_data, contrast_data, ncontrasts, n_clicks):
    model_dict = {'Pars': par_data, 'Layers': layer_data}

    contrast_list = []
    for i, contrast in enumerate(contrast_data):
        contrast_list.append({'Contrast_' + str(i + 1): contrast})

    model_dict['Contrasts'] = contrast_list
    return dict(content=json.dumps(model_dict, indent=4), filename="model.json")


@app.callback(
    output=dict(
        par_table=Output('parameter-table', 'data'),
        layer_table=Output('layer-table', 'data'),
        acc_item_out=Output('accordion-item', 'children'),
        add_contrast_out=Output("btn-add-contrast", "n_clicks"),
        max_contrast_out=Output("input_contrast_range", "max"),
    ),
    inputs=dict(
        data_in=Input('upload-data', 'contents'),
        add_contrast=Input("btn-add-contrast", "n_clicks"),
        del_contrast=Input("btn-delete-contrast", "n_clicks"),
        fname=State('upload-data', 'filename'),
        last_mod=State('upload-data', 'last_modified'),
        acc_item_in=State('accordion-item', 'children'),
        max_contrast_in=State("input_contrast_range", "value"),
    ),
)
def contrast_handling(**kwargs):
    @dataclass
    class Update:
        par_table: ... = no_update
        layer_table: ... = no_update
        acc_item_out: ... = no_update
        add_contrast_out: ... = no_update
        max_contrast_out: ... = no_update

    if ctx.triggered_id == 'upload-data':
        params, layers, acc_item = load_model(kwargs["data_in"], kwargs["fname"], kwargs["last_mod"],
                                              kwargs["acc_item_in"])
        return asdict(Update(par_table=params, layer_table=layers, acc_item_out=acc_item))

    if ctx.triggered_id == 'btn-add-contrast':
        div_children, max_con = add_contrast(kwargs["add_contrast"], kwargs["acc_item_in"])
        return asdict(Update(acc_item_out=div_children, max_contrast_out=max_con, add_contrast_out=max_con))

    if ctx.triggered_id == 'btn-delete-contrast':
        div_children, max_con = delete_contrast(kwargs["del_contrast"], kwargs["acc_item_in"],
                                                  kwargs["add_contrast"], kwargs["max_contrast_in"])
        return asdict(Update(acc_item_out=div_children, max_contrast_out=max_con, add_contrast_out=max_con))

    return asdict(Update())


if __name__ == '__main__':
    app.run_server(debug=True)
