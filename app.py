import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
import psycopg2
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import itertools
from sqlalchemy import create_engine, text
import numpy as np 
import base64
import io
import seaborn as sns
import plotly.colors as colors
import plotly.express as px
from plotly.subplots import make_subplots
import postgres_upload
import webbrowser

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'] # just styling

app = dash.Dash(__name__, external_stylesheets=external_stylesheets) # app

app.title = "Abacus Visualizer"

chromosomes_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y']
chromosome_sizes = [249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983, 63025520, 48129895, 51304566, 155270560, 59373566]


# Calculate cumulative sizes
cumulative_sizes = [0] + list(itertools.accumulate(chromosome_sizes)) 




def get_red_palette(num_colours):
    '''Get the specified number of red colours in hex format.'''
    reds = []

    red_shades = sns.color_palette('Reds', num_colours)

    for red_shade in red_shades:
        rgb = tuple([int(x * 255) for x in red_shade])
        reds.append('#%02x%02x%02x' % rgb)

    return reds

def get_copy_number_palette(num_states):
    '''Get a copy number colour palette for the specified number of states.'''
    blues = ['#3498db', '#85c1e9']
    grey = ['#d3d3d3']
    purples = ['#780428', '#530031', '#40092e', '#2d112b']
    segment_colours = blues + grey

    if num_states <= 11:
        reds = get_red_palette(num_states - 3)
        segment_colours.extend(reds)
    else:
        reds = get_red_palette(num_states - 7)
        segment_colours.extend(reds)
        segment_colours.extend(purples)

    return segment_colours

def discrete_colorscale(bvals, colors):
    """
    bvals - list of values bounding intervals/ranges of interest
    colors - list of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],0<=k < len(bvals)-1
    returns the plotly  discrete colorscale
    """
    if len(bvals) != len(colors)+1:
        raise ValueError('len(boundary values) should be equal to  len(colors)+1')
    bvals = sorted(bvals)     
    nvals = [(v-bvals[0])/(bvals[-1]-bvals[0]) for v in bvals]  #normalized values
    
    dcolorscale = [] # discrete colorscale
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k+1], colors[k]]])
    return dcolorscale    



# def main():
engine = None
df = None
heatmap_df = None
x_mapping = None
heatmap_fig = go.Figure()

postgres_upload.run_upload()
# root.mainloop()  # This will start the Tkinter GUI main loop
# Run Dash app in the main thread
database, user, password, host, port = postgres_upload.get_connection_details()
print("PORT:", port)
print("user:", user)
print("password:", password)
print("host:", host)
print("database:", database)
engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')

# engine = create_engine('postgresql://postgres:password@gphost08:5432/postgres')

# Fetch data from the database
query = "SELECT id, cell_id, chrom, start, end_pos, two, total, copy_number, modal_corrected, assignment FROM details ORDER BY id"

df = pd.read_sql(query, engine)
heatmap_df = df

x_mapping = [start + cumulative_sizes[chromosomes_names.index(chrom)] for start, end_pos, chrom in zip(df['start'], df['end_pos'], df['chrom'])]

bvals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
color_schemes = ['#3498db', '#85c1e9', '#d3d3d3', '#fee2d5', '#fcc3ac', '#fc9f81', '#fb7c5c', '#f5543c', '#e32f27', '#c1151b', '#9d0d14', '#780428', '#530031', '#40092e', '#2d112b']

# colorscale = [[0, '#3498db'], [1, '#85c1e9'], [2, '#d3d3d3'], [3, '#fee2d5'], [4, '#fcc3ac'], [5, '#fc9f81'], [6, '#fb7c5c'], [7, '#f5543c'], [8, '#e32f27'], [9, '#c1151b'], [10, '#9d0d14'], [11, '#780428'], [12, '#530031'], [13, '#40092e'], [14, '#2d112b']]
# colorscale = colors.make_colorscale(['#3498db', '#85c1e9', '#d3d3d3', '#fee2d5', '#fcc3ac', '#fc9f81', '#fb7c5c', '#f5543c', '#e32f27', '#c1151b', '#9d0d14', '#780428', '#530031', '#40092e', '#2d112b'])
colorscale = discrete_colorscale(bvals, color_schemes)

bvals = np.array(bvals)
tickvals = [str(i) for i in range(15)] #position with respect to bvals where ticktext is displayed
ticktext = [f'<{bvals[1]}'] + [f'{bvals[k]}-{bvals[k+1]}' for k in range(1, len(bvals)-2)]+[f'>{bvals[-2]}']


# print(x_mapping)
heatmap_fig = go.Figure(data=go.Heatmap(
    z=df['copy_number'],
    x=x_mapping,
    y=df['cell_id'],
    colorscale=colorscale,
    zmin=0,  # Set the minimum value for the colorbar scale
    zmax=14,  # Set the maximum value for the colorbar scale
    colorbar=dict(title="Copy Number"),
))
heatmap_fig.update_yaxes(title_text="Cell ID")

heatmap_fig.update_layout(
    title='Heatmap',
    xaxis=dict(
        title='Chromosome',
        tickvals=cumulative_sizes,
        ticktext=chromosomes_names + [''],
        tickmode='array',
        ticklabelmode='period',
        ticklabelposition='outside',
        tickformat='d',
        range=[0, cumulative_sizes[-1] + chromosome_sizes[-1]], 
    )
)


x_options = df.columns
y_options = df.columns
z_options = df.columns
cell_id_options = df['cell_id'].unique().tolist()

# Create the Dash layout
app.layout = html.Div(
    [
    # dcc.Loading(
    # id="loading-component",
    # type="circle",  # You can also use "circle" or "dot"
    # children=[
    html.H1("Abacus Visualizer", style={"background-color": "#fb7c5c", "color": "white", "padding": "13px"}),
    
    dcc.Loading(
        id="loading-component",
        type="circle",  # You can also use "circle" or "dot"
        children=[dcc.Graph(
        id='heatmap',
        figure=heatmap_fig,
        style={'overflow': 'scroll', 'height': '900px'}
    )]),
    dcc.Store(id='heatmap-state'),

    html.Div(id='info-block', children=[], style={'width': '97%', 'text-align': 'right', 'font-size': '15px'}),
    html.Div([
        html.Div([
            html.Label(['Metadata:'], style={'font-weight': 'bold', "text-align": "left"}),
            dcc.Upload(
                id='upload-metadata',
                children=html.Button('Upload CSV File')
            ),
            html.Div(id='output-message'),
            html.Div(id='intermediate-data', style={'display': 'none'}),
        ], style={'width': '13%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.Label(['Parameter:'], style={'font-weight': 'bold', "text-align": "left"}),
            dcc.Dropdown(
                id='column-dropdown',
                options=[{'label': x, 'value': x} for x in x_options],
                value=None,
                placeholder='Select a value'
            ),
        ], style={'width': '13%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.Label(['filter by:'], style={'font-weight': 'bold', "text-align": "left"}),
            dcc.Dropdown(
                id='fil-cond-dropdown-metadata',
                options=[
                    {'label': '>', 'value': '>'},
                    {'label': '=', 'value': '='},
                    {'label': '<', 'value': '<'}
                ],
                style={'width': '100%'},
            ),
        ], style={'width': '13%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.Label(['filter by:'], style={'font-weight': 'bold', "text-align": "left"}),
            dcc.Dropdown(
                id='value-dropdown',
                options=[],
                value=None,
                placeholder='Select a value'
            ),
        ], style={'width': '13%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ]),

    html.Div([
        html.Button('reset', id='reset-button', n_clicks=0),
        html.Button('apply', id='apply-button', n_clicks=0),
    ], style={'display': 'inline-block', 'justify-content': 'center', 'margin-top': '10px'}),
    html.Div([
        dcc.Graph(id='scatter', style={'width': '60%'}),
        dcc.Graph(id='multiplicity', style={'width': '40%'})
    ], style={'display': 'flex', 'width': '100%'}),
    html.Div([
        html.Div([
            html.Label(['Cell ID:'], style={'font-weight': 'bold', "text-align": "left"}),
            dcc.Dropdown(
                id='cell-id-dropdown',
                options=[{'label': cell_id, 'value': cell_id} for cell_id in cell_id_options],
                style={'width': '100%'},
                value=df['cell_id'][0]
            ),
        ], style={'width': '13%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.Label(['Parameter:'], style={'font-weight': 'bold', "text-align": "left"}),
            dcc.Dropdown(
                id='x-axis-dropdown-scatter',
                options=[{'label': x, 'value': x} for x in x_options],
                style={'width': '100%'},
            ),
        ], style={'width': '13%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.Label(['filter by:'], style={'font-weight': 'bold', "text-align": "left"}),
            dcc.Dropdown(
                id='fil-cond-dropdown-scatter',
                options=[
                    {'label': '>', 'value': '>'},
                    {'label': '=', 'value': '='},
                    {'label': '<', 'value': '<'}
                ],
                style={'width': '100%'},
            ),
        ], style={'width': '3%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.Label(['value:'], style={'font-weight': 'bold', "text-align": "left"}),
            dcc.Input(id='input-val-scatter', type='number', min=0, value=0)
        ], style={'width': '10%', 'display': 'inline-block', 'vertical-align': 'top'}),  
    ]),
    html.Div([     
        # html.Label(['gpfit data:'], style={'font-weight': 'bold', "text-align": "left"}),      
        # dcc.Upload(
        #     id='upload-data',
        #     children=html.Button('Upload CSV File')
        # ),
        html.Div(id='output-message'),
        html.Button('apply', id='apply-button-scatter', n_clicks=0, style={'width': '105px'}),
    ], style={'margin-top': '10px', 'display': 'flex', 'flex-direction': 'row', 'align-items': 'center'})
    # ])
    ], style={'display': 'flex', 'flex-direction': 'column', 'width': '100%'})


    
print('done')


def get_coverage_order(df_group):
    '''Get the order of magnitude of the coverage depth for the panel.'''
    group_order = 1

    if len(df_group) > 0:
        if df_group['two'].max() > 0:
            group_order = int(np.log10(df_group['two'].max()))

    return group_order 

@app.callback(
    Output('intermediate-data', 'children'),
    Output('column-dropdown', 'options'),
    Input('upload-metadata', 'contents'),
    State('upload-metadata', 'filename'),
    prevent_initial_call=True
)
def update_dropdown(contents, filename):
    if contents is None:
        raise dash.exceptions.PreventUpdate

    # Read the uploaded file into a DataFrame
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df_meta = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    # Update options for column-dropdown based on DataFrame columns

    column_options = [{'label': col, 'value': col} for col in (df_meta.columns.union(x_options))]

    return df_meta.to_json(date_format='iso', orient='split'), column_options

# what is this function even doing ?? need to figure this out before moving forward with metadata filtering
@app.callback(
    Output('value-dropdown', 'options'),
    Input('column-dropdown', 'value'),
    State('intermediate-data', 'children'),
    prevent_initial_call=True
)
def update_value_dropdown(selected_column, jsonified_df):
    if selected_column is None:
        raise dash.exceptions.PreventUpdate
    
    if jsonified_df is None:
        return [{'label': str(val), 'value': val} for val in df[selected_column].unique()]

    # Read the DataFrame from the intermediate data
    df_meta = pd.read_json(jsonified_df, orient='split')

    if selected_column in df_meta.columns:
        # Get unique values for the selected column
        unique_values = df_meta[selected_column].unique() 
    else:
        unique_values = df[selected_column].unique()

    # Update options for value-dropdown based on unique values
    value_options = [{'label': str(val), 'value': val} for val in unique_values]

    return value_options


@app.callback(
    [Output('heatmap', 'figure'),
     Output('apply-button', 'n_clicks'),
     Output('reset-button', 'n_clicks'),
     Output('heatmap-state', 'data')],
    [Input('apply-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('upload-metadata', 'contents')],
    [State('column-dropdown', 'value'),
    State('fil-cond-dropdown-metadata', 'value'),
    State('value-dropdown', 'value'),
    State('intermediate-data', 'children')]
)
def update_heatmap(n_clicks_apply, n_clicks_reset, uploaded_metadata, x_axis_val, cond_type, input_val, intermediate_data):

    curr_heatmap_fig = go.Figure()
    heatmap_state = None

    if 'apply-button' == ctx.triggered_id:
        # # Map the dropdown value to the corresponding conditional operator
        # # Get the selected conditional operator from the dropdown value
        # conditional_operator = conditional_operators[cond_type]

        df = None
        # Construct the SQL statement dynamically
        if intermediate_data is None or x_axis_val in x_options:
            sql_query = f"SELECT id, cell_id, chrom, start, end_pos, copy_number, modal_corrected, assignment FROM details WHERE {x_axis_val} {cond_type} {input_val} ORDER BY id"
            df = pd.read_sql(sql_query, engine)
        else:
            content_type, content_string = uploaded_metadata.split(',')
            decoded = base64.b64decode(content_string)
            
            # Read the file using pandas
            metadata_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            # Apply the selected filter to the metadata dataset
            filtered_data = metadata_df
            condition = None

            if cond_type == '=':
                    cond_type = '=='

            if isinstance(input_val, (int, float)):
                condition = f"{x_axis_val} {cond_type} {input_val}"
            else:
                condition = f"{x_axis_val} {cond_type} '{input_val}'"
            print(condition)
            filtered_data = filtered_data.query(condition)

            metadata_df_sel_cell_ids = filtered_data['cell_id'].unique()
            metadata_df_sel_cell_ids = metadata_df_sel_cell_ids.tolist()

            # Define the SQL query with an "IN" clause and parameter placeholders
            sql_query = text("SELECT id, cell_id, chrom, start, end_pos, copy_number, modal_corrected, assignment FROM details WHERE cell_id IN :param1 ORDER BY id")

            # Bind the parameter using bindparams
            sql_query = sql_query.bindparams(param1=tuple(metadata_df_sel_cell_ids))

            df = pd.read_sql(sql_query, engine)            
        

        dd_heatmap_fig = go.Figure(data=go.Heatmap(
            z=df['copy_number'],
            x=x_mapping,
            y=df['cell_id'],
            colorscale=colorscale,
            zmin=0,  # Set the minimum value for the colorbar scale
            zmax=14,  # Set the maximum value for the colorbar scale
            colorbar=dict(title="Copy Number"),
        ))
        dd_heatmap_fig.update_yaxes(title_text="Cell ID")

        dd_heatmap_fig.update_layout(
            title='Heatmap',
            xaxis=dict(
                title='Chromosome',
                tickvals=cumulative_sizes,
                ticktext=chromosomes_names + [''],
                tickmode='array',
                ticklabelmode='period',
                ticklabelposition='outside',
                tickformat='d',
                range=[0, cumulative_sizes[-1] + chromosome_sizes[-1]], 
            )
        )

        curr_heatmap_fig = dd_heatmap_fig
        heatmap_state = dd_heatmap_fig
        n_clicks_apply = 0

    elif "reset-button" == ctx.triggered_id:
        curr_heatmap_fig = heatmap_fig
        n_clicks_reset = 0
    else:
        curr_heatmap_fig = heatmap_fig

    return curr_heatmap_fig, n_clicks_apply, n_clicks_reset, heatmap_state

@app.callback(
    Output('info-block', 'children'),
    [Input('heatmap', 'clickData'),
     Input('heatmap-state', 'data')]
)
def update_info_block(clickData, heatmap_state):
    if clickData is not None and heatmap_state is None:
        point_data = clickData['points'][0]
        chromosome = get_chrom_from_x_val(point_data['x'])
        cell_id = point_data['y']
        copy_number = point_data['z']

        # Construct the text content for the info block
        text_content = f'''
            **Chromosome:** {chromosome} \t
            **Cell ID:** {cell_id} \t
            **Copy Number:** {copy_number}
        '''
        return dcc.Markdown(text_content)
    else:
        text_content = f'''
            **Chromosome:**  \t
            **Cell ID:**  \t
            **Copy Number:** 
        '''
        return text_content

def get_chrom_from_x_val(val):
    # [0, 249250621, 492449994, 690472424, 881626700, 1062541960, 1233657027, 1392795690, 1539159712, 1680373143, 1815907890, 1950914406, 2084766301, 2199936179, 2307285719, 
    # 2409817111, 2500171864, 2581367074, 2659444322, 2718573305, 2781598825, 2829728720, 2881033286, 3036303846, 3095677412]
    if 0 <= val <= 249250621:
        return '1'
    elif 249250621 < val <= 492449994:
        return '2'
    elif 492449994 < val <= 690472424:
        return '3'
    elif 690472424 < val <= 881626700:
        return '4'
    elif 881626700 < val <= 1062541960:
        return '5'
    elif 1062541960 < val <= 1233657027:
        return '6'
    elif 1233657027 < val <= 1392795690:
        return '7'
    elif 1392795690 < val <= 1539159712:
        return '8'
    elif 1539159712 < val <= 1680373143:
        return '9'
    elif 1680373143 < val <= 1815907890:
        return '10'
    elif 1815907890 < val <= 1950914406:
        return '11'
    elif 1950914406 < val <= 2084766301:
        return '12'
    elif 2084766301 < val <= 2199936179:
        return '13'
    elif 2199936179 < val <= 2307285719:
        return '14'
    elif 2307285719 < val <= 2409817111:
        return '15'
    elif 2409817111 < val <= 2500171864:
        return '16'
    elif 2500171864 < val <= 2581367074:
        return '17'
    elif 2581367074 < val <= 2659444322:
        return '18'
    elif 2659444322 < val <= 2718573305:
        return '19'
    elif 2718573305 < val <= 2781598825:
        return '20'
    elif 2781598825 < val <= 2829728720:
        return '21'
    elif 2829728720 < val <= 2881033286:
        return '22'
    elif 2881033286 < val <= 3036303846:
        return 'X'
    elif 3036303846 < val <= 3095677412:
        return 'Y'

# Define the callback to show the scatter plot on hover
@app.callback(
    [Output('scatter', 'figure'),
    Output('apply-button-scatter', 'n_clicks')],
    [Input('heatmap', 'clickData'),
    Input('heatmap-state', 'data'),
    Input('apply-button-scatter', 'n_clicks')],
    [State('cell-id-dropdown', 'value'),
     State('x-axis-dropdown-scatter', 'value'),
     State('fil-cond-dropdown-scatter', 'value'),
     State('input-val-scatter', 'value')]
)
def show_scatter(clickData, heatmap_state, n_clicks_apply_scatter, cell_id, x_axis_val, cond_type, input_val):
    # print("n clicks scatter apply", n_clicks_apply_scatter)
    if clickData is not None and not n_clicks_apply_scatter:
        # Get the rows for the selected 'y_value'
        selected_cell_id = clickData['points'][0]['y']

        # i think this is the trouble code
        statement = text("SELECT cell_id, chrom, start, end_pos, copy_number, modal_corrected, assignment FROM details WHERE cell_id = :param1 ORDER BY cell_id DESC")
        statement = statement.bindparams(param1=selected_cell_id)

        scatter_df = pd.read_sql(statement, engine)

        scatter_fig = go.Figure(data=go.Scatter(
            x=[start + cumulative_sizes[chromosomes_names.index(chrom)] for start, chrom in zip(scatter_df['start'], scatter_df['chrom'])],
            y=scatter_df['modal_corrected']*scatter_df['assignment'],
            mode='markers',
            marker=dict(
                size=5,
                color=scatter_df['copy_number'],
                colorscale=colorscale,
                cmin=0,
                cmax=14,
                showscale=True,
                colorbar={"title": "Copy Number"}, 
            ),
            text=chromosomes_names,
            hovertemplate='Position: %{x} <br>Scaled Sequencing Coverage: %{y}',
            showlegend=False 
        ))

        scatter_fig.update_layout(
            title='Scatter Plot',
            xaxis=dict(
                title='Chromosome',
                tickvals=cumulative_sizes,
                ticktext=chromosomes_names + [''],
                tickmode='array',
                ticklabelmode='period',
                ticklabelposition='outside',
                tickformat='d',
                range=[0, cumulative_sizes[-1] + chromosome_sizes[-1]], 
            ),
        )
        scatter_fig.update_yaxes(title_text="Scaled sequencing coverage")

        return scatter_fig, n_clicks_apply_scatter
    
    if "apply-button-scatter" == ctx.triggered_id:
        triggered_component_id = ctx.triggered[0]['prop_id'].split('.')[0]
        sql_query = None
        if cell_id is not None:
            if x_axis_val is None:
                sql_query = f"SELECT cell_id, chrom, start, end_pos, copy_number, modal_corrected, assignment FROM details WHERE cell_id = '{cell_id}' ORDER BY copy_number DESC"
            else:
                sql_query = f"SELECT cell_id, chrom, start, end_pos, copy_number, modal_corrected, assignment FROM details WHERE {x_axis_val}  {cond_type} {input_val}  AND cell_id = '{cell_id}' ORDER BY copy_number DESC"

        scatter_df = pd.read_sql(sql_query, engine)
    

        # if x_val == None:
            
        scatter_fig = go.Figure(data=go.Scatter(
            x=[start + cumulative_sizes[chromosomes_names.index(chrom)] for start, chrom in zip(scatter_df['start'], scatter_df['chrom'])],
            y=scatter_df['modal_corrected']*scatter_df['assignment'],
            mode='markers',
            marker=dict(
                size=5,
                color=scatter_df['copy_number'],
                colorscale=colorscale,
                showscale=True,
                colorbar={"title": "Copy Number"}, 
                cmin = 0,
                cmax = 14
            ),
            text=chromosomes_names,
            hovertemplate='position: %{x}<br>Scaled Sequencing Coverage: %{y}',
            showlegend=False 
        ))

        scatter_fig.update_layout(
            title='Scatter Plot',
            xaxis=dict(
                title='Chromosome',
                tickvals=cumulative_sizes,
                ticktext=chromosomes_names + [''],
                tickmode='array',
                ticklabelmode='period',
                ticklabelposition='outside',
                tickformat='d',
                range=[0, cumulative_sizes[-1] + chromosome_sizes[-1]], 
            ),
        )
        scatter_fig.update_yaxes(title_text="Scaled sequencing coverage")
        n_clicks_apply_scatter = 0

        return scatter_fig, n_clicks_apply_scatter

        # scatter_fig = go.Figure(data=go.Scatter(
        #     x=scatter_df[x_val] if x_val != "chrom" else [start + cumulative_sizes[chromosomes_names.index(chrom)] for start, chrom in zip(scatter_df['start'], scatter_df['chrom'])],
        #     y=scatter_df['modal_corrected']*scatter_df['assignment'],
        #     mode='markers',
        #     marker=dict(
        #         size=5,
        #         color=scatter_df['copy_number'],
        #         colorscale=colorscale,
        #         showscale=True,
        #         colorbar={"title": "Copy Number"}, 
        #     ),
        #     text=chromosomes_names,
        #     hovertemplate='Position: %{x}<br>Scaled Sequencing Coverage: %{y}',
        #     showlegend=False 
        # ))
        # n_clicks_apply_scatter = 0


    return go.Figure(), n_clicks_apply_scatter

@app.callback(
    Output('multiplicity', 'figure'),
    [Input('heatmap', 'clickData'),
    Input('heatmap-state', 'data')],
    [State('cell-id-dropdown', 'value'),
     State('x-axis-dropdown-scatter', 'value'),
     State('fil-cond-dropdown-scatter', 'value'),
     State('input-val-scatter', 'value')]
)
def update_multiplicity(clickData, heatmap_state, cell_id, x_axis_val, cond_type, input_val):
    # print( "print n clicks apply scatter in multiplicity",n_clicks_apply_scatter)
    if clickData is not None and "apply-button-scatter" != ctx.triggered_id:
        
        # Get the rows for the selected 'y_value'
        selected_cell_id = clickData['points'][0]['y']
        # HERE CHANGE THE TABLE WE ARE QUERYING FROM
        statement = text("SELECT cell_id,training_cell_id,ref_condition,modal_ploidy,state,num_bins,two_coverage,total_coverage FROM gpfit WHERE cell_id = :param1")
        statement = statement.bindparams(param1=selected_cell_id)

        multiplicity_df = pd.read_sql(statement, engine)
        # print(multiplicity_df.columns)


        sql_query = text("SELECT cell_id,chrom,two,total,copy_number,assignment FROM details WHERE cell_id = :param1")

        sql_query = sql_query.bindparams(param1=selected_cell_id)

        df_coverage_order = pd.read_sql(sql_query, engine)
        df_coverage_order = df_coverage_order.groupby(['copy_number', 'assignment']).apply(get_coverage_order).reset_index()

        df_coverage_order.rename({0: 'order'}, axis=1, inplace=True)

        if max(df_coverage_order['order']) >= 6:
            scale_values = 1000000
            scale_label = 'Mb'
        elif max(df_coverage_order['order']) >= 3:
            scale_values = 1000
            scale_label = 'Kb'
        else:
            scale_values = 1
            scale_label = 'Bases'

        dfg1 = multiplicity_df[multiplicity_df['state'] == 2]

        dfg1 = dfg1[dfg1['training_cell_id'] != '']

        dfg1 = dfg1[dfg1['ref_condition'] == 'G1']
        # print("dfg1",dfg1)

        dfg2 = multiplicity_df[multiplicity_df['state'] == 2]

        dfg2 = dfg2[dfg2['training_cell_id'] != '']

        dfg2 = dfg2[dfg2['ref_condition'] == 'G2']

        dfg_test_1 = multiplicity_df[multiplicity_df['state'] == 2]

        dfg_test_1 = dfg_test_1.loc[dfg_test_1['training_cell_id'].isnull()]
        # print(dfg_test_1)

        dfg_test_1 = dfg_test_1[dfg_test_1['ref_condition'] == 'G1']

        dfg_test_2 = multiplicity_df[multiplicity_df['state'] == 2]

        dfg_test_2 = dfg_test_2.loc[dfg_test_2['training_cell_id'].isnull()]
        # print(dfg_test_2)

        dfg_test_2 = dfg_test_2[dfg_test_2['ref_condition'] == 'G2']

        multiplicity_fig = make_subplots(rows=1, cols=2, subplot_titles=("State 2","State 4"))
        multiplicity_fig.add_trace(go.Scatter(
            x=dfg1['total_coverage']/scale_values,
            y=dfg1['two_coverage']/scale_values,
            mode='markers',
            marker=dict(
                size=4,
                color='#d3d3d3'
            ),
            showlegend=False
        ))
        multiplicity_fig.add_trace(go.Scatter(
            x=dfg2['total_coverage']/scale_values,
            y=dfg2['two_coverage']/scale_values,
            mode='markers',
            marker=dict(
                size=4,
                color='#fcc3ac'
            ),
            showlegend=False
        ))

        
        if (dfg_test_1.empty or dfg_test_1['num_bins'].iloc[0] == 0):
            multiplicity_fig.add_trace(go.Scatter(
            ))
        else:
            multiplicity_fig.add_trace(go.Scatter(
                x=dfg_test_1['total_coverage']/scale_values,
                y=dfg_test_1['two_coverage']/scale_values,
                mode='markers',
                marker=dict(
                    symbol="cross",
                    size = 17,
                    color='#d3d3d3'
                ),
                showlegend=False
            ))
        
        if (dfg_test_2.empty):
            multiplicity_fig.add_trace(go.Scatter(
            ))
        else:
            # print("dfg_test_2 cross is  being put in here")
            multiplicity_fig.add_trace(go.Scatter(
                x=dfg_test_2['total_coverage']/scale_values,
                y=dfg_test_2['two_coverage']/scale_values,
                mode='markers',
                marker=dict(
                    symbol="cross",
                    size = 17,
                    color='#fcc3ac'
                ),
                showlegend=False
            ))
        # multiplicity_fig.update_layout(
        #     title='Ploidy coverage curves',
        # )
        # multiplicity_fig.update_yaxes(title_text="Mb covered")
        # multiplicity_fig.update_xaxes(title_text="Mb sequenced")
        

        # MULTIPLICITY REF

        dfg1_ref = multiplicity_df[multiplicity_df['state'] == 4]

        dfg1_ref = dfg1_ref[dfg1_ref['training_cell_id'] != '']

        dfg1_ref = dfg1_ref[dfg1_ref['ref_condition'] == 'G1']


        dfg2_ref = multiplicity_df[multiplicity_df['state'] == 4]

        dfg2_ref = dfg2_ref[dfg2_ref['training_cell_id'] != '']

        dfg2_ref = dfg2_ref[dfg2_ref['ref_condition'] == 'G2']


        dfg_test_1_ref = multiplicity_df[multiplicity_df['state'] == 4]

        dfg_test_1_ref = dfg_test_1_ref.loc[dfg_test_1_ref['training_cell_id'].isnull()]

        dfg_test_1_ref = dfg_test_1_ref[dfg_test_1_ref['ref_condition'] == 'G1']


        dfg_test_2_ref = multiplicity_df[multiplicity_df['state'] == 4]

        dfg_test_2_ref = dfg_test_2_ref.loc[dfg_test_2_ref['training_cell_id'].isnull()]

        dfg_test_2_ref = dfg_test_2_ref[dfg_test_2_ref['ref_condition'] == 'G2']

        multiplicity_fig.add_trace(go.Scatter(
            x=dfg1_ref['total_coverage']/scale_values,
            y=dfg1_ref['two_coverage']/scale_values,
            mode='markers',
            marker=dict(
                size=4,
                color='#d3d3d3'
            ),
            showlegend=False
        ),
        row=1, col=2)
        multiplicity_fig.add_trace(go.Scatter(
            x=dfg2_ref['total_coverage']/scale_values,
            y=dfg2_ref['two_coverage']/scale_values,
            mode='markers',
            marker=dict(
                size=4,
                color='#fcc3ac'
            ),
            showlegend=False
        ),
        row=1, col=2)


        if (dfg_test_1_ref.empty):
            multiplicity_fig.add_trace(go.Scatter(
            ))
        else:

            multiplicity_fig.add_trace(go.Scatter(
                x=dfg_test_1_ref['total_coverage']/scale_values,
                y=dfg_test_1_ref['two_coverage']/scale_values,
                mode='markers',
                marker=dict(
                    symbol="cross",
                    size = 17,
                    color='#d3d3d3'
                ),
                showlegend=False
            ),
            row=1, col=2)

        if (dfg_test_2_ref.empty or dfg_test_2_ref['num_bins'].iloc[0] == 0):
            multiplicity_fig.add_trace(go.Scatter(
            ))
        else:

            multiplicity_fig.add_trace(go.Scatter(
                x=dfg_test_2_ref['total_coverage']/scale_values,
                y=dfg_test_2_ref['two_coverage']/scale_values,
                mode='markers',
                marker=dict(
                    symbol="cross",
                    size = 17,
                    color='#fcc3ac'
                ),
                showlegend=False
            ),
            row=1, col=2)

        multiplicity_fig.update_layout(title_text="Ploidy Coverage Curves")

        multiplicity_fig.update_layout(coloraxis=dict(colorscale=colorscale))

        df_2 = multiplicity_df[multiplicity_df['state'] == 2]
        df_4 = multiplicity_df[multiplicity_df['state'] == 4]
        multiplicity_fig.add_annotation(
                xref="x domain",
                yref="y domain",
                x=0,
                y=1,
                text=f"{df_2['num_bins'].iloc[0]} bins",
                showarrow=False,
                font=dict(
                    size=16,
                    color="#000000"
                    ),

                row=1,

                col=1
            )
        multiplicity_fig.add_annotation(
                xref="x domain",
                yref="y domain",
                x=0,
                y=1,
                text= f"{df_4['num_bins'].iloc[0]} bins",
                showarrow=False,
                font=dict(
                    size=16,
                    color="#000000"
                    ),
                row=1,

                col=2
            )
        return multiplicity_fig

    return go.Figure()

run = False
if __name__ == '__main__':
    if not run:
        run = True
        # main()
        webbrowser.open_new_tab('http://127.0.0.1:8050/')
        print('finished main')
        app.run(debug=True, use_reloader=False)
    
    
