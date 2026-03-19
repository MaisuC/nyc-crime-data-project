import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output

df = pd.read_csv('dataset.csv', dtype={'ADDR_PCT_CD': str, 8: str})

# cleaning the data
df['BORO_NM'] = df['BORO_NM'].fillna('UNKNOWN').astype(str).str.upper()
df['ADDR_PCT_CD'] = df['ADDR_PCT_CD'].fillna('UNKNOWN').astype(str)
df['OFNS_DESC'] = df['OFNS_DESC'].fillna('UNKNOWN').astype(str).str.upper()
df['VIC_AGE_GROUP_CLEAN'] = df['VIC_AGE_GROUP_CLEAN'].fillna('UNKNOWN').astype(str).str.upper()

# Filter out unknown option for the dropdown selection
boro_options = sorted([b for b in df['BORO_NM'].unique() if b != 'UNKNOWN'])

age_order = ['<18', '18-24', '25-44', '45-64', '65+', 'UNKNOWN']
df['VIC_AGE_GROUP_CLEAN'] = pd.Categorical(df['VIC_AGE_GROUP_CLEAN'], categories=age_order, ordered=True)

# starting plotly dash
app = Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#f4f7f9', 'padding': '25px', 'fontFamily': 'sans-serif'}, children=[
    
    html.Div([
        html.H1("NYPD Incident Analysis Dashboard", style={'color': '#002d62', 'margin': '0'}),
        html.P("Comparative Analysis of Precinct Data", style={'color': '#666'})
    ], style={'textAlign': 'center', 'marginBottom': '30px'}),

    # filter panel
    html.Div([
        html.Div([
        html.Label("1. Select Borough", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='boro-dropdown',
            options=[{'label': b, 'value': b} for b in boro_options],
            value=boro_options[0] if boro_options else None,
            clearable=False
        ),
        ], style={'width': '45%'}),

        html.Div([
            html.Label("2. Select Precinct"),
            dcc.Dropdown(id='precinct-dropdown', clearable=False),
        ], style={'width': '45%'}),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'backgroundColor': 'white', 
              'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.05)', 'marginBottom': '25px'}),

    
    html.Div([
        dcc.Graph(id='crime-type-chart')
    ], style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'marginBottom': '25px'}),

    html.Br(),
    
    html.Div([
        html.Div([
            dcc.Graph(id='precinct-comp-chart')
        ], style={'flex': '1', 'marginRight': '10px', 'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}),
        
        html.Div([
            dcc.Graph(id='age-dist-chart')
        ], style={'flex': '1', 'marginLeft': '10px', 'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
    ], style={'display': 'flex'})
])

# callback for plotly
@app.callback(
    [Output('precinct-dropdown', 'options'),
     Output('precinct-dropdown', 'value')],
    [Input('boro-dropdown', 'value')]
)
def update_precinct_list(selected_boro):
    # Get precincts for the selected borough
    relevant_precincts = sorted(df[df['BORO_NM'] == selected_boro]['ADDR_PCT_CD'].unique())
    options = [{'label': f"Precinct {p}", 'value': p} for p in relevant_precincts]
    return options, (relevant_precincts[0] if relevant_precincts else None)

@app.callback(
    [Output('crime-type-chart', 'figure'),
     Output('precinct-comp-chart', 'figure'),
     Output('age-dist-chart', 'figure')],
    [Input('precinct-dropdown', 'value'),
     Input('boro-dropdown', 'value')]
)
def update_visuals(selected_precinct, selected_boro):
    pct_df = df[df['ADDR_PCT_CD'] == selected_precinct]
    
    # top crime counts
    ct_counts = pct_df['OFNS_DESC'].value_counts().nlargest(10).reset_index()
    ct_counts.columns = ['Type', 'Count']
    crime_fig = go.Figure(go.Bar(y=ct_counts['Type'], x=ct_counts['Count'], orientation='h', marker_color='#002d62'))
    crime_fig.update_layout(
        title=f"Top 10 Crimes in Precinct {selected_precinct}",
        xaxis_title="Number of Incidents",
        yaxis_title="Offense Description",
        template="plotly_white",
        yaxis={'categoryorder':'total ascending'}
    )

    # Comparison Graph
    boro_df = df[df['BORO_NM'] == selected_boro]
    comp_counts = boro_df['ADDR_PCT_CD'].value_counts().reset_index()
    comp_counts.columns = ['Precinct', 'Count']
    colors = ['#d63031' if p == selected_precinct else '#cccccc' for p in comp_counts['Precinct']]
    
    comp_fig = go.Figure(go.Bar(x=comp_counts['Precinct'], y=comp_counts['Count'], marker_color=colors))
    comp_fig.update_layout(
        title="Specified Precinct Crime count vs Other Precinct Crime count",
        xaxis_title="Precinct ID (ADDR_PCT_CD)",
        yaxis_title="Total Crime Count",
        template="plotly_white"
    )

    # Victim Age
    age_counts = pct_df.groupby('VIC_AGE_GROUP_CLEAN', observed=False).size().reset_index(name='Count')
    age_fig = go.Figure(go.Bar(x=age_counts['VIC_AGE_GROUP_CLEAN'], y=age_counts['Count'], marker_color='#0984e3'))
    age_fig.update_layout(
        title=f"Victim Demographics (Precinct {selected_precinct})",
        xaxis_title="Age Group",
        yaxis_title="Number of Victims",
        template="plotly_white"
    )

    return crime_fig, comp_fig, age_fig

if __name__ == '__main__':
    app.run(debug=True)