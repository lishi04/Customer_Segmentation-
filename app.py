import pandas as pd
import numpy as np
from flask import Flask, render_template
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import LabelEncoder
from dash.dash_table import DataTable
import dash_bootstrap_components as dbc
import networkx as nx

# ===== Load & Clean Dataset =====
df = pd.read_csv('saltish_data.csv')

df['Payment Method'] = df['Payment Method'].str.strip().str.lower()
df['Order'] = df['Order'].str.strip().str.lower()
df['Address'] = df['Address'].str.strip().str.lower()
df['Customer Name'] = df['Customer Name'].str.strip()
df['Contact Number'] = df['Contact Number'].str.strip()
df['Total Bill'] = pd.to_numeric(df['Total Bill'], errors='coerce')

df.reset_index(inplace=True)
df.rename(columns={'index': 'Bill No'}, inplace=True)

# ===== Clustering Logic =====
payment_map = {
    'easypaisa': 'Digital Wallet',
    'easy paisa': 'Digital Wallet',
    'jazz cash': 'Digital Wallet',
    'jazzcash': 'Digital Wallet',
    'cash on delivery': 'Cash',
    'cash': 'Cash',
    'card': 'Card',
    'bank payment': 'Bank'
}

def cluster_payment(df):
    return df['Payment Method'].map(payment_map).fillna('Other')

def cluster_bill(df):
    bins = [0, 500, 1000, 2000, np.inf]
    labels = ['Low', 'Medium', 'High', 'Very High']
    return pd.cut(df['Total Bill'], bins=bins, labels=labels)

def cluster_order(df):
    return df['Order']

def cluster_address(df):
    return df['Address']

# ===== ARM on Orders =====
def prepare_basket(df):
    df_orders = df[['Bill No', 'Order']].copy()
    df_orders['Order'] = df_orders['Order'].str.split(',')
    df_exploded = df_orders.explode('Order')
    df_exploded['Order'] = df_exploded['Order'].str.strip()

    basket = df_exploded.groupby(['Bill No', 'Order'])['Order'].count().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    return basket

def perform_arm():
    basket = prepare_basket(df)
    freq_items = apriori(basket, min_support=0.003, use_colnames=True)
    if freq_items.empty:
        return pd.DataFrame(columns=['Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift'])
    rules = association_rules(freq_items, metric="lift", min_threshold=0.01)
    rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
    return rules

# ===== ARM Network Graph Visualization =====
def create_arm_graph(rules_df):
    G = nx.DiGraph()

    # Add nodes and edges from rules
    for _, row in rules_df.iterrows():
        antecedents = row['antecedents'].split(', ')
        consequents = row['consequents'].split(', ')
        for ant in antecedents:
            G.add_node(ant)
            for cons in consequents:
                G.add_node(cons)
                G.add_edge(ant, cons, weight=row['lift'])

    pos = nx.spring_layout(G, k=0.5, iterations=50)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='#FF5733',
            size=20,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Association Rules Network Graph',
                        title_x=0.5,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    return fig

# ===== Flask & Dash Setup =====
server = Flask(__name__)

app = dash.Dash(
    __name__,
    server=server,
    routes_pathname_prefix='/dashboard/',
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

feature_options = ['Payment Method', 'Order', 'Total Bill', 'Address', 'Combined']

app.layout = dbc.Container([
    html.H2("Saltish AI Dashboard", className='text-center mt-3 mb-4'),

    dcc.Tabs(id='tabs', value='cluster-tab', children=[
        dcc.Tab(label='🔍 Clustering', value='cluster-tab'),
        dcc.Tab(label='🛒 Association Rules (Orders)', value='arm-tab'),
    ]),

    html.Div(id='tabs-content', className='p-3')
], fluid=True)

@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'cluster-tab':
        return html.Div([
            html.Label("Select Feature for Clustering:", className='mb-2'),
            dcc.Dropdown(id='feature-dropdown',
                         options=[{'label': i, 'value': i} for i in feature_options],
                         value='Payment Method',
                         clearable=False,
                         style={'width': '50%'}),
            html.Div(id='cluster-output', className='mt-4')
        ])
    else:
        rules_df = perform_arm()
        if rules_df.empty:
            return html.Div("No association rules found with the given support and lift thresholds.",
                            className='text-danger')

        arm_graph_fig = create_arm_graph(rules_df)

        return html.Div([
            html.H5("Market Basket Analysis on Orders", className='mb-3'),
            dcc.Graph(figure=arm_graph_fig),
            DataTable(
                columns=[{"name": i.capitalize(), "id": i} for i in rules_df.columns],
                data=rules_df.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '5px'},
                page_size=10,
                filter_action='native',
                sort_action='native',
                sort_mode='multi',
                export_format='csv'
            )
        ])

@app.callback(Output('cluster-output', 'children'), Input('feature-dropdown', 'value'))
def update_output(feature):
    clustered = df.copy()
    fig = None

    if feature == 'Payment Method':
        clustered['Cluster'] = cluster_payment(clustered)
        fig = px.scatter(clustered, x='Total Bill', y='Customer ID', color='Cluster',
                         title='Total Bill vs Customer ID by Payment Method Cluster',
                         labels={'Total Bill': 'Total Bill', 'Customer ID': 'Customer ID'})
    elif feature == 'Order':
        clustered['Cluster'] = cluster_order(clustered)
        fig = px.scatter(clustered, x='Total Bill', y='Customer ID', color='Cluster',
                         title='Total Bill vs Customer ID by Order Cluster',
                         labels={'Total Bill': 'Total Bill', 'Customer ID': 'Customer ID'})
    elif feature == 'Total Bill':
        clustered['Cluster'] = cluster_bill(clustered)
        fig = px.scatter(clustered, x='Customer ID', y='Total Bill', color='Cluster',
                         title='Customer ID vs Total Bill by Bill Cluster',
                         labels={'Customer ID': 'Customer ID', 'Total Bill': 'Total Bill'})
    elif feature == 'Address':
        clustered['Cluster'] = cluster_address(clustered)
        fig = px.scatter(clustered, x='Total Bill', y='Customer ID', color='Cluster',
                         title='Total Bill vs Customer ID by Address Cluster',
                         labels={'Total Bill': 'Total Bill', 'Customer ID': 'Customer ID'})
    elif feature == 'Combined':
        clustered['PaymentCluster'] = cluster_payment(clustered)
        clustered['BillCluster'] = cluster_bill(clustered)
        clustered['Cluster'] = clustered['PaymentCluster'].astype(str) + " | " + clustered['BillCluster'].astype(str)

        le = LabelEncoder()
        clustered['ClusterNum'] = le.fit_transform(clustered['Cluster'].astype(str))

        fig = px.scatter(clustered, x='Total Bill', y='Customer ID', color='Cluster',
                         title='Total Bill vs Customer ID by Combined Cluster',
                         labels={'Total Bill': 'Total Bill', 'Customer ID': 'Customer ID'})
    else:
        return html.Div("Invalid feature selected.", className='text-danger')

    summary = clustered['Cluster'].value_counts().reset_index()
    summary.columns = ['Cluster', 'Count']

    summary_table = DataTable(
        columns=[{"name": i, "id": i} for i in summary.columns],
        data=summary.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
        page_size=10,
        export_format='csv'
    )

    sample_cols = ['Customer ID', 'Customer Name', 'Order', 'Total Bill', 'Payment Method', 'Address', 'Cluster']
    data_preview = DataTable(
        columns=[{"name": i, "id": i} for i in sample_cols],
        data=clustered[sample_cols].head(20).to_dict('records'),
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
        export_format='csv'
    )

    return html.Div([
        html.H5(f"Clustering Based on {feature}"),
        dcc.Graph(figure=fig),
        html.H6("Cluster Summary:"),
        summary_table,
        html.H6("Sample Data in Clusters:"),
        data_preview
    ])

# Flask route for homepage with Bootstrap UI
@server.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
