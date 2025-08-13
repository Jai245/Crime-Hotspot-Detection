# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import streamlit as st

# %%
df = pd.read_csv(r"C:\Users\rosel\Downloads\New folder\LA Crime\LA_Crime_Data_from_2020_to_2024.csv", chunksize=100000)
df = pd.concat(df)

# %%
df.head()

# %%
df.columns = df.columns.str.strip()

# %%
# Compute correlation matrix
corr_matrix = df.select_dtypes(include=["number"]).corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# %%
#Removing few columns due to multicollinearity
df.drop(columns=['DR_NO','Rpt Dist No'],inplace=True)

# %%
print(df.columns.tolist())

# %%
invalid_dates = df[pd.to_datetime(df['DATE OCC'], errors='coerce').isna()]
print(invalid_dates[['DATE OCC']])

# %%
# Automatically infers the format and handles bad data safely
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], format='%m/%d/%Y', errors='coerce')

# Now safely extract the weekday name
df['weekday'] = df['DATE OCC'].dt.day_name()

# %%
# Ensure TIME OCC is zero-padded
df['TIME OCC'] = df['TIME OCC'].astype(str).str.zfill(4)

# %%
# Extract hour and minute
df['Hour'] = df['TIME OCC'].str[:2].astype(int)
df['Minute'] = df['TIME OCC'].str[2:].astype(int)

# Creating "Time of Day" buckets
def time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['TimeOfDay'] = df['Hour'].apply(time_of_day)

# %%
df['YearMonth'] = df['DATE OCC'].dt.to_period('M')

# %%
from sklearn.cluster import KMeans

# Drop rows with missing coordinates
df_clean = df.dropna(subset=['LAT', 'LON'])

# Use KMeans to group into zones (e.g., 10)
kmeans = KMeans(n_clusters=10, random_state=42)
df_clean['Zone'] = kmeans.fit_predict(df_clean[['LAT', 'LON']])

# Merge back if needed
df['Zone'] = df_clean['Zone']

# %%
inertias = []
K = range(2, 20)
for k in K:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(df_clean[['LAT', 'LON']])
    inertias.append(model.inertia_)

plt.plot(K, inertias, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Zone Count')
plt.show()

# %%
from sklearn.cluster import KMeans

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Zone'] = kmeans.fit_predict(df[['LAT', 'LON']])

# %%
# LA bounding box (roughly)
df = df[(df['LAT'] > 33) & (df['LAT'] < 35) & (df['LON'] < -117) & (df['LON'] > -119)]

# %%
plt.scatter(df['LON'], df['LAT'], c=df['Zone'], cmap='tab10', alpha=0.5)
plt.title("Crime Zones Based on Clustering")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# %%
#Group by Location to count how many crimes happened at or near the same location
hotspot_counts = df.groupby(['LAT', 'LON']).size().reset_index(name='crime_count')

# %%
#Merge Back to Original Data
df = df.merge(hotspot_counts, on=['LAT', 'LON'], how='left')

# %%
print(df.columns)

# %%
print(df['crime_count'].isna().sum())

# %%
threshold = df['crime_count'].quantile(0.90)
df['Hotspot'] = df['crime_count'].apply(lambda x: 'Hotspot' if x >= threshold else 'Normal')

# %%
colors = {'Hotspot': 'red', 'Normal': 'blue'}
plt.figure(figsize=(8, 6))
for label in df['Hotspot'].unique():
    subset = df[df['Hotspot'] == label]
    plt.scatter(subset['LON'], subset['LAT'], 
                c=colors[label], label=label, s=10, alpha=0.6)

plt.title("Crime Hotspots in the City")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc='upper right')
plt.show()

# %%
print(df.columns)

# %%
# After clustering
df['ClusterCrimeCount'] = df.groupby('Zone')['Zone'].transform('count')
threshold = df['ClusterCrimeCount'].quantile(0.9)

# Labeling Hotspots
df['Hotspot'] = df['ClusterCrimeCount'].apply(lambda x: 'Hotspot' if x >= threshold else 'Normal')

# %%
import folium
from folium.plugins import FastMarkerCluster

# Define the map center
map_center = [df['LAT'].mean(), df['LON'].mean()]

# Create the base map
m = folium.Map(location=map_center, zoom_start=11)

# Prepare data for FastMarkerCluster
locations = df[['LAT', 'LON']].values.tolist()
FastMarkerCluster(data=locations).add_to(m)

# Display the map
m

# %%
hotspot_trend = df.groupby(['Zone', 'YearMonth']).size().reset_index(name='CrimeCount')

# %%
df.columns

# %%
top_crimes = df[df['Hotspot'] == 'Hotspot']['Crm Cd Desc'].value_counts().head(10)
print(top_crimes)

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Encode target
df['Hotspot'] = (df['crime_count'] > 50).astype(int)

# Drop rows where target is NaN
df_encoded = df.dropna(subset=['Hotspot'])

# Assuming 'DATE OCC' is a datetime column
df_encoded['DATE_OCC_year'] = df_encoded['DATE OCC'].dt.year
df_encoded['DATE_OCC_month'] = df_encoded['DATE OCC'].dt.month
df_encoded['DATE_OCC_day'] = df_encoded['DATE OCC'].dt.day

# Convert categorical columns to category dtype
categorical_columns = ['Vict Sex', 'Vict Descent', 'Status', 'Zone', 'Cross Street', 'weekday', 'TimeOfDay', 'YearMonth']
for col in categorical_columns:
    df_encoded[col] = df_encoded[col].astype('category')

# Now drop any columns you don't need
X = df_encoded.drop(columns=[
    'Hotspot', 'crime_count', 'ClusterCrimeCount', 
    'Date Rptd', 'AREA NAME', 'Crm Cd Desc', 'Mocodes', 
    'Premis Desc', 'Weapon Desc', 'Status Desc', 'DATE OCC'
])
y = df_encoded['Hotspot']

# Convert 'YearMonth' from period to category
if 'YearMonth' in X.columns:
    X['YearMonth'] = X['YearMonth'].astype(str).astype('category')

# Convert all object columns to category
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category')

# %%
print(df.shape)               # original size
print(df['Hotspot'].value_counts(dropna=False))  # see what's in the 'Hotspot' column
print(df_encoded.shape)       # shape after dropping NaNs

# %%
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Optuna objective
def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'tree_method': 'auto',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
    }

    model = xgb.XGBClassifier(**param, enable_categorical=True)
    model.fit(X_train, y_train)

    preds = model.predict(X_valid)
    acc = accuracy_score(y_valid, preds)
    return acc

# %%
# Run Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Show results
print("Best hyperparameters:", study.best_params)
print("Best accuracy:", study.best_value)

# %%
# Get best model
best_params = study.best_params
best_model = xgb.XGBClassifier(**best_params, enable_categorical=True)
best_model.fit(X_train, y_train)

# %%
from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the test set
y_pred = best_model.predict(X_valid)

# Accuracy
accuracy = accuracy_score(y_valid, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification report
report = classification_report(y_valid, y_pred)
print("Classification Report:")
print(report)

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_valid, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# %%
import shap

# Use the underlying booster model
booster = best_model.get_booster()

# Prepare validation data for booster
dvalid = xgb.DMatrix(X_valid, enable_categorical=True)

# Predict SHAP values
shap_values = booster.predict(dvalid, pred_contribs=True)

# visualize one example
explainer = shap.TreeExplainer(best_model)
shap_values_plot = explainer.shap_values(X_valid)

# %%
import os

# Create force plot for the first prediction
force_html_path = "assets/xgb_shap_force.html"
os.makedirs("assets", exist_ok=True)

# Generate force plot HTML
shap.initjs()
force_plot = shap.force_plot(
    base_value=explainer.expected_value,
    shap_values=shap_values[0][:-1],  # Remove last element
    features=X_valid.iloc[0],
    matplotlib=False
)

shap.save_html(force_html_path, force_plot)

# %%
# SHAP summary DataFrame
shap_df = pd.DataFrame(shap_values[:, :-1], columns=X_valid.columns)
shap_importance = shap_df.abs().mean().sort_values(ascending=False).reset_index()
shap_importance.columns = ['Feature', 'Mean SHAP Value']

# %%
top_feature = shap_importance.iloc[0]['Feature']

# %%
feature_names = X.columns.tolist()
print("Loaded feature_columns:", feature_names)

# %%
import joblib

# Save KMeans model
joblib.dump(kmeans, "kmeans_zone.pkl")

# Save the trained model
joblib.dump(model, "crime_model.pkl")

# Save the SHAP explainer
joblib.dump(explainer, "shap_explainer.pkl")

# Save the feature names
joblib.dump(X_valid.columns.tolist(), "feature_columns.pkl")

# Save the SHAP values
joblib.dump(shap_values, "shap_values.pkl")

print("Model, explainer, and feature names saved successfully!")

# %%
# Load the KMeans model
kmeans = joblib.load("kmeans_zone.pkl")

# Load the model
model = joblib.load("crime_model.pkl")

# Load the SHAP explainer
explainer = joblib.load("shap_explainer.pkl")

# Load the feature names
feature_columns = joblib.load("feature_columns.pkl")

shap_values = joblib.load("shap_values.pkl")

# Print out feature names
print("Loaded feature_columns:", feature_names)

# %%
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from apscheduler.schedulers.background import BackgroundScheduler
import os

# Data update function (runs daily)
def update_data():
    print("Updating crime data...")

    # Load raw data
    df = pd.read_csv(r"C:\Users\rosel\Downloads\New folder\LA Crime\raw_crime_data.csv")

    # Initial row count
    initial_rows = len(df)
    print(f"Loaded {initial_rows} rows from raw CSV.")

    # Parse dates
    df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
    print(f"Invalid DATE OCC values: {df['DATE OCC'].isna().sum()}")
    
    # Report invalid dates
    invalid_dates = df['DATE OCC'].isna().sum()
    print(f"Invalid DATE OCC values: {invalid_dates}")

    # Try converting LAT/LON and report issues
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')

    invalid_coords = df[['LAT', 'LON']].isna().any(axis=1).sum()
    print(f"Rows with invalid LAT/LON: {invalid_coords}")

    # Drop rows with bad values
    df = df.dropna(subset=['DATE OCC', 'LAT', 'LON'])

    # Add time features
    df['TimeOfDay'] = df['DATE OCC'].dt.to_period('M').astype(str)
    df['weekday'] = df['DATE OCC'].dt.day_name()

    final_rows = len(df)
    print(f"Rows after cleaning: {final_rows}")

    # Save debug version before writing final CSV
    df.to_csv('latest_crime_data_debug.csv', index=False)  # <-- temporary debugging file
    df.to_csv('latest_crime_data.csv', index=False)

    if df.empty:
        print("⚠️ WARNING: All rows were dropped after cleaning. Check raw data.")

# %%
# Schedule daily updates
update_data()
scheduler = BackgroundScheduler()
scheduler.add_job(update_data, 'interval', days=1)
scheduler.start()

# %%
# Load and process data
df = pd.read_csv('latest_crime_data.csv')
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], format='%m/%d/%Y', errors='coerce')
df['weekday'] = df['DATE OCC'].dt.day_name()
df['TimeOfDay'] = df['DATE OCC'].dt.to_period('M').astype(str)
df = df.dropna(subset=['LAT', 'LON'])
df['LAT'] = df['LAT'].astype(float)
df['LON'] = df['LON'].astype(float)

# %%
import dash
from dash import dcc, html
import plotly.express as px
from sklearn.cluster import KMeans

# Ensure 'DATE OCC' is datetime
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')

# Clean and convert 'TIME OCC'
df['TIME OCC'] = pd.to_numeric(df['TIME OCC'], errors='coerce')  # Converts to float, invalid to NaN
df = df.dropna(subset=['TIME OCC'])  # Drop rows where TIME OCC is invalid
df['TIME OCC'] = df['TIME OCC'].astype(int)

# Extract hour/minute and create TimeOfDay label
df['Minute'] = df['TIME OCC'] % 100
df['Hour'] = df['TIME OCC'] // 100
df['TimeOfDay'] = pd.cut(
    df['Hour'],
    bins=[-1, 5, 11, 17, 21, 24],
    labels=['Night', 'Morning', 'Afternoon', 'Evening', 'Night'],
    ordered=False,
    include_lowest=True
)

# Extract date parts
df['YearMonth'] = df['DATE OCC'].dt.to_period('M').astype(str)
df['DATE_OCC_year'] = df['DATE OCC'].dt.year
df['DATE_OCC_month'] = df['DATE OCC'].dt.month
df['DATE_OCC_day'] = df['DATE OCC'].dt.day
df['weekday'] = df['DATE OCC'].dt.day_name()

# KMeans clustering on valid coordinates
coords = df[['LAT', 'LON']].dropna()

if not coords.empty:
    kmeans = KMeans(n_clusters=6, random_state=42)
    df.loc[coords.index, 'Zone'] = kmeans.fit_predict(coords)
else:
    print("No data available for clustering.")
    df['Zone'] = np.nan  # Safe fallback

# %%
# Load and engineer features
df = pd.read_csv('latest_crime_data.csv')

# Convert DATE OCC to datetime
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')

# Optional clean-up or checks
df = df.dropna(subset=['DATE OCC'])

app = dash.Dash(__name__)
server = app.server

# Your layout and callbacks (example layout)
app.layout = html.Div([
    html.H1("Crime Hotspots Dashboard", style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.Label("Crime Type:"),
            dcc.Dropdown(
                id='crime-type-filter',
                options=[{'label': ct, 'value': ct} for ct in sorted(df['Crm Cd Desc'].dropna().unique())],
                placeholder="All types",
                multi=True
            ),
        ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),

        html.Div([
            html.Label("Area:"),
            dcc.Dropdown(
                id='area-filter',
                options=[{'label': a, 'value': a} for a in sorted(df['AREA'].dropna().unique())],
                placeholder="All areas"
            ),
        ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),

        html.Div([
            html.Label("Weekday:"),
            dcc.Dropdown(
                id='weekday-filter',
                options=[{'label': day, 'value': day} for day in df['weekday'].unique()],
                placeholder="All days"
            ),
        ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),

        html.Div([
            html.Label("Date Range:"),
            dcc.DatePickerRange(
                id='date-range-filter',
                min_date_allowed=df['DATE OCC'].min().date(),
                max_date_allowed=df['DATE OCC'].max().date(),
                start_date=df['DATE OCC'].min().date(),
                end_date=df['DATE OCC'].max().date(),
            ),
        ], style={'width': '24%', 'display': 'inline-block'}),
    ], style={'paddingBottom': '20px'}),

    html.Div([
        html.Div([
            html.Label("Time of Day:"),
            dcc.Dropdown(
                id='timeofday-filter',
                options=[{'label': t, 'value': t} for t in df['TimeOfDay'].unique()],
                placeholder="Select time of day",
                searchable=True
            ),
        ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '1%'}),

        html.Div([
            html.Label("Victim Gender:"),
            dcc.Dropdown(
                id='vict-sex-filter',
                options=[{'label': sex, 'value': sex} for sex in df['Vict Sex'].dropna().unique()],
                placeholder="Victim sex",
                searchable=True
            ),
        ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '1%'}),

        html.Div([
            html.Label("Victim Age Range:"),
            dcc.RangeSlider(
                id='vict-age-filter',
                min=df['Vict Age'].min(),
                max=df['Vict Age'].max(),
                step=1,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True},
                value=[df['Vict Age'].min(), df['Vict Age'].max()]
            )
        ], style={'width': '24%', 'display': 'inline-block'})
    ], style={'paddingBottom': '20px'}),

    html.Div([
        html.Label("Status:"),
        dcc.Dropdown(
            id='status-filter',
            options=[{'label': s, 'value': s} for s in df['Status Desc'].dropna().unique()],
            placeholder="Select status",
            searchable=True
        )
    ], style={'width': '24%', 'display': 'inline-block', 'paddingBottom': '20px'}),

    html.Div(id='active-filters-display'),
    html.Iframe(id='map-graph', width='100%', height='600'),

    
    html.Div([
        html.H3("SHAP Summary Plot"),
        html.Img(src="/assets/shap_summary.png", style={'width': '100%', 'height': 'auto'}),
    ], style={'marginTop': '40px'}),

    html.Div([
        html.H2("XGBoost SHAP Feature Importance", style={'textAlign': 'center'}),

        dcc.Graph(
            id='xgb-shap-bar-plot',
            figure=px.bar(
                shap_importance.head(20),
                x='Mean SHAP Value',
                y='Feature',
                orientation='h',
                title="Top 20 SHAP Features (XGBoost)"
            )
        ),

    html.Div([
        html.Label("Select Feature for XGBoost SHAP Dependence Plot:"),
        dcc.Dropdown(
            id='xgb-shap-feature-dropdown',
            options=[{'label': col, 'value': col} for col in X_valid.columns],
            value=top_feature
        ),
        dcc.Graph(id='xgb-shap-dependence-plot')
    ], style={'marginTop': '30px'})
    ], style={'marginTop': '60px'}),

    html.Div([
        html.H3("XGBoost SHAP Force Plot (First Instance)"),
        html.Iframe(src="/assets/xgb_shap_force.html", width="100%", height="400")
    ], style={'marginTop': '40px'}),
    
    dcc.Graph(id='shap-bar-plot')

])

# %%
from dash import Input, Output
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import io

@app.callback(
    Output('map-graph', 'srcDoc'),
    Input('crime-type-filter', 'value'),
    Input('area-filter', 'value'),
    Input('weekday-filter', 'value'),
    Input('date-range-filter', 'start_date'),
    Input('date-range-filter', 'end_date'),
    Input('timeofday-filter', 'value'),
    Input('vict-sex-filter', 'value'),
    Input('vict-age-filter', 'value'),
    Input('status-filter', 'value')
)
    
def update_map(crime_types, area, weekday, start_date, end_date, time_of_day, vict_sex, age_range, status):
    # Start with full dataframe
    filtered_df = df.copy()

    # Apply filters
    if crime_types:
        filtered_df = filtered_df[filtered_df['Crm Cd Desc'].isin(crime_types)]

    if area:
        filtered_df = filtered_df[filtered_df['AREA'] == area]

    if weekday:
        filtered_df = filtered_df[filtered_df['weekday'] == weekday]

    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['DATE OCC'] >= start_date) & (filtered_df['DATE OCC'] <= end_date)]

    if time_of_day:
        filtered_df = filtered_df[filtered_df['TimeOfDay'] == time_of_day]

    if vict_sex:
        filtered_df = filtered_df[filtered_df['Vict Sex'] == vict_sex]

    if age_range:
        min_age, max_age = age_range
        filtered_df = filtered_df[(filtered_df['Vict Age'] >= min_age) & (filtered_df['Vict Age'] <= max_age)]
        
    if status:
        filtered_df = filtered_df[filtered_df['Status Desc'] == status]

    active_filters = html.Ul([
        html.Li(f"Crime Types: {', '.join(crime_types) if crime_types else 'All'}"),
        html.Li(f"Area: {area if area else 'All'}"),
        html.Li(f"Weekday: {weekday if weekday else 'All'}"),
        html.Li(f"Date Range: {start_date} to {end_date}"),
        html.Li(f"Time of Day: {time_of_day if time_of_day else 'All'}"),
        html.Li(f"Victim Sex: {vict_sex if vict_sex else 'All'}"),
        html.Li(f"Victim Age: {min_age} to {max_age}"),
        html.Li(f"Status: {status if status else 'All'}")
    ])

# Create map with Folium
    

    if not filtered_df.empty:
        m = folium.Map(
        location=[filtered_df['LAT'].mean(), filtered_df['LON'].mean()],
        zoom_start=11,
        tiles='CartoDB positron')  # or 'Stamen Toner', 'OpenStreetMap'
        marker_cluster = MarkerCluster().add_to(m)

        for _, row in filtered_df.iterrows():
            popup_text = f"""
            <b>Crime:</b> {row['Crm Cd Desc']}<br>
            <b>Date:</b> {row['DATE OCC']}<br>
            <b>Victim Sex:</b> {row['Vict Sex']}<br>
            <b>Victim Age:</b> {row['Vict Age']}<br>
            <b>Status:</b> {row['Status']}
            """
            folium.Marker(
                location=[row['LAT'], row['LON']],
                popup=popup_text
            ).add_to(marker_cluster)

        return m.get_root().render()
    else:
        return "<h3>No data available for selected filters.</h3>"

@app.callback(
    Output('xgb-shap-dependence-plot', 'figure'),
    Input('xgb-shap-feature-dropdown', 'value')
)
def update_xgb_dependence_plot(selected_feature):
    fig = px.scatter(
        x=X_valid[selected_feature],
        y=shap_df[selected_feature],
        labels={'x': selected_feature, 'y': 'SHAP Value'},
        title=f"SHAP Dependence Plot for {selected_feature} (XGBoost)"
    )
    return fig

# %%
@app.callback(
    Output('shap-bar-plot', 'figure'),
    Input('crime-type-filter', 'value'),
    Input('area-filter', 'value'),
    Input('weekday-filter', 'value'),
    Input('status-filter', 'value'),
    Input('vict-sex-filter', 'value'),
    Input('timeofday-filter', 'value'),
    Input('vict-age-filter', 'value'),
    Input('date-range-filter', 'start_date'),
    Input('date-range-filter', 'end_date'),
)
def update_shap_plot(crime_type, area, weekday, status, gender, time_of_day, age_range, start_date, end_date):
    filtered_df = df.copy()

    # Ensure datetime
    filtered_df['DATE OCC'] = pd.to_datetime(filtered_df['DATE OCC'], errors='coerce')

    # Apply filters
    if crime_type:
        filtered_df = filtered_df[filtered_df['Crm Cd Desc'].isin(crime_type)]
    if area:
        filtered_df = filtered_df[filtered_df['AREA'] == area]
    if weekday:
        filtered_df = filtered_df[filtered_df['weekday'] == weekday]
    if status:
        filtered_df = filtered_df[filtered_df['Status Desc'] == status]
    if gender:
        filtered_df = filtered_df[filtered_df['Vict Sex'] == gender]
    if time_of_day:
        filtered_df = filtered_df[filtered_df['TimeOfDay'] == time_of_day]
    if age_range and len(age_range) == 2:
        filtered_df = filtered_df[(filtered_df['Vict Age'] >= age_range[0]) & (filtered_df['Vict Age'] <= age_range[1])]
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['DATE OCC'] >= start_date) & (filtered_df['DATE OCC'] <= end_date)]

    # Create needed columns if not present
    if 'Hour' not in filtered_df:
        filtered_df['Hour'] = filtered_df['TIME OCC'] // 100
        filtered_df['Minute'] = filtered_df['TIME OCC'] % 100
    if 'YearMonth' not in filtered_df:
        filtered_df['YearMonth'] = filtered_df['DATE OCC'].dt.to_period('M').astype(str)
    if 'DATE_OCC_year' not in filtered_df:
        filtered_df['DATE_OCC_year'] = filtered_df['DATE OCC'].dt.year
    if 'DATE_OCC_month' not in filtered_df:
        filtered_df['DATE_OCC_month'] = filtered_df['DATE OCC'].dt.month
    if 'DATE_OCC_day' not in filtered_df:
        filtered_df['DATE_OCC_day'] = filtered_df['DATE OCC'].dt.day
    if 'Zone' not in filtered_df:
        filtered_df['Zone'] = filtered_df['AREA']

    X_filtered = filtered_df[feature_columns]

    if X_filtered.empty:
        fig = px.bar(title="No data available after applying filters.")
        fig.update_layout(title_x=0.5)
        return fig

    # PREPROCESS to fix dtype and NaNs
    X_filtered = X_filtered.fillna(-999)  # You can also consider other fill methods
    for col in X_filtered.select_dtypes(include='object').columns:
        X_filtered[col] = X_filtered[col].astype('category').cat.codes

    dfilter = xgb.DMatrix(X_filtered)
    shap_values = booster.predict(dfilter, pred_contribs=True)
    
    shap_df = pd.DataFrame(shap_values, columns=list(X_filtered.columns) + ['bias'])
    shap_df.drop(columns='bias', inplace=True)

    shap_importance = shap_df.abs().mean().sort_values(ascending=False).reset_index()
    shap_importance.columns = ['Feature', 'Mean SHAP Value']

    fig = px.bar(
        shap_importance.head(20),
        x='Mean SHAP Value',
        y='Feature',
        orientation='h',
        title='Top SHAP Features for Filtered Data'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(title_x=0.5)

    return fig


# %%
if __name__ == '__main__':
    app.run(debug=True)

# %%



