import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('cs2_market_items.csv')

print("First few rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

item_names = df['name'].copy()

df = df.drop(['name'], axis=1, errors='ignore')
df['float_value'] = df['float_value'].fillna(0)
df['paint_seed'] = df['paint_seed'].fillna(0)

categorical_features = ['rarity', 'quality']
numerical_features = ['float_value', 'paint_seed', 'stat_trak', 'souvenir']

X = df[categorical_features + numerical_features]
y = df['price']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"\nMean Absolute Error: ${mae:.2f}")
print(f"R² Score: {r2:.2f}")

df_results = pd.DataFrame({
    'Actual Price': y_test.reset_index(drop=True),
    'Predicted Price': predictions,
    'Item Name': item_names.loc[y_test.index].reset_index(drop=True)
})

df_results['Difference'] = df_results['Predicted Price'] - df_results['Actual Price']
df_results['% Difference'] = (df_results['Difference'] / df_results['Actual Price']) * 100

def classify_price(row, threshold=20):
    if row['% Difference'] > threshold:
        return 'Undervalued (Good Deal!)'
    elif row['% Difference'] < -threshold:
        return 'Overvalued (Beware)'
    else:
        return 'Fairly Valued'

df_results['Value Status'] = df_results.apply(classify_price, axis=1)

undervalued = df_results[df_results['Value Status'] == 'Undervalued (Good Deal!)'] \
                .sort_values(by='% Difference', ascending=False).head(5)

overvalued = df_results[df_results['Value Status'] == 'Overvalued (Beware)'] \
              .sort_values(by='% Difference').head(5)

print("\n🔥 Top 5 Undervalued Skins (Great Deals):")
print(undervalued[['Item Name', 'Actual Price', 'Predicted Price', '% Difference']])

print("\n💸 Top 5 Overvalued Skins (Beware of These):")
print(overvalued[['Item Name', 'Actual Price', 'Predicted Price', '% Difference']])

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_results,
    x='Actual Price',
    y='Predicted Price',
    hue='Value Status',
    palette={'Overvalued (Beware)': 'red', 'Fairly Valued': 'gray', 'Undervalued (Good Deal!)': 'green'},
    alpha=0.7
)

plt.plot([df_results['Actual Price'].min(), df_results['Actual Price'].max()],
         [df_results['Actual Price'].min(), df_results['Actual Price'].max()], 'r--')
plt.title('Market Mismatches: Overvalued vs Undervalued CS2 Skins')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.legend(title='Price Status')
plt.tight_layout()
plt.show()

plot_df = df_results.copy()
fig = px.scatter(
    plot_df,
    x='Actual Price',
    y='Predicted Price',
    color='Value Status',
    hover_data=['Item Name'],
    title='Interactive View: Actual vs Predicted Prices (Color-coded by Value Status)',
    color_discrete_map={
        'Undervalued (Good Deal!)': 'green',
        'Fairly Valued': 'gray',
        'Overvalued (Beware)': 'red'
    }
)

fig.add_shape(type="line", x0=plot_df['Actual Price'].min(), y0=plot_df['Actual Price'].min(),
              x1=plot_df['Actual Price'].max(), y1=plot_df['Actual Price'].max(),
              line=dict(color="Black", dash="dash"))
fig.show()