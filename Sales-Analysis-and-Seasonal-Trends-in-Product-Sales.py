# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from mlxtend.frequent_patterns import apriori, association_rules
import calendar

# Step 2: Load the dataset
df = pd.read_csv(r"C:\Users\LENOVO\Documents\Project_Data_SetFolder\Sales Data.csv")

# Step 3: Check the first few rows and columns
df.head()

# Step 4: Check for missing values
print("Missing Values:")
print(df.isnull().values.any())

# Step 5: Check for duplicates
duplicates = df[df.duplicated()]
if not duplicates.empty:
    print(duplicates)
else:
    print("No duplicates found.")

# Step 6: Clean the data
df = df.drop(columns=['Unnamed: 0'])
df['Order ID'] = df['Order ID'].astype(str)
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Product'] = df['Product'].astype('category')
df['City'] = df['City'].astype('category')
df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce')
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
df['Month'] = df['Month'].astype(str)
df['Month'] = df['Month'].apply(lambda x: calendar.month_name[int(x)])
print("Cleaned Data_Types:")
print(df.dtypes)

# Step 7: Normalize sales revenue
scaler = MinMaxScaler()
df['Normalized_Sales'] = scaler.fit_transform(df['Sales'].values.reshape(-1, 1))

# Step 8: Detect and replace outliers
z_scores = stats.zscore(df['Sales'])
threshold = 3
outliers = (z_scores > threshold) | (z_scores < -threshold)
median_sales = df['Sales'].median()
df.loc[outliers, 'Sales'] = median_sales
z_scores = stats.zscore(df['Sales'])
df['Outlier'] = (z_scores > threshold) | (z_scores < -threshold)

# Step 9: Aggregate data
product_data = df.groupby('Product').agg({'Quantity Ordered': 'sum', 'Sales': 'sum'}).reset_index()

# Step 10: Rank products
product_data['Quantity Rank'] = product_data['Quantity Ordered'].rank(ascending=False)
product_data['Sales Rank'] = product_data['Sales'].rank(ascending=False)

# Step 11: Seasonal analysis
monthly_sales = df.groupby(['Product', 'Month']).agg({'Quantity Ordered': 'sum', 'Sales': 'sum'}).reset_index()

# Step 12: Visualize seasonal trends
plt.figure(figsize=(12, 6))
for product in monthly_sales['Product'].unique():
    data = monthly_sales[monthly_sales['Product'] == product]
    plt.plot(data['Month'], data['Sales'], label=product)
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Seasonal Trends in Product Sales')
plt.legend()
plt.show()

# Step 13: Identify seasonal patterns
decomposition = seasonal_decompose(monthly_sales['Sales'], period=12)
seasonal_pattern = decomposition.seasonal

# Step 14: Compare products
mean_sales_per_product = monthly_sales.groupby('Product')['Sales'].mean()
mean_sales_per_product.plot(kind='bar', figsize=(12, 6))
plt.xlabel('Product')
plt.ylabel('Average Sales')
plt.title('Comparison of Average Sales per Product')
plt.show()

# Step 15: Identify popular products and seasonal trends
# Group by 'Product' and calculate total quantity ordered and sales revenue
product_analysis = df.groupby('Product').agg({'Quantity Ordered': 'sum', 'Sales': 'sum'})

# Identify popular products by sorting
popular_products_quantity = product_analysis.sort_values(by='Quantity Ordered', ascending=False)
popular_products_sales = product_analysis.sort_values(by='Sales', ascending=False)

# Seasonal trends analysis
df['Month'] = df['Order Date'].dt.month
df['Month'] = df['Month'].apply(lambda x: calendar.month_name[x])
seasonal_trends = df.groupby(['Product', 'Month']).agg({'Quantity Ordered': 'sum', 'Sales': 'sum'})

# Print the results
print("\nPopular Products by Quantity Ordered:")
print(popular_products_quantity.head())

print("\nPopular Products by Sales Revenue:")
print(popular_products_sales.head())

print("\nSeasonal Trends Analysis:")
print(seasonal_trends.head())

# Step 16: Visualize popular products by quantity and sales revenue
fig, axs = plt.subplots(2, 1, figsize=(12, 12))

# Plot popular products by quantity ordered
sns.barplot(x=popular_products_quantity.index, y='Quantity Ordered', data=popular_products_quantity, ax=axs[0])
axs[0].set_title('Popular Products by Quantity Ordered', fontsize=14)
axs[0].set_xlabel('Product', fontsize=12)
axs[0].set_ylabel('Total Quantity Ordered', fontsize=12)
axs[0].tick_params(axis='x', labelrotation=45)

# Plot popular products by sales revenue
sns.barplot(x=popular_products_sales.index, y='Sales', data=popular_products_sales, ax=axs[1])
axs[1].set_title('Popular Products by Sales Revenue', fontsize=14)
axs[1].set_xlabel('Product', fontsize=12)
axs[1].set_ylabel('Total Sales Revenue', fontsize=12)
axs[1].tick_params(axis='x', labelrotation=45)

# Adjust the layout and spacing
plt.tight_layout()

# Show the plots
plt.show()
