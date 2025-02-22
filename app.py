import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\ideal\anaconda3\envs\DIC\src\Online Retail.csv")
    return df

df = load_data()

# Preprocess data
def preprocess_data(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Total_Price'] = df['Quantity'] * df['UnitPrice']
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    # Use .loc to avoid SettingWithCopyWarning
    df.loc[:, 'Description'] = df['Description'].str.strip().str.replace(r'[^\w\s]', '', regex=True)
    df.loc[:, 'ProductCategory'] = df['Description'].apply(lambda x: 'Gift' if 'GIFT' in x.upper() else 'Set' if 'SET' in x.upper() else 'Regular')
    df.loc[:, 'Year'] = df['InvoiceDate'].dt.year
    df.loc[:, 'Month'] = df['InvoiceDate'].dt.month
    df.loc[:, 'DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    
    df.loc[:, 'Season'] = df['Month'].apply(get_season)
    Q1 = df['Total_Price'].quantile(0.25)
    Q3 = df['Total_Price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Total_Price'] >= Q1 - 1.5 * IQR) & (df['Total_Price'] <= Q3 + 1.5 * IQR)]
    def valid_product(desc):
        invalid_desc = ['damaged', 'sample', 'postage', 'lost', 'found', 'check', 'adjustment']
        return not any(pattern in desc.lower() for pattern in invalid_desc)
    df = df[df['Description'].apply(valid_product)]
    avg_prices = df.groupby('StockCode')['UnitPrice'].median()
    def correct_decimal(price, avg_price):
        if pd.notnull(price) and pd.notnull(avg_price):
            if price > 100 * avg_price:
                return price / 100
            elif price < avg_price / 100:
                return price * 100
        return price
    df['UnitPrice'] = df.apply(lambda row: correct_decimal(row['UnitPrice'], avg_prices.get(row['StockCode'], np.nan)), axis=1)
    customer_totals = df.groupby('CustomerID')['Total_Price'].sum()
    def categorize_customer(total):
        if total > 10000:
            return 'High Value'
        elif total > 5000:
            return 'Medium Value'
        else:
            return 'Low Value'
    df['CustomerSegment'] = df['CustomerID'].map(customer_totals).apply(categorize_customer)
    purchase_frequency = df.groupby('CustomerID')['InvoiceNo'].nunique()
    df['PurchaseFrequency'] = df['CustomerID'].map(purchase_frequency)
    # RFM Analysis
    max_date = df['InvoiceDate'].max()
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (max_date - x.max()).days,
        'InvoiceNo': 'count',
        'Total_Price': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm.reset_index()
    r_labels = range(4, 0, -1)
    f_labels = range(1, 5)
    m_labels = range(1, 5)
    r_quartiles = pd.qcut(rfm['Recency'], q=4, labels=r_labels)
    f_quartiles = pd.qcut(rfm['Frequency'], q=4, labels=f_labels)
    m_quartiles = pd.qcut(rfm['Monetary'], q=4, labels=m_labels)
    rfm['R'] = r_quartiles
    rfm['F'] = f_quartiles
    rfm['M'] = m_quartiles
    rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
    def rfm_segment(row):
        if row['RFM_Score'] in ['444', '434', '443', '433']:
            return 'Best Customers'
        elif row['RFM_Score'] in ['441', '442', '432', '423', '424']:
            return 'Loyal Customers'
        elif row['RFM_Score'] in ['311', '422', '421', '412', '411']:
            return 'Lost Customers'
        elif row['RFM_Score'] in ['211', '212', '221']:
            return 'Lost Cheap Customers'
        else:
            return 'Other'
    rfm['RFM_Segment'] = rfm.apply(rfm_segment, axis=1)
    df = df.merge(rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'RFM_Score', 'RFM_Segment']], on='CustomerID', how='left')
    return df

df = preprocess_data(df)

# Define df_sample
sample_size = 10000
df_sample = df.sample(n=sample_size, random_state=42)

# Prediction Page
st.title("Predict Cluster for RFM Score")

# Scale features
features = ['Recency', 'Frequency', 'Monetary']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_sample[features])

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df_sample['KMeans_Cluster'] = kmeans.fit_predict(scaled_features)

# Input fields for prediction
recency = st.number_input("Recency (days)", min_value=0, max_value=365, value=30)
frequency = st.number_input("Frequency (number of purchases)", min_value=0, max_value=8000, value=10)
monetary = st.number_input("Monetary (total spend)", min_value=0.0, max_value=80000.0, value=500.0)

# Scale the input features
input_features = np.array([[recency, frequency, monetary]])
input_scaled = scaler.transform(input_features)

# Predict the cluster
predicted_cluster = kmeans.predict(input_scaled)

# Visualize the input point on a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the clusters
scatter = ax.scatter(
    df_sample['Recency'], 
    df_sample['Frequency'], 
    df_sample['Monetary'], 
    c=df_sample['KMeans_Cluster'], 
    cmap='viridis', 
    alpha=0.6, 
    edgecolors='w', 
    s=100
)

# Plot the input point
ax.scatter(recency, frequency, monetary, color='red', label='Input Point', s=200, edgecolors='k')

# Plot cluster centers
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], color='black', marker='X', s=300, label='Centroids')

# Set labels and title
ax.set_title('K-Means Clustering: Recency vs Frequency vs Monetary')
ax.set_xlabel('Recency (days)')
ax.set_ylabel('Frequency (no of purchases)')
ax.set_zlabel('Monetary (total spend)')
plt.colorbar(scatter, ax=ax, label='Cluster')
ax.legend()
st.pyplot(fig)

st.write(f"The predicted cluster for the given RFM score is: {predicted_cluster[0]}")
