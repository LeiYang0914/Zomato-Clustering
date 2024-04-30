import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import string
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from sklearn.neighbors import NearestNeighbors

# Download NLTK stopwords data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Define global stopwords list for sentiment analysis
stop_words = set(stopwords.words('english'))

def replace_empty_with_nan(df, column_names):
    for column in column_names:
        df[column] = df[column].apply(lambda x: np.nan if x == [] else x)
    return df

def clean_binary_columns(value):
    if value == 'Yes':
        return 1
    elif value == 'No':
        return 0
    else:
        return pd.NaT
    
def clean_rate_column(df, column_name):


    # Remove '/5' from the column values
    df[column_name] = df[column_name].str.replace('/5', '')

    # Convert the column to numeric, coercing errors to NaN
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

    return df

def convert_to_numeric(df, column_name):
    # Remove commas from the column values
    df[column_name] = df[column_name].str.replace(',', '')

    # Convert the column to numeric, coercing errors to NaN
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

    return df

def fillna_with_median(df, column_names):

    for column_name in column_names:
        median_value = df[column_name].median()
        df[column_name] = df[column_name].fillna(median_value)

    return df

def clean_text(text):
    # Remove non-alphanumeric characters
    text = ''.join([char for char in text if char not in string.punctuation])
    # Convert text to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def analyze_sentiment(review):
    review = clean_text(review)  # Clean the text
    testimonial = TextBlob(review)
    polarity = testimonial.sentiment.polarity
    return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"


def get_removal_condition(df, votes_threshold, rate_threshold, average_cost_threshold):
    removal_condition = (df['votes'] >= votes_threshold) | \
                        (df['rate'] <= rate_threshold) | \
                        (df['average_cost'] >= average_cost_threshold)

    return removal_condition

def clean_categorical_columns(value, common_values):
    if pd.isnull(value):
        return pd.NaT
    elif value in common_values:
        return value
    else:
        return 'Other'

def get_top_values(df, column_name, n=10):
    """
    Get the top n most common values in the column.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    column_name (str): The name of the column to find the top values for.
    n (int): The number of top values to retrieve.

    Returns:
    list: A list of the top n values in the column.
    """
    return df[column_name].value_counts().index.tolist()[:n]

def clean_columns_using_top_values(df, columns):
    """
    Clean the specified columns in the DataFrame based on their top n values.

    Parameters:
    df (pandas.DataFrame): The DataFrame to operate on.
    columns (list of str): The names of the columns to clean.

    Returns:
    pandas.DataFrame: The DataFrame with cleaned categorical columns.
    """
    cleaned_df = df.copy()
    for column in columns:
        top_values = get_top_values(cleaned_df, column)
        cleaned_df[column] = cleaned_df[column].apply(clean_categorical_columns, common_values=top_values)
    return cleaned_df

def add_expensive_flag(df, cost_column, flag_column):
    """
    Add a binary column to the DataFrame indicating whether the cost is above the median.

    Parameters:
    df (pandas.DataFrame): The DataFrame to operate on.
    cost_column (str): The name of the column with cost values.
    flag_column (str): The name of the new binary column to be added.

    Returns:
    pandas.DataFrame: The DataFrame with the added binary column.
    """
    median_cost = df[cost_column].median()
    df[flag_column] = df[cost_column].apply(lambda x: 1 if x > median_cost else 0)
    df = df.drop(columns='average_cost')
    return df

def prepare_data_for_clustering(df, categorical_columns, numerical_columns):
    # One-Hot Encode Categorical Columns
    encoder = OneHotEncoder(sparse_output=False)
    encoded_categorical = encoder.fit_transform(df[categorical_columns])
    encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out())
    
    # Scale Numerical Columns
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    # Concatenate Encoded Categorical DataFrame and Scaled Numerical DataFrame
    final_df = pd.concat([df[numerical_columns].reset_index(drop=True),
                          encoded_categorical_df.reset_index(drop=True)], axis=1)
    
    return final_df

def preprocessing(df):
    df.rename(columns={'approx_cost(for two people)': 'average_cost', 'listed_in(city)': 'locality',}, inplace=True)    
    df = replace_empty_with_nan(df, ['menu_item', 'reviews_list'])
    df['online_order'] = df['online_order'].apply(clean_binary_columns)
    df['book_table'] = df['book_table'].apply(clean_binary_columns)
    df = clean_rate_column(df, 'rate')
    df = convert_to_numeric(df, 'average_cost')
    df = fillna_with_median(df, ['average_cost', 'rate'])
    df.dropna(subset=['rest_type', 'cuisines'], inplace=True)
    df.dropna(subset=['reviews_list'], inplace=True)
    df.rename(columns={'reviews_list': 'review_sentiment'}, inplace=True)
    stop_words = set(stopwords.words('english'))
    df['review_sentiment'] = df['review_sentiment'].apply(analyze_sentiment)
    df.drop('url', axis=1, inplace=True)
    df.drop('phone', axis=1, inplace=True)
    df.drop('address', axis=1, inplace=True)
    df.drop('name', axis=1, inplace=True)
    df.drop('location', axis=1, inplace=True)
    df.drop('dish_liked', axis=1, inplace=True)
    df.drop('menu_item',axis=1,inplace=True)
    votes_threshold = 6876
    rate_threshold = 2.4
    average_cost_threshold = 4500
    removal_condition = get_removal_condition(df, votes_threshold, rate_threshold, average_cost_threshold)
    df = df[~removal_condition]
    columns_to_clean = ['locality', 'rest_type', 'cuisines', 'listed_in(type)']
    df = clean_columns_using_top_values(df, columns_to_clean)
    df = add_expensive_flag(df, 'average_cost', 'expensive')
    categorical_columns = ['rest_type', 'cuisines', 'listed_in(type)', 'locality', 'review_sentiment']
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    final_df_for_clustering = prepare_data_for_clustering(df, categorical_columns, numerical_cols)
    return final_df_for_clustering

def calculate_optimal_eps(data):
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    gradient = np.diff(distances)
    optimal_index = np.argmax(gradient)
    return distances[optimal_index + 1]

# Function to perform DBSCAN clustering
def cluster_data(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data)
    return clusters

# Function to calculate silhouette score
def calculate_silhouette(data, clusters):
    noise = clusters == -1
    valid_clusters = clusters[~noise]
    valid_points = data[~noise]
    if len(np.unique(valid_clusters)) > 1:
        sil_score = silhouette_score(valid_points, valid_clusters)
        return sil_score
    else:
        return None

def cluster_data(data, method, params):
    if method == 'DBSCAN':
        eps = params.get('eps', 0.5)
        min_samples = params.get('min_samples', 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(data)
    elif method == 'GMM':
        n_components = params.get('n_components', 3)
        model = GaussianMixture(n_components=n_components, random_state=42)
        model.fit(data)
        labels = model.predict(data)
    return labels

def plot_results(data, labels):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    fig, ax = plt.subplots()
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, ax=ax)
    plt.title('Clustering Results Visualization')
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    st.pyplot(fig)


# Streamlit interface
st.title("Clustering Dashboard")
algorithm = st.selectbox("Choose the clustering algorithm", ['DBSCAN', 'GMM'])

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = preprocessing(data)  # Apply your preprocessing function

    params = {}
    if algorithm == 'DBSCAN':
        eps = st.slider("DBSCAN: Select eps", 0.1, 10.0, 0.5)
        min_samples = st.slider("DBSCAN: Select min_samples", 1, 50, 5)
        params = {'eps': eps, 'min_samples': min_samples}
    elif algorithm == 'GMM':
        n_components = st.slider("GMM: Select number of clusters", 1, 10, 3)
        params = {'n_components': n_components}

    if st.button("Cluster"):
        with st.spinner('Clustering data...'):
            labels = cluster_data(data, algorithm, params)
            silhouette = silhouette_score(data, labels)
            st.success(f'Silhouette Score: {silhouette:.2f}')
            plot_results(data, labels)
