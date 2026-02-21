# customer-segment-analysis
This project focuses on analyzing customer purchase behavior and segmenting customers into distinct groups using the K-Means clustering algorithm. The goal is to identify different types of customers based on their income and spending patterns so that businesses can make data-driven marketing decisions.

The dataset is first preprocessed using Python libraries such as Pandas and NumPy, where missing values are handled, categorical variables are encoded, and relevant features are selected. After preprocessing, the K-Means clustering algorithm is applied to group customers into different segments.

To determine the optimal number of clusters, the Elbow Method is used, which helps in identifying the point where adding more clusters does not significantly improve the model. The results are then visualized using scatter plots to clearly show how customers are grouped based on their purchasing behavior.

This project demonstrates how unsupervised machine learning techniques can be used in real-world business scenarios to improve customer targeting, personalize marketing strategies, and enhance overall customer experience.

# Objectives
Analyze customer purchasing behavior
Perform data cleaning and preprocessing
Apply K-Means clustering algorithm
Determine optimal number of clusters using Elbow Method
Visualize customer segments
Generate business insights from clusters

# Workflows
1. Data Loading using Pandas  
2. Data Inspection and Cleaning  
3. Feature Selection  
4. Elbow Method to determine optimal clusters  
5. K-Means Clustering  
6. Cluster Visualization  
7. Business Insight Generation

# Feature Scaling
Since K-Means relies on distance calculations (Euclidean distance), feature scaling was applied using **StandardScaler** to ensure that both income and spending score contribute equally to clustering.
Without scaling, features with larger numeric ranges could dominate the clustering process.
# Results
- Optimal number of clusters determined using Elbow Method: **5**
- Customers segmented into:
  - High Income – High Spending (Target Customers)
  - High Income – Low Spending
  - Medium Income – Medium Spending
  - Low Income – High Spending
  - Low Income – Low Spending
# Streamlit Web Application
A simple interactive web app was built using Streamlit.
Users can:
- Enter customer income and spending behavior
- Predict the customer segment
- Understand business classification

#  Tech Stack
Python
Pandas
NumPy
Matplotlib / Seaborn
Scikit-learn
Joblib

# Used dataset from kaggle:
Link: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python?select=Mall_Customers.csv

## To run it locally:
```bash
python -m streamlit run app.py
