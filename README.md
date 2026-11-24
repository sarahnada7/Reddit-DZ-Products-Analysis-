# Reddit DZ Products Analysis 🇩🇿

## 📌 Project Overview

This project focuses on collecting, cleaning, analyzing, and visualizing Reddit discussions about products in Algeria.

Using data collected from Algerian Reddit communities, the project explores product-related discussions, identifies popular categories and brands, and extracts useful insights through statistical analysis and machine learning.

## 🔍 Project Features

### 1. Data Collection
- Scraping product-related discussions from Reddit.
- Identifying products using predefined keywords.
- Categorizing products into four categories: Food, Electronics, Appliances, and Cosmetics.

### 2. Data Cleaning & Preprocessing
- Handling missing values.
- Removing duplicate posts.
- Removing URLs, special characters, and stopwords.
- Cleaning and normalizing text.

### 3. Data Analysis
- Statistical analysis of reviews and engagement.
- Word frequency analysis.
- Category-based comparisons.
- K-Nearest Neighbors (KNN) classification to predict product categories.

### 4. Data Visualization
Generating charts to explore:
- Product category distribution.
- Average scores and comments.
- Review lengths.
- Most mentioned products and brands.

### 5. Interactive Dashboard
An interactive dashboard built with Dash and Plotly to explore the collected data and visualize insights.

## 🛠️ Technologies Used

- Python
- Pandas & NumPy
- Matplotlib & Plotly
- Scikit-learn
- Requests
- Dash & Dash Bootstrap Components

## 📂 Project Structure

```text
reddit_DZproducts_analysis/
│
├── 01_scraping_reddit.py
├── 02_cleaning_data.py
├── 03_preprocessing_show.py
├── 04_data_analysis.py
├── 05_visualization.py
├── dashboard.py
│
├── data/
│   ├── raw/
│   └── cleaned/
│
├── outputs/
│   ├── charts/
│   └── reports/
│
└── assets/
```

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib plotly scikit-learn requests urllib3 tqdm dash dash-bootstrap-components
   ```

3. Run the scripts in order:
   ```bash
   python 01_scraping_reddit.py
   python 02_cleaning_data.py
   python 03_preprocessing_show.py
   python 04_data_analysis.py
   python 05_visualization.py
   ```

4. Launch the interactive dashboard:
   ```bash
   python dashboard.py
   ```

## 📊 Results

The collected dataset contains **425 product-related Reddit posts** across four categories.

The KNN classification model achieved an accuracy of **61.18%** using K=5.

## 🎯 Objective

The main objective is to explore Algerian consumers' discussions and interests regarding different products through data analysis, visualization, and machine learning.
