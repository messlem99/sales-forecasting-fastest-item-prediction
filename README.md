# Sales Forecasting & Fastest Selling Item Prediction

This repository contains a Streamlit web application that forecasts sales and identifies the fastest selling item using historical sales data and Facebook Prophet. The app includes interactive visualizations powered by Plotly and supports both English and French languages.

## Features

- **Data Upload:** Users can upload a CSV file with historical daily sales.
- **Multilingual Support:** The app is available in English and French.
- **Missing Data Handling:** Options to fill missing days with zero sales or to interpolate missing values.
- **Forecasting:** Uses Prophet to predict future sales for each item.
- **Visualization:** Displays forecasted results with confidence intervals using Plotly.
- **Results Saving:** Saves forecast tables and plots to a results folder.

## Requirements

- Python 3.7+
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Prophet](https://facebook.github.io/prophet/) (install as `prophet`)
- [Plotly](https://plotly.com/python/)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/messlem99/sales-forecasting-fastest-item-prediction.git
   cd sales-forecasting-fastest-item-prediction
