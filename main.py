"""
Sales Forecasting & Fastest Selling Item Prediction App

This Streamlit app allows users to upload a CSV file with historical sales data,
handles missing data, and performs forecasting using Prophet for each item. It
identifies the fastest selling item over the chosen forecast period and provides
interactive visualizations with Plotly.
"""

import streamlit as st
import pandas as pd
import logging
import io
import os
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go

# Configure logging for debugging and info purposes
logging.basicConfig(level=logging.INFO)

# Translation dictionaries for multilingual support (English and French)
translations = {
    "en": {
        "title": "Sales Forecasting & Fastest Selling Item Prediction",
        "instructions_title": "**Instructions:**",
        "instructions_upload": "Please upload a CSV file containing historical daily sales data.",
        "instructions_keywords": "Ensure your data includes columns with names similar to the following (case-insensitive):",
        "keywords_item": "- For item name: item, product, name, id, sku, article, produit, nom, référence",
        "keywords_sales": "- For sales quantity: sales, quantity, sold, amount, volume, ventes, quantité, vendu, montant",
        "keywords_date": "- For date: date, day, time, period, jour, temps, période (YYYY-MM-DD format)",
        "instructions_minimum_records": "Each item should have a minimum of 10 records for reliable forecasting.",
        "upload_button": "Upload your CSV file",
        "upload_success": "Data uploaded and validated successfully!",
        "data_preview": "Data Preview",
        "error_columns": "Error: Could not find all required columns in your data. Ensure your CSV contains columns with names similar to: {item_keywords}, {sales_keywords}, and {date_keywords}.",
        "error_date_format": "Error: Some dates could not be parsed. Ensure they are in YYYY-MM-DD format.",
        "error_insufficient_data": "Warning: The following items have insufficient data (less than 10 records) and will not be forecasted:",
        "run_prediction": "Run Prediction",
        "running_prediction": "Running predictions...",
        "prediction_completed": "Prediction completed!",
        "forecasted_sales": "Forecasted Sales for Next {period}",
        "item_column": "Item",
        "predicted_sales_column": "Predicted Sales",
        "fastest_selling": "Fastest Selling Item (Next {period}):",
        "forecast_visualizations": "{period} Forecast Visualizations",
        "forecast_for_item": "Forecast for Item:",
        "date_label": "Date",
        "sales_label": "Sales",
        "sales_forecast_title": "Sales Forecast for {item} ({period})",
        "error_occurred": "An error occurred:",
        "forecast": "Forecast",
        "historical_sales": "Historical Sales",
        "save_results": "Save Results",
        "forecasted_sales_table": "Forecasted Sales Table",
        "forecast_period": "Select Forecast Period:",
        "next_day": "Next Day",
        "next_week": "Next Week",
        "next_10_days": "Next 10 Days",
        "next_month": "Next Month",
        "limiting_items": "Limiting to top 10 items based on historical sales.",
        "many_items_warning": "Warning: More than 50 items detected. Forecasting may take longer.",
        "confidence_interval": "Confidence Interval",
        "confidence_interval_explanation": "The shaded area represents the confidence interval, indicating the range within which future sales are likely to fall.",
        "missing_handling_option": "Select Missing Data Handling Option:",
        "fill_missing": "Fill missing days with zero sales",
        "interpolate_missing": "Estimate missing days using linear interpolation",
        "items_processing_option": "Select Items Processing Option:",
        "limit_items": "Limit to top 10 items based on historical sales",
        "process_all": "Process all items",
        "full_timeline": "Full Timeline Forecast",
        "save_folder_warning": "Results will be saved in a folder (not zipped).",
    },
    "fr": {
        "title": "Prévision des Ventes & Prédiction de l'Article le Plus Vendu",
        "instructions_title": "**Instructions :**",
        "instructions_upload": "Veuillez télécharger un fichier CSV contenant les données historiques des ventes quotidiennes.",
        "instructions_keywords": "Assurez-vous que vos données comprennent des colonnes avec des noms similaires à ceux-ci (non sensible à la casse) :",
        "keywords_item": "- Pour le nom de l'article : item, product, name, id, sku, article, produit, nom, référence",
        "keywords_sales": "- Pour la quantité des ventes : sales, quantity, sold, amount, volume, ventes, quantité, vendu, montant",
        "keywords_date": "- Pour la date : date, day, time, period, jour, temps, période (AAAA-MM-JJ)",
        "instructions_minimum_records": "Chaque article doit avoir un minimum de 10 enregistrements pour une prévision fiable.",
        "upload_button": "Télécharger votre fichier CSV",
        "upload_success": "Données téléchargées et validées avec succès !",
        "data_preview": "Aperçu des Données",
        "error_columns": "Erreur : Impossible de trouver toutes les colonnes requises dans vos données. Veuillez vous assurer que votre fichier CSV contient des colonnes avec des noms similaires à : {item_keywords}, {sales_keywords} et {date_keywords}.",
        "error_date_format": "Erreur : Certaines dates n'ont pas pu être analysées. Veuillez vous assurer qu'elles sont au format AAAA-MM-JJ.",
        "error_insufficient_data": "Avertissement : Les articles suivants n'ont pas suffisamment de données (moins de 10 enregistrements) et ne seront pas prévus :",
        "run_prediction": "Lancer la Prédiction",
        "running_prediction": "Exécution des prédictions...",
        "prediction_completed": "Prédiction terminée !",
        "forecasted_sales": "Prévisions des Ventes pour les {period} Prochains Jours",
        "item_column": "Article",
        "predicted_sales_column": "Ventes Prévues",
        "fastest_selling": "Article le Plus Vendu (Prochains {period}) :",
        "forecast_visualizations": "Visualisations des Prévisions ({period})",
        "forecast_for_item": "Prévisions pour l'Article :",
        "date_label": "Date",
        "sales_label": "Ventes",
        "sales_forecast_title": "Prévisions des Ventes pour {item} ({period})",
        "error_occurred": "Une erreur est survenue :",
        "forecast": "Prévisions",
        "historical_sales": "Ventes Historiques",
        "save_results": "Enregistrer les Résultats",
        "forecasted_sales_table": "Tableau des Prévisions de Ventes",
        "forecast_period": "Sélectionner la Période de Prévision :",
        "next_day": "Jour Suivant",
        "next_week": "Semaine Suivante",
        "next_10_days": "10 Jours Suivants",
        "next_month": "Mois Suivant",
        "limiting_items": "Limitation aux 10 articles principaux en fonction des ventes historiques.",
        "many_items_warning": "Avertissement : Plus de 50 articles détectés. La prévision peut prendre plus de temps.",
        "confidence_interval": "Intervalle de Confiance",
        "confidence_interval_explanation": "La zone ombrée représente l'intervalle de confiance, indiquant la plage dans laquelle les ventes futures sont susceptibles de se situer.",
        "missing_handling_option": "Sélectionnez l'option de gestion des données manquantes :",
        "fill_missing": "Remplacer les jours manquants par des ventes nulles",
        "interpolate_missing": "Estimer les jours manquants par interpolation linéaire",
        "items_processing_option": "Sélectionnez l'option de traitement des articles :",
        "limit_items": "Limiter aux 10 articles principaux en fonction des ventes historiques",
        "process_all": "Traiter tous les articles",
        "full_timeline": "Prévision sur toute la période",
        "save_folder_warning": "Les résultats seront enregistrés dans un dossier (non zippé).",
    },
}

def get_translation(language: str):
    """
    Retrieve translation dictionary based on selected language.
    Defaults to English if the language is not found.
    """
    return translations.get(language, translations["en"])

@st.cache_data
def predict_fastest_selling_item_prophet(data, forecast_period):
    """
    For each unique item in the data, fit a Prophet model and forecast sales.
    Returns:
      - The fastest selling item (based on total forecasted sales)
      - A dictionary with total forecasted sales per item
      - A dictionary with full forecast and period-specific forecast data for each item
    """
    forecast_results = {}
    period_forecast = {}
    fastest_item = None
    max_forecast = -float('inf')

    for item in data['item'].unique():
        # Prepare data for Prophet: sort by date and rename columns
        item_data = data[data['item'] == item].sort_values('date').rename(columns={'date': 'ds', 'sales': 'y'})
        if len(item_data) < 10:
            logging.warning(f"Insufficient data for item: {item}")
            continue

        model = Prophet()
        try:
            model.fit(item_data)
            future = model.make_future_dataframe(periods=forecast_period)
            forecast = model.predict(future)

            # Forecast for the future period (for computing totals)
            forecast_period_df = forecast.iloc[-forecast_period:].rename(columns={
                'ds': 'date', 'yhat': 'predicted_sales',
                'yhat_lower': 'lower_bound', 'yhat_upper': 'upper_bound'
            })
            # Full forecast (historical + future) for visualization
            full_forecast_df = forecast.rename(columns={
                'ds': 'date', 'yhat': 'predicted_sales',
                'yhat_lower': 'lower_bound', 'yhat_upper': 'upper_bound'
            })
            forecast_results[item] = {"full": full_forecast_df, "period": forecast_period_df}

            total_fc = forecast_period_df['predicted_sales'].sum()
            period_forecast[item] = total_fc

            if total_fc > max_forecast:
                max_forecast = total_fc
                fastest_item = item
        except Exception as e:
            logging.error(f"Error during Prophet forecasting for item {item}: {e}")

    return fastest_item, period_forecast, forecast_results

def main():
    """
    Main function to run the Streamlit app. This includes:
    - Language selection
    - CSV file upload and data validation
    - Missing data handling and item processing options
    - Forecasting and visualization
    - Saving forecast results
    """
    # Set default language
    if 'language' not in st.session_state:
        st.session_state['language'] = 'en'
        
    # Sidebar: Language selection
    st.sidebar.title("Language Selection")
    language = st.sidebar.selectbox("Select Language", ("English", "Français"), key="language_selector")
    st.session_state['language'] = "en" if language == "English" else "fr"
    t = get_translation(st.session_state['language'])

    # Main title and instructions
    st.title(t["title"])
    st.markdown(f"""
    {t["instructions_title"]}

    {t["instructions_upload"]}
    """)
    st.markdown(t["instructions_keywords"])
    st.markdown(f"- {t['keywords_item']}")
    st.markdown(f"- {t['keywords_sales']}")
    st.markdown(f"- {t['keywords_date']}")
    st.markdown(f"""
    {t["instructions_minimum_records"]}
    """)

    # File uploader for CSV file
    uploaded_file = st.file_uploader(t["upload_button"], type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            data.columns = list(map(str.lower, data.columns))

            # Define keyword lists for identifying columns
            item_keywords = ['item', 'product', 'name', 'id', 'sku', 'article', 'produit', 'nom', 'référence']
            sales_keywords = ['sales', 'quantity', 'sold', 'amount', 'volume', 'ventes', 'quantité', 'vendu', 'montant']
            date_keywords = ['date', 'day', 'time', 'period', 'jour', 'temps', 'période']

            # Identify the required columns based on keywords
            item_col = None
            sales_col = None
            date_col = None

            for col in data.columns:
                if col in item_keywords:
                    item_col = col
                elif col in sales_keywords:
                    sales_col = col
                elif col in date_keywords:
                    date_col = col

            # Check if all required columns are present
            if not all([item_col, sales_col, date_col]):
                st.error(t["error_columns"].format(
                    item_keywords=', '.join(item_keywords),
                    sales_keywords=', '.join(sales_keywords),
                    date_keywords=', '.join(date_keywords)))
                return

            # Rename columns for consistency
            data = data.rename(columns={item_col: 'item', sales_col: 'sales', date_col: 'date'})
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            if data['date'].isnull().any():
                st.error(t["error_date_format"])
                return

            # Option for handling missing data in date ranges
            missing_handling_option = st.radio(t["missing_handling_option"], 
                                               (t["fill_missing"], t["interpolate_missing"]))
            full_data = []
            for item in data['item'].unique():
                item_df = data[data['item'] == item].set_index('date').sort_index()
                start_date = item_df.index.min()
                end_date = item_df.index.max()
                if pd.notna(start_date) and pd.notna(end_date):
                    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                    reindexed_df = item_df.reindex(date_range)
                    reindexed_df['item'] = item
                    if missing_handling_option == t["fill_missing"]:
                        reindexed_df['sales'] = reindexed_df['sales'].fillna(0)
                    elif missing_handling_option == t["interpolate_missing"]:
                        reindexed_df['sales'] = reindexed_df['sales'].interpolate(method='linear')
                    full_data.append(reindexed_df.reset_index().rename(columns={'index': 'date'}))
            if full_data:
                data = pd.concat(full_data)
            else:
                st.warning("No valid date ranges found for any item to handle missing days.")

            # Option to limit the number of items processed
            items_option = st.radio(t["items_processing_option"], (t["limit_items"], t["process_all"]))
            if items_option == t["limit_items"]:
                if data['item'].nunique() > 10:
                    st.info(t["limiting_items"])
                    top_items = data.groupby('item')['sales'].sum().sort_values(ascending=False).head(10).index
                    data = data[data['item'].isin(top_items)]
            else:
                st.warning("You have chosen to process all items. This may take a lot of computation resources and time.")

            if data['item'].nunique() > 50:
                st.warning(t["many_items_warning"])

            st.success(t["upload_success"])
            st.subheader(t["data_preview"])
            st.dataframe(data.head())

            # Forecast period selection
            forecast_period_option = st.selectbox(t["forecast_period"],
                                                  (t["next_day"], t["next_week"], t["next_10_days"], t["next_month"]))
            forecast_days = 0
            forecast_period_name = ""
            if forecast_period_option == t["next_day"]:
                forecast_days = 1
                forecast_period_name = "Day"
            elif forecast_period_option == t["next_week"]:
                forecast_days = 7
                forecast_period_name = "Week"
            elif forecast_period_option == t["next_10_days"]:
                forecast_days = 10
                forecast_period_name = "10 Days"
            elif forecast_period_option == t["next_month"]:
                last_date = data['date'].max()
                first_day_next_month = pd.Timestamp(last_date.year, last_date.month, 1) + pd.DateOffset(months=1)
                first_day_following_month = first_day_next_month + pd.DateOffset(months=1)
                forecast_days = (first_day_following_month - first_day_next_month).days
                forecast_period_name = "Month"

            # Run prediction when button is clicked
            if st.button(t["run_prediction"]):
                if forecast_days > 0:
                    with st.spinner(t["running_prediction"]):
                        fastest_item, period_forecast, forecast_results = predict_fastest_selling_item_prophet(data.copy(), forecast_days)
                    st.success(t["prediction_completed"])

                    # Display forecasted sales table
                    st.subheader(t["forecasted_sales"].format(period=f"Next {forecast_period_name}"))
                    forecast_df = pd.DataFrame(list(period_forecast.items()), columns=[t["item_column"], t["predicted_sales_column"]])
                    st.table(forecast_df.sort_values(by=t["predicted_sales_column"], ascending=False))
                    st.markdown(f"### **{t['fastest_selling'].format(period=f'Next {forecast_period_name}')}** **{fastest_item}**")

                    # Display forecast visualizations for each item
                    st.subheader(t["forecast_visualizations"].format(period=t["full_timeline"]))
                    st.markdown(t["confidence_interval_explanation"])
                    for item, forecast_dict in forecast_results.items():
                        st.markdown(f"**{t['forecast_for_item']} {item}**")
                        forecast_data = forecast_dict["full"]
                        fig = px.line(
                            forecast_data,
                            x='date',
                            y='predicted_sales',
                            title=t["sales_forecast_title"].format(item=item, period=t["full_timeline"]),
                            labels={'date': t["date_label"], 'predicted_sales': t["sales_label"]}
                        )
                        history = data[data['item'] == item].sort_values('date')
                        # Add historical sales data to the plot
                        fig.add_scatter(x=history['date'], y=history['sales'], mode='lines',
                                        name=t["historical_sales"], line=dict(color='lightgray'))
                        # Add confidence interval bounds
                        fig.add_trace(go.Scatter(
                            x=forecast_data['date'], y=forecast_data['upper_bound'],
                            fill='tonexty', mode='lines',
                            line=dict(color='rgba(0,176,246,0)'),
                            name=t["confidence_interval"]
                        ))
                        fig.add_trace(go.Scatter(
                            x=forecast_data['date'], y=forecast_data['lower_bound'],
                            fill='tonexty', mode='lines',
                            line=dict(color='rgba(0,176,246,0)'),
                            showlegend=False
                        ))
                        fig.update_layout(yaxis_title=t["sales_label"], xaxis_title=t["date_label"])
                        st.plotly_chart(fig)

                    # Option to save forecast results locally
                    if st.button(t["save_results"]):
                        output_folder = "results"
                        os.makedirs(output_folder, exist_ok=True)
                        st.info(t["save_folder_warning"])
                        forecast_df_save = pd.DataFrame(list(period_forecast.items()), columns=["Item", f"{t['predicted_sales_column']}_Next_{forecast_period_name}"])
                        csv_filename = os.path.join(output_folder, f"forecast_table_next_{forecast_period_name.lower()}.csv")
                        forecast_df_save.to_csv(csv_filename, index=False)
                        # Save forecast plots for each item
                        for item, forecast_dict in forecast_results.items():
                            forecast_data = forecast_dict["full"]
                            fig = px.line(
                                forecast_data,
                                x='date',
                                y='predicted_sales',
                                title=t["sales_forecast_title"].format(item=item, period=t["full_timeline"]),
                                labels={'date': t["date_label"], 'predicted_sales': t["sales_label"]}
                            )
                            history = data[data['item'] == item].sort_values('date')
                            fig.add_scatter(x=history['date'], y=history['sales'], mode='lines',
                                            name=t["historical_sales"], line=dict(color='lightgray'))
                            fig.add_trace(go.Scatter(
                                x=forecast_data['date'], y=forecast_data['upper_bound'],
                                fill='tonexty', mode='lines',
                                line=dict(color='rgba(0,176,246,0)'),
                                name=t["confidence_interval"]
                            ))
                            fig.add_trace(go.Scatter(
                                x=forecast_data['date'], y=forecast_data['lower_bound'],
                                fill='tonexty', mode='lines',
                                line=dict(color='rgba(0,176,246,0)'),
                                showlegend=False
                            ))
                            fig.update_layout(yaxis_title=t["sales_label"], xaxis_title=t["date_label"])
                            buf = io.BytesIO()
                            fig.write_image(buf, format="png")
                            buf.seek(0)
                            plot_filename = os.path.join(output_folder, f"forecast_plot_{item.replace(' ', '_')}_full_timeline.png")
                            with open(plot_filename, "wb") as f:
                                f.write(buf.getvalue())
                        st.success(f"Results saved to folder: {output_folder}")
                else:
                    st.warning("Please select a forecast period.")

        except Exception as e:
            st.error(f"{t['error_occurred']} {e}")

if __name__ == '__main__':
    main()