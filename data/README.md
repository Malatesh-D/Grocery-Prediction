# Dataset handling and provenance

`grocery_data.csv` in the repository is **synthetic demonstration data**, generated with a fixed seed. It contains no real household observations. `dataset_manifest.json` identifies its type and generator provenance.

To use authorized real data, upload a CSV in the Streamlit **Dataset Explorer**. Map the available numeric columns to the app's supported fields, select the monthly expenditure target (INR), optionally identify a household/group column or a date/time column, and record collection period, origin, units, licensing/usage restrictions, and limitations. The app validates the mapping and trains only after the explicit button is selected. Uploaded row-level records remain in session memory and are not written by the app; the fitted model, feature schema, dataset fingerprint, aggregate metrics, and test predictions are persisted. Remove direct identifiers before upload. A group key is used only for split isolation and excluded from model inputs; a time key is only for chronological splitting.

The official India Household Consumption Expenditure Survey 2023–24 includes detailed food consumption data. Its microdata catalogue currently requires account login to access data files, so this project has not downloaded it. Review portal terms and request any necessary permission before use. Public catalog entry: <https://microdata.gov.in/nada/index.php/catalog/237/get-microdata>.

The source survey may not include shopping trips, average basket size, branded share, online share, or previous-month expenditure. These features must not be fabricated to make a model schema fit. Map only genuine available variables and evaluate the resulting model on an appropriate sample.
