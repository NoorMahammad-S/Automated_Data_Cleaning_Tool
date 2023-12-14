from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging
import os
from tempfile import NamedTemporaryFile
from scipy import stats

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='cleaning_log.txt', level=logging.INFO, filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')


def clean_data_chunk(chunk):
    # Handling missing values by filling with mean
    cleaned_chunk = chunk.fillna(chunk.mean())

    # Handling outliers by clipping values within a certain range
    cleaned_chunk = cleaned_chunk.clip(lower=cleaned_chunk.quantile(0.05), upper=cleaned_chunk.quantile(0.95))

    # Converting categorical variables to numerical using one-hot encoding
    cleaned_chunk = pd.get_dummies(cleaned_chunk, columns=['categorical_column'])

    # Drop columns with too many missing values
    threshold = 0.8  # Set your own threshold
    cleaned_chunk = cleaned_chunk.dropna(axis=1, thresh=int(threshold * len(cleaned_chunk)))

    # Convert date columns to datetime objects
    date_columns = ['date_column1', 'date_column2']
    cleaned_chunk[date_columns] = cleaned_chunk[date_columns].apply(pd.to_datetime, errors='coerce')

    # Remove duplicates
    cleaned_chunk = cleaned_chunk.drop_duplicates()

    # Normalize numeric columns
    numeric_columns = cleaned_chunk.select_dtypes(include=['number']).columns
    cleaned_chunk[numeric_columns] = (cleaned_chunk[numeric_columns] - cleaned_chunk[numeric_columns].min()) \
                                     / (cleaned_chunk[numeric_columns].max() - cleaned_chunk[numeric_columns].min())

    # Add more cleaning logic based on your specific requirements
    return cleaned_chunk


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        # Save the file to a temporary location
        temp_file = NamedTemporaryFile(delete=False)
        file.save(temp_file.name)

        # Process the file in chunks
        chunk_size = 10000
        data_chunks = pd.read_csv(temp_file.name, chunksize=chunk_size)

        # Log the start of the cleaning process
        logging.info("Data cleaning process started...")

        cleaned_data = pd.DataFrame()
        cleaning_logs = []

        for chunk in data_chunks:
            try:
                # Invoke your data cleaning module on each chunk
                cleaned_chunk = clean_data_chunk(chunk)

                # Concatenate cleaned_chunk to the cleaned_data DataFrame
                cleaned_data = pd.concat([cleaned_data, cleaned_chunk], ignore_index=True)

                # Log information for each chunk if needed
                logging.info(f"Data cleaning completed for chunk with shape: {cleaned_chunk.shape}")

            except Exception as e:
                # Handle exceptions for each chunk
                logging.error(f"Error during data cleaning for a chunk: {str(e)}")

        # Log the end of the cleaning process
        logging.info("Data cleaning process completed.")

        # Check if cleaned_data is not empty and contains numeric columns
        if not cleaned_data.empty:
            numeric_columns = cleaned_data.select_dtypes(include=['number']).columns
            if not numeric_columns.empty:
                # Use describe only on numeric columns
                summary_stats = cleaned_data[numeric_columns].describe().to_html()
            else:
                summary_stats = "No numeric columns available for summary statistics."
        else:
            summary_stats = "No data available for summary statistics."

        # Remove the temporary file
        os.remove(temp_file.name)

        # Render the results template with cleaned data and logs
        return render_template('results.html', summary_stats=summary_stats, cleaned_data=cleaned_data,
                               cleaning_logs=cleaning_logs)
    except Exception as e:
        logging.error(f"Error during data cleaning: {str(e)}")
        return render_template('error.html', error_message=str(e))


# Route for data cleaning
@app.route('/handle-missing-values', methods=['POST'])
def handle_missing_values():
    temp_file = None
    try:
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        # Save the file to a temporary location
        temp_file = NamedTemporaryFile(delete=False)
        file.save(temp_file.name)

        # Load data using Pandas
        df = pd.read_csv(temp_file.name)

        # Identify missing values
        missing_values = df.isnull().sum()

        # Provide options for imputation (example: mean, median)
        imputation_methods = ['mean', 'median']

        # Render a template to display missing values and options for imputation
        return render_template('missing_values.html', missing_values=missing_values,
                               imputation_methods=imputation_methods)
    except Exception as e:
        return render_template('error.html', error_message=str(e))
    finally:
        # Remove the temporary file if it exists
        if temp_file:
            os.remove(temp_file.name)


# Route for handling imputation
@app.route('/impute-missing-values', methods=['POST'])
def impute_missing_values():
    temp_file = None  # Initialize temp_file to None
    try:
        imputation_method = request.form['imputation_method']

        # Retrieve the file from the request
        file = request.files['file']

        # Check if a file was selected
        if file.filename == '':
            return "No selected file"

        # Save the file to a temporary location
        temp_file = NamedTemporaryFile(delete=False)
        file.save(temp_file.name)

        # Load data using Pandas
        df = pd.read_csv(temp_file.name)

        # Perform imputation based on the selected method (example: mean, median)
        if imputation_method == 'mean':
            df_imputed = df.fillna(df.mean())
        elif imputation_method == 'median':
            df_imputed = df.fillna(df.median())
        else:
            return "Invalid imputation method"

        # Log the imputed DataFrame
        logging.info("Imputed DataFrame:\n%s", df_imputed)

        # Further processing or render a template with the imputed data
        return "Imputation completed successfully"
    except Exception as e:
        return render_template('error.html', error_message=str(e))
    finally:
        # Remove the temporary file if it exists
        if temp_file is not None:
            os.remove(temp_file.name)


# Route for handling outliers
@app.route('/handle-outliers', methods=['POST'])
def handle_outliers():
    temp_file = None
    try:
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        # Save the file to a temporary location
        temp_file = NamedTemporaryFile(delete=False)
        file.save(temp_file.name)

        # Load data using Pandas
        df = pd.read_csv(temp_file.name)

        # Implement Z-score or IQR for outlier detection (example: Z-score)
        z_scores = df.apply(lambda column: (column - column.mean()) / column.std())
        outliers = (z_scores.abs() > 3).any(axis=1)

        # Provide options for handling outliers (example: remove, transform)
        handling_methods = ['remove', 'transform']

        # Render a template to display detected outliers and options for handling
        return render_template('outlier_detection.html', outliers=outliers, handling_methods=handling_methods)
    except Exception as e:
        return render_template('error.html', error_message=str(e))
    finally:
        # Remove the temporary file if it exists
        if temp_file is not None:
            os.remove(temp_file.name)


# Route for handling outliers
@app.route('/handle-detected-outliers', methods=['POST'])
def handle_detected_outliers():
    temp_file = None
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"})

        # Save the file to a temporary location
        temp_file = NamedTemporaryFile(delete=False)
        file.save(temp_file.name)

        # Load data using Pandas
        df = pd.read_csv(temp_file.name)

        # Implement Z-score or IQR for outlier detection (example: Z-score)
        z_scores = df.apply(lambda column: (column - column.mean()) / column.std())
        outliers = (z_scores.abs() > 3).any(axis=1)

        # Retrieve handling method from the form
        handling_method = request.form['handling_method']

        # Perform action based on the selected method (example: remove, transform)
        if handling_method == 'remove':
            df_filtered = df[~outliers]
        elif handling_method == 'transform':
            # Implement transformation logic based on your requirements
            # Replace outliers with a specified value
            df_filtered = df.copy()
            df_filtered.loc[outliers, :] = df_filtered.mean()
        else:
            return jsonify({"error": "Invalid handling method"})

        # Log the filtered DataFrame and handling method
        logging.info("Filtered DataFrame using %s:\n%s", handling_method, df_filtered)

        # Further processing or render a template with the modified data
        return jsonify({"message": "Handling completed successfully"})
    except FileNotFoundError:
        return jsonify({"error": "File not found"})
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        # Remove the temporary file if it exists
        if temp_file is not None:
            os.remove(temp_file.name)


# Route for data normalization
@app.route('/normalize-data', methods=['POST'])
def normalize_data():
    temp_file = None
    try:
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        # Save the file to a temporary location
        temp_file = NamedTemporaryFile(delete=False)
        file.save(temp_file.name)

        # Load data using Pandas
        df = pd.read_csv(temp_file.name)

        # Utilize Scikit-Learn for Min-Max scaling
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(df.select_dtypes(include=['number']))

        # Handle categorical variables (example: one-hot encoding)
        categorical_columns = df.select_dtypes(include=['object']).columns
        df_categorical = pd.get_dummies(df[categorical_columns])
        normalized_df = pd.concat(
            [pd.DataFrame(normalized_data, columns=df.select_dtypes(include=['number']).columns),
             df_categorical], axis=1)

        # Render a template to display normalized data and options for handling categorical variables
        return render_template('normalized_data.html', normalized_data=normalized_df)
    except Exception as e:
        return render_template('error.html', error_message=str(e))
    finally:
        # Remove the temporary file if it exists
        if temp_file is not None:
            os.remove(temp_file.name)


# Route for handling categorical variables
@app.route('/handle_categorical', methods=['POST'])
def handle_categorical_variables(normalized_df=None):
    try:
        handling_method = request.form['handling_method']

        # Perform action based on the selected method (example: one-hot encoding)
        if handling_method == 'one_hot_encoding':
            # Implement one-hot encoding logic
            normalized_df_encoded = pd.get_dummies(normalized_df, drop_first=True)
        else:
            return "Invalid handling method"

        # Log the encoded DataFrame
        logging.info("Encoded DataFrame:\n%s", normalized_df_encoded)

        # Further processing or render a template with the modified data
        return "Handling completed successfully"
    except Exception as e:
        return render_template('error.html', error_message=str(e))


# Route for data transformation
@app.route('/transform-data', methods=['POST'])
def transform_data():
    temp_file = None
    try:
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        # Save the file to a temporary location
        temp_file = NamedTemporaryFile(delete=False)
        file.save(temp_file.name)

        # Load data using Pandas
        df = pd.read_csv(temp_file.name)

        # Implement log transformation
        df_log_transformed = np.log1p(df.select_dtypes(include=['number']))

        # Implement Box-Cox transformation (requires positive values)
        df_boxcox_transformed = df.select_dtypes(include=['number']).apply(
            lambda x: np.log1p(x) if x.min() <= 0 else stats.boxcox(x)[0])

        # Render a template to display transformed data and options for transformation
        return render_template('transformed_data.html', df_log_transformed=df_log_transformed,
                               df_boxcox_transformed=df_boxcox_transformed)
    except Exception as e:
        return render_template('error.html', error_message=str(e))
    finally:
        # Remove the temporary file if it exists
        if temp_file is not None:
            os.remove(temp_file.name)


# Route for applying data transformation
@app.route('/apply_transformation', methods=['POST'])
def apply_data_transformation():
    temp_file = None
    try:
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        # Save the file to a temporary location
        temp_file = NamedTemporaryFile(delete=False)
        file.save(temp_file.name)

        # Load data using Pandas
        df = pd.read_csv(temp_file.name)

        transformation_method = request.form['transformation_method']

        # Perform action based on the selected method (example: log, boxcox)
        if transformation_method == 'log':
            # Implement log transformation logic
            transformed_data = np.log1p(df.select_dtypes(include=['number']))
        elif transformation_method == 'boxcox':
            # Implement Box-Cox transformation logic (requires positive values)
            transformed_data = df.select_dtypes(include=['number']).apply(lambda x: np.log1p(x) if x.min() <= 0 else stats.boxcox(x)[0])
        else:
            return "Invalid transformation method"

        # Log the transformed data
        logging.info("Transformed Data:\n%s", transformed_data)

        # Further processing or render a template with the transformed data
        return "Transformation completed successfully"
    except Exception as e:
        return render_template('error.html', error_message=str(e))
    finally:
        # Remove the temporary file if it exists
        if temp_file is not None:
            os.remove(temp_file.name)


# Route for data quality checks
@app.route('/check-data-quality', methods=['POST'])
def check_data_quality():
    temp_file = None
    try:
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        # Save the file to a temporary location
        temp_file = NamedTemporaryFile(delete=False)
        file.save(temp_file.name)

        # Load data using Pandas
        df = pd.read_csv(temp_file.name)

        # Detect duplicated records
        duplicated_records = df[df.duplicated()]

        # Provide options for consistency checks
        consistency_checks = ['check1', 'check2']

        # Render a template to display duplicated records and options for consistency checks
        return render_template('data_quality_checks.html', duplicated_records=duplicated_records,
                               consistency_checks=consistency_checks)
    except Exception as e:
        return render_template('error.html', error_message=str(e))
    finally:
        # Remove the temporary file if it exists
        if temp_file is not None:
            os.remove(temp_file.name)


# Route for applying consistency check
@app.route('/apply_consistency_check', methods=['POST'])
def apply_consistency_check():
    temp_file = None
    try:
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        # Save the file to a temporary location
        temp_file = NamedTemporaryFile(delete=False)
        file.save(temp_file.name)

        # Load data using Pandas
        df = pd.read_csv(temp_file.name)

        consistency_check = request.form['consistency_check']

        # Perform action based on the selected consistency check
        if consistency_check == 'check1':
            # Implement consistency check 1 logic
            checked_data = df.copy()  # Replace with your logic for check1
            # Add your logic to modify 'checked_data' based on check1
            checked_data['column1'] = checked_data['column1'] * 2  # Example modification
            checked_data['column2'] = checked_data['column2'].apply(lambda x: x + 1)  # Another example modification
        elif consistency_check == 'check2':
            # Implement consistency check 2 logic
            checked_data = df.copy()  # Replace with your logic for check2
            # Add your logic to modify 'checked_data' based on check2
            checked_data['column1'] = checked_data['column1'] + 10  # Example modification
            checked_data['column3'] = checked_data['column3'].apply(lambda x: x * 3)  # Another example modification
        else:
            return "Invalid consistency check"

        # Log the checked data
        logging.info("Checked Data:\n%s", checked_data)

        # Further processing or render a template with the checked data
        return "Consistency check completed successfully"
    except Exception as e:
        return render_template('error.html', error_message=str(e))
    finally:
        # Remove the temporary file if it exists
        if temp_file is not None:
            os.remove(temp_file.name)


# Route for data cleaning
@app.route('/clean-data', methods=['POST'])
def clean_data():
    temp_file = None
    try:
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        # Save the file to a temporary location
        temp_file = NamedTemporaryFile(delete=False)
        file.save(temp_file.name)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(temp_file.name)

        # Log the start of the cleaning process
        logging.info("Data cleaning process started...")

        # Example: Handling missing values
        missing_values = df.isnull().sum()
        logging.info(f"Missing values before imputation:\n{missing_values}")

        # Example: Imputation
        df.fillna(df.mean(), inplace=True)
        logging.info("Missing values imputed using mean.")

        # Continue with other cleaning steps...

        # Log the end of the cleaning process
        logging.info("Data cleaning process completed.")

        # Generate a cleaning report
        generate_cleaning_report(df)

        return "Data cleaning completed successfully"
    except Exception as e:
        logging.error(f"Error during data cleaning: {str(e)}")
        return render_template('error.html', error_message=str(e))
    finally:
        # Close the temporary file before removing it
        if temp_file is not None:
            temp_file.close()
            os.remove(temp_file.name)


def generate_cleaning_report(df):
    # Create a summary report
    report = f"Cleaning Report:\n"
    report += f"Number of rows: {df.shape[0]}\n"
    report += f"Number of columns: {df.shape[1]}\n"
    report += f"Columns: {', '.join(df.columns)}\n"
    report += "\nAdditional Information:\n"
    report += f"Data Types:\n{df.dtypes}\n"
    report += f"Summary Statistics:\n{df.describe()}\n"
    # Add more information to the report as needed

    # Save the report to a file
    with open('cleaning_report.txt', 'w') as report_file:
        report_file.write(report)

    return report


# Example route for API integration
@app.route('/api/clean', methods=['POST'])
def api_clean():
    try:
        # Extract data from the API request
        data = request.get_json()

        # Create a DataFrame from the input data
        df = pd.DataFrame(data)

        # Log the start of the cleaning process
        logging.info("API data cleaning process started...")

        # Example: Handling missing values
        missing_values = df.isnull().sum()
        logging.info(f"Missing values before imputation:\n{missing_values}")

        # Example: Imputation
        df.fillna(df.mean(), inplace=True)
        logging.info("Missing values imputed using mean.")

        # Example: Data normalization
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(df.select_dtypes(include=['number']))
        logging.info("Data normalized using Min-Max scaling.")

        # Continue with other cleaning steps...

        # Log the end of the cleaning process
        logging.info("API data cleaning process completed.")

        # Generate a summary report

        report = generate_cleaning_report(df)

        return jsonify({"cleaned_data": normalized_data.tolist(), "cleaning_report": report})
    except Exception as e:
        logging.error(f"Error during API data cleaning: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Route for documentation
@app.route('/documentation')
def documentation():
    # Render the "documentation.html" template
    return render_template('documentation.html')


if __name__ == '__main__':
    app.run(debug=True)
