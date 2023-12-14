# Automated Data Cleaning Tool

## Overview

The Automated Data Cleaning Tool is a web application built using Flask that facilitates the automated cleaning of tabular data. The tool is designed to handle common data preprocessing tasks, such as handling missing values, outliers, normalization, and more. It allows users to upload CSV files, apply various data cleaning techniques, and visualize the results.

## Features

- **Data Cleaning**: Automates common data cleaning tasks, including handling missing values, outliers, normalization, and categorical variable encoding.
- **Web Interface**: Provides a user-friendly web interface for uploading data files and viewing the cleaned data.
- **Logging**: Logs the data cleaning process, making it easy to trace errors and understand the applied transformations.
- **API Integration**: Offers an API endpoint for integrating the data cleaning process into external applications.

## Getting Started

### Prerequisites

- Python 3.x
- Flask
- Pandas
- Scikit-Learn
- NumPy
- SciPy

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/NoorMahammad-S/Automated_Data_Cleaning_Tool.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python app.py
   ```

   The web application will be accessible at `http://localhost:5000`.

## Usage

1. Open the web interface in your browser (`http://localhost:5000`).
2. Upload a CSV file containing the data you want to clean.
3. Select the desired data cleaning options.
4. View the summary statistics and cleaned data in the results.

For API integration, send a POST request to `/api/clean` with the input data in JSON format.

## Configuration

- The application's configuration can be adjusted in the `config.py` file.

## Contributing

Contributions are welcome! Please follow the [Contribution Guidelines](CONTRIBUTING.md).

## Acknowledgments

- Special thanks to [Flask](https://flask.palletsprojects.com/) and [Pandas](https://pandas.pydata.org/) for providing the foundation for this tool.

## Contact

For issues or inquiries, please [create an issue](https://github.com/your-username/automated-data-cleaning-tool/issues).
