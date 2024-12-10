# Synthetic Data Generator

**This project provides a user-friendly Streamlit-based web application for generating synthetic datasets. It allows you to create artificial text, tabular data (with configurable distributions, categorical fields, and correlations), and time series data. The tool also supports loading user samples to produce similar synthetic datasets, saving/loading configurations, and offers integration with a GPT-2 model for advanced text generation.**

## **Key Features**

- Text Data Generation:
    - Faker: Generate simple fake sentences, paragraphs, names, and addresses in various locales.
    - GPT-2 Integration: Generate text using a pre-trained GPT-2 model for more complex and realistic outputs. Adjust parameters like prompt, max length, temperature, top-k, and top-p.

- Tabular Data Generation:
    - Create numeric columns with configurable distributions (normal or uniform) and parameters (mean, std, min, max).
    - Create categorical columns using Faker (e.g., names, cities, companies) or custom categories.
    - Set correlations between multiple normal-distributed numeric columns.
    - Large Data Mode: Generate very large datasets in chunks, streaming the results directly to a file to avoid memory issues.
    - Display only a sample of the generated data (for large datasets) and provide a download button for the entire dataset.

- Time Series Generation:
    - Generate synthetic time series with adjustable trend, noise level, and seasonality.
    - Visualize the generated time series with a line chart and download the resulting CSV.

- Imitation from a Sample:
    - Upload a real CSV dataset.
    - Automatically infer column types (numeric, categorical).
    - Suggest distributions and categories based on the uploaded sample.
    - Generate a synthetic dataset with similar statistical properties.

- Configuration Management:
    - Save current generation settings to a JSON configuration file.
    - Load a previously saved configuration and apply it to the interface.

- UI/UX Enhancements:
    - Clear tabs and headers for different data types.
    - Markdown formatting, tooltips, and explanations for parameters.
    - About tab with instructions, tool descriptions, and future plans.


# **Installation**

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/synthetic-data-generator.git
    cd synthetic-data-generator
    ```

2. Start using Docker:

    ```bash
    docker build . -t synt_data
    ```

3. After launching, open the provided URL (e.g., http://localhost:8501) in your browser.

# **Usage**

- Navigate through the tabs:
    - Text Data: Choose between Faker or GPT-2 to generate text. Adjust parameters and generate. Download the results as CSV.
    - Tabular Data: Set the number of rows, locale, and the number of columns. For each column, select numeric or categorical data, specify distributions or categories. Optionally define correlations between numeric columns. Generate and download the CSV.
    - Time Series: Specify the number of points, trend, noise, and seasonality. Generate a line chart and download the series as CSV.
    - Imitation from a Sample: Upload your own CSV to analyze and mimic. Adjust suggested parameters, generate synthetic data, and download it.
    - Configurations: Save current settings to a JSON file or load an existing configuration.
    - About: Overview of the application and future plans.
- For large datasets, enable the "large data mode" in the Tabular Data tab to generate data in chunks and avoid running out of memory. Use the progress bar as an indicator of the generation status.

# **Examples**
- Text Generation (Faker):
Generate 100 sentences in English (en_US) and download the resulting CSV.

- Text Generation (GPT-2):
Provide a prompt "Once upon a time" and generate a short story with temperature=1.0, top-k=50, top-p=0.9.

- Tabular with Correlations:
Create 5 columns of numeric data (all normal) and define a correlation matrix. Generate thousands of rows and download the CSV.

- Imitation from a Sample:
Upload your own dataset, let the tool suggest parameters, and generate a similar synthetic dataset.
