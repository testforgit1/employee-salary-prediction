# Employee Salary Prediction

## Project Overview
This project develops a machine learning model to predict whether an individual's annual income exceeds $50,000 based on demographic and employment-related features from the UCI Adult Income dataset. The target variable is binary (`<=50K` or `>50K`), and the model leverages features such as age, workclass, education level, occupation, and hours worked per week. The project involves data preprocessing, feature engineering, model training, and evaluation to achieve accurate income predictions, with applications in workforce analytics and financial planning.

## Dataset
The dataset used is the UCI Adult Income dataset, available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult). It contains 48,842 records with 15 features, including:
- **Numerical Features**: `age`, `fnlwgt`, `educational-num`, `capital-gain`, `capital-loss`, `hours-per-week`
- **Categorical Features**: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `gender`, `native-country`
- **Target**: `income` (binary: `<=50K` or `>50K`)

The dataset is stored in `adult 3.csv` and loaded using pandas for analysis.

## Requirements
To run the project, ensure the following are installed:
- **Python**: Version 3.12.5 (as per notebook metadata)
- **Libraries**:
  - `pandas`: For data manipulation and preprocessing
  - `numpy`: For numerical computations
  - `scikit-learn`: For feature encoding, data splitting, and model training
  - `matplotlib` and `seaborn`: For data visualization
  - `jupyter`: For running the notebook
- **Hardware**: Standard laptop/desktop with at least 8GB RAM and a multi-core processor

Install dependencies using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/employee-salary-prediction.git
   cd employee-salary-prediction
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the dataset (`adult 3.csv`) is placed in the appropriate directory (e.g., `E:\Documents\MSS\Machine Learning\ML-Datasets\adult 3.csv`) or update the file path in the notebook.

## Usage
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `Employee Salary prediction-edunet.ipynb` in the Jupyter interface.
3. Run the cells sequentially to:
   - Load and explore the dataset
   - Perform data cleaning (e.g., replacing `?` with `NotListed`/`Others`, removing low-frequency categories)
   - Conduct feature engineering (e.g., one-hot encoding, scaling)
   - Train and evaluate machine learning models
4. Review outputs, including cleaned data previews and model performance metrics.

## Methodology
The project follows a structured machine learning pipeline:

1. **Data Loading and EDA**:
   - Load the dataset using `pandas.read_csv`.
   - Check dataset shape (`48,842 rows, 15 columns`) and missing values (`df.isna().sum()`).
   - Analyze feature distributions using `value_counts()` for categorical columns (e.g., `workclass`, `occupation`).

2. **Data Preprocessing**:
   - Replace `?` in `workclass` and `occupation` with `NotListed` and `Others`, respectively.
   - Remove low-frequency categories in `workclass` (e.g., `Without-pay`, `Never-worked`) and `education` (e.g., `Preschool`, `1st-4th`, `5th-6th`).
   - Drop the `education` column due to redundancy with `educational-num`.
   - One-hot encode categorical features using `pd.get_dummies`.
   - Scale numerical features using `StandardScaler`.

3. **Model Training**:
   - Split data into training and testing sets using `train_test_split`.
   - Train multiple classification models (e.g., Logistic Regression, Random Forest) using scikit-learn.
   - Evaluate models based on accuracy, as visualized in the model comparison plot.

4. **Model Deployment**:
   - Save the best-performing model using `pickle` for future predictions.
   - The model can be integrated into a web application or API for real-time use.

## Results
- **Data Cleaning**: Successfully handled missing values and removed noisy categories, resulting in a clean dataset.
- **Feature Engineering**: Transformed categorical features into one-hot encoded columns and scaled numerical features.
- **Model Performance**: Achieved high accuracy with the best model (specific results in the notebook’s model comparison plot).
- **Outputs**: Key outputs include cleaned data previews (e.g., `df.head()`) and category distributions (e.g., `df.workclass.value_counts()`).

## Future Work
- Apply advanced feature selection to reduce dimensionality.
- Address class imbalance using techniques like SMOTE or weighted loss functions.
- Experiment with ensemble models (e.g., XGBoost, LightGBM) or neural networks.
- Deploy the model as a web-based API using Flask or FastAPI.
- Incorporate additional datasets to improve generalizability.

## References
- Dua, D., & Graff, C. (2017). UCI Machine Learning Repository: Adult Data Set. [https://archive.ics.uci.edu/ml/datasets/adult](https://archive.ics.uci.edu/ml/datasets/adult)
- McKinney, W. (2010). Data Structures for Statistical Computing in Python. *pandas Documentation*. [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830. [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. [https://doi.org/10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)

## Contributing
Contributions are welcome! Please submit a pull request or open an issue on the [GitHub repository](https://github.com/username/employee-salary-prediction) for suggestions or improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.