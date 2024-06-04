from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


def preprocess_data(df):
    """
    Written for rent data; will drop null values and 
    split into training and testing sets. Uses price
    as the target column.
    """
    # raw_num_df_rows = len(df)
    # df = df.dropna()
    # remaining_num_df_rows = len(df)
    # percent_na = (
    #     (raw_num_df_rows - remaining_num_df_rows) / raw_num_df_rows * 100
    # )
    # print(f"Dropped {round(percent_na,2)}% rows")
    X = df.drop(columns='actual_productivity')
    y = df['actual_productivity'].values.reshape(-1, 1)
    return train_test_split(X, y)

def r2_adj(x, y, model):
    """
    Calculates adjusted r-squared values given an X variable, 
    predicted y values, and the model used for the predictions.
    """
    r2 = model.score(x,y)
    n_cols = x.shape[1]
    return 1 - (1 - r2) * (len(y) - 1) / (len(y) - n_cols - 1)

def check_metrics(X_test, y_test, model):
    # Use the pipeline to make predictions
    y_pred = model.predict(X_test)

    # Print out the MSE, r-squared, and adjusted r-squared values
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R-squared: {r2_score(y_test, y_pred)}")
    print(f"Adjusted R-squared: {r2_adj(X_test, y_test, model)}")

    return r2_adj(X_test, y_test, model)

def get_best_pipeline(pipeline, pipeline2, df):
    """
    Accepts two pipelines and rent data.
    Uses two different preprocessing functions to 
    split the data for training the different 
    pipelines, then evaluates which pipeline performs
    best.
    """
    X_train, X_test, y_train, y_test = preprocess_data(df)
    pipeline.fit(X_train, y_train)
    pipeline2.fit(X_train, y_train)
    r2_adj1 = check_metrics(X_test, y_test, pipeline)
    r2_adj2 = check_metrics(X_test, y_test, pipeline2)
    if r2_adj1 > r2_adj2:
        print('Pipeline 1 is better')
        return pipeline
    else:
        print('Pipeline 2 is better')
        return pipeline2
    
def create_pipeline(garment_df):
    steps = [('one-hot', OneHotEncoder(drop='first', handle_unknown='ignore')),
        ('scaler', StandardScaler(with_mean=False))]
    
    pipeline = Pipeline(steps.copy())
    pipeline2 = Pipeline(steps.copy())
    pipeline3 = Pipeline(steps.copy())

    pipeline.steps.append(('model', LinearRegression()))
    pipeline2.steps.append(('model', Lasso()))
    pipeline3.steps.append(('model', Ridge(alpha=0.5)))

    best = get_best_pipeline(pipeline, pipeline2, garment_df)
    very_best = get_best_pipeline(best, pipeline3, garment_df)

    return very_best

# if __name__ == "__main__":
#     # Path: 12-Regression/3/03-Grp_Regression_Mini_Project/garment_worker_productivity.csv
#     garment_df = pd.read_csv('garment_worker_productivity.csv')
#     garment_pipeline = garment_pipeline_generator(garment_df)
#     print(garment_pipeline)
