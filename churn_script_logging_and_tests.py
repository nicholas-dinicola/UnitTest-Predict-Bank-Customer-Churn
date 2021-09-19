"""
Unit testing library to test the churn_library.py file

Author: Nicholas Di Nicola
Date: 19/09/2021
"""

import os
import logging
import churn_library as cl

# Create a logs directory where storing the results of the tests
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename='logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    """
        test data import - this example is completed for you to assist with the other test functions
        """
    try:
        df = import_data("data/BankChurners.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df


def test_eda(perform_eda, df):
    """

    Parameters
    ----------
    perform_eda: function
    df: dataframe
    -------

    """
    # compute the churn var (target)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # run eda function and store the  plots
    perform_eda(df)

    try:
        assert 5 == len(os.listdir("images/eda"))
        logging.info("Testing EDA analysis: SUCCESS")
    except AssertionError as err:
        logging.warning("Testing EDA analysis: images not correctly saved.")
        raise err


def test_encoder_helper(encoder_helper, df):
    '''
        test encoder helper
        '''
    # select the cat vars to encode
    cat_vars = ['Gender', 'Education_Level', 'Marital_Status',
                'Income_Category', 'Card_Category'
                ]

    # apply the encoder helper function
    df = encoder_helper(df, cat_vars, 'Churn')

    try:
        assert set(cat_vars).intersection(set(df.columns))
        logging.info("Testing categorical encoder: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing categorical encoder: encoder did not work correctly")
        return err

    return df


def test_perform_feature_engineering(perform_feature_engineering, df):
    """
        Parameters
    ----------
    perform_feature_engineering: function for feature engineering
    df: dataframe
    -------
    Return
    ----------
    X_train, X_test, y_train, y_test
        """
    # Run the feature engineering function
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')
    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing Feature Engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing Feature Engineering: The expected four objects were not computed correctly")
        raise err

    return X_train, X_test, y_train, y_test


def test_train_models(X_train, X_test, y_train, y_test, train_models):
    '''
        test train_models
        '''

    # train the models
    train_models(X_train, X_test, y_train, y_test)

    try:
        assert 2 == len(os.listdir("models"))
        logging.info("Testing Training Models: SUCCESS")
    except AssertionError as err:
        logging.warning("Testing Training Models: models could not be trained")
        raise err


if __name__ == "__main__":
    # Test import_data
    DF = test_import(cl.import_data)

    # Test perform_data
    test_eda(cl.perform_eda, DF)

    # Test encoder_helper
    DF = test_encoder_helper(cl.encoder_helper, DF)

    # Test feature_engineering
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cl.perform_feature_engineering, DF)

    # Test train_models
    test_train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, cl.train_models)
