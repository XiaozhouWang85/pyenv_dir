import pandas as pd
import numpy as np

from src import data_transformations as dt

nan = np.nan

def test_DT_to_time_of_day():
    data = pd.Series([18411805,18403928,18405189,27975266,27975346,34214345])

    transformed_data = dt.DT_to_time_of_day(data)

    pd.testing.assert_series_equal(transformed_data, pd.Series([27,27,27,42,42,51]))

def test_get_cat_values():
    data = pd.Series(["A","A","A","A","B","B","B","C","C","C","D","D","Z","Q","K"])

    assert set(dt.get_cat_values(data,3)) == set(['A', 'B', 'C'])
    assert set(dt.get_cat_values(data,4)) == set(['A', 'B', 'C', 'D'])


def test_transform_pipeline():
    train_data = [
        [2987954, 0, 109559, 179.95, 'W', 7919, 194.0, 150.0, 'mastercard', 202.0, 'debit', 485.0, 87.0, nan, nan, nan, nan, 10.0, 0.0, 63.0, 63.0, 63.0, nan, nan, nan, nan, nan, nan, nan, 0.0, 0.0], 
        [3174179, 0, 4160317, 300.0, 'R', 17188, 321.0, 150.0, 'visa', 226.0, 'debit', 225.0, 87.0, nan, nan, 'aol.com', 'aol.com', 1.0, 1.0, 0.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], 
        [3516697, 0, 13952975, 26.95, 'W', 12544, 321.0, 150.0, 'visa', 226.0, 'debit', 184.0, 87.0, 87.0, nan, 'gmail.com', nan, 117.0, 0.0, 496.0, 496.0, 460.0, 496.0, 'T', 'T', 'F', 'T', 1.0, 1.0, 1.0, 1.0], 
        [3127687, 0, 2899597, 151.0, 'W', 2772, 512.0, 150.0, 'visa', 226.0, 'debit', 191.0, 87.0, nan, nan, 'yahoo.com', nan, 2.0, 0.0, 120.0, 120.0, 120.0, nan, nan, nan, nan, nan, nan, nan, 1.0, 1.0], 
        [3246843, 0, 6223214, 21.0, 'W', 17055, 393.0, 150.0, 'mastercard', 117.0, 'debit', 325.0, 87.0, nan, nan, 'gmail.com', nan, 2.0, 0.0, 67.0, nan, 203.0, nan, nan, nan, nan, nan, nan, nan, 0.0, 1.0], 
        [3180845, 0, 4372670, 107.95, 'W', 11218, 579.0, 150.0, 'visa', 226.0, 'debit', 441.0, 87.0, nan, nan, nan, nan, 1.0, 0.0, 352.0, 352.0, 408.0, nan, nan, nan, nan, nan, nan, nan, 0.0, 0.0], 
        [3497317, 0, 13367483, 34.0, 'W', 7919, 194.0, 150.0, 'mastercard', 166.0, 'debit', 184.0, 87.0, nan, nan, 'aol.com', nan, 1.0, 0.0, 376.0, 376.0, 0.0, 410.0, 'T', 'F', 'F', 'F', 1.0, 1.0, 2.0, 2.0], 
        [3548449, 0, 14863015, 106.0, 'S', 15775, 481.0, 150.0, 'mastercard', 102.0, 'credit', 330.0, 87.0, nan, nan, nan, 'yahoo.com', 5.0, 3.0, 43.0, 43.0, 253.0, nan, nan, nan, nan, nan, nan, nan, 0.0, 0.0], 
        [3426983, 0, 11144705, 141.0, 'W', 6237, nan, 150.0, 'visa', 166.0, 'debit', 272.0, 87.0, nan, nan, nan, nan, 1.0, 0.0, 0.0, nan, 0.0, nan, 'T', 'T', 'F', 'T', nan, nan, 0.0, 0.0], 
        [3224322, 0, 5605561, 248.0, 'W', 1675, 174.0, 150.0, 'visa', 226.0, 'debit', 330.0, 87.0, nan, nan, 'gmail.com', nan, 2.0, 0.0, 543.0, 543.0, 543.0, nan, nan, nan, nan, nan, nan, nan, 0.0, 0.0], 
        [3128007, 0, 2907786, 108.95, 'W', 4153, 549.0, 150.0, 'visa', 226.0, 'debit', 325.0, 87.0, 2564.0, nan, 'gmail.com', nan, 96.0, 0.0, 260.0, 229.0, 260.0, 426.0, 'T', 'T', 'F', 'F', 1.0, 1.0, 0.0, 0.0], 
        [3426780, 0, 11140540, 39.0, 'W', 14089, 512.0, 150.0, 'mastercard', 117.0, 'debit', 264.0, 87.0, nan, nan, 'yahoo.com', nan, 2.0, 0.0, 22.0, 22.0, 22.0, nan, nan, nan, nan, nan, nan, nan, 1.0, 1.0], 
        [3517030, 0, 13962285, 59.0, 'W', 16116, 568.0, 150.0, 'visa', 226.0, 'debit', 330.0, 87.0, nan, nan, 'yahoo.com', nan, 1.0, 0.0, 0.0, nan, 0.0, 0.0, 'T', 'T', 'F', 'T', 1.0, 1.0, 1.0, 1.0], 
        [3551978, 0, 14959754, 24.1, 'C', 9026, 545.0, 185.0, 'visa', 137.0, 'credit', nan, nan, nan, 1.0, 'hotmail.com', 'hotmail.com', 2.0, 2.0, 0.0, nan, 0.0, nan, nan, nan, nan, nan, nan, nan, 0.0, 0.0], 
        [3349859, 0, 8992110, 107.95, 'W', 3484, 372.0, 150.0, 'mastercard', 117.0, 'debit', 264.0, 87.0, 21.0, nan, 'gmail.com', nan, 97.0, 0.0, 281.0, 281.0, 34.0, 307.0, 'T', 'T', 'F', 'T', 1.0, 1.0, 0.0, 0.0]
    ]

    test_data = [
        [3941168, 27996391, 30.003, 'C', 15885, 545.0, 185.0, 'visa', 138.0, 'debit', nan, nan, nan, nan, 'hotmail.com', 'hotmail.com', 1.0, 1.0, 0.0, nan, 0.0, nan, nan, nan, nan, nan, nan, nan, 0.0, 0.0], 
        [3789279, 22716839, 159.95, 'W', 17131, 111.0, 150.0, 'mastercard', 224.0, 'debit', 264.0, 87.0, nan, nan, 'yahoo.com', nan, 2.0, 0.0, 33.0, 33.0, 0.0, 33.0, 'T', 'T', 'F', 'T', 1.0, 1.0, 1.0, 1.0], 
        [4109206, 33016470, 24.1, 'C', 15885, 545.0, 185.0, 'visa', 138.0, 'debit', nan, nan, nan, nan, 'gmail.com', 'gmail.com', 1.0, 1.0, 0.0, 0.0, 0.0, nan, nan, nan, nan, nan, nan, nan, 0.0, 0.0], 
        [3725468, 20521748, 374.95, 'W', 8695, 170.0, 150.0, 'visa', 226.0, 'credit', 315.0, 87.0, nan, nan, 'yahoo.com', nan, 2.0, 0.0, 0.0, nan, 0.0, nan, nan, nan, 'F', 'T', nan, nan, 1.0, 1.0], 
        [3856527, 25126736, 57.95, 'W', 12932, 361.0, 150.0, 'visa', 226.0, 'debit', 204.0, 87.0, nan, nan, 'yahoo.com', nan, 1.0, 0.0, 0.0, nan, 0.0, 0.0, 'T', 'T', 'F', 'T', 1.0, 1.0, 1.0, 1.0], 
        [3852406, 24957318, 58.95, 'W', 4663, 490.0, 150.0, 'visa', 166.0, 'debit', 191.0, 87.0, nan, nan, 'gmail.com', nan, 1.0, 0.0, 0.0, nan, 0.0, nan, nan, nan, nan, nan, nan, nan, 0.0, 0.0], 
        [4156685, 33852139, 472.04, 'W', 2455, 321.0, 150.0, 'visa', 226.0, 'credit', 428.0, 87.0, nan, nan, 'yahoo.com', nan, 1.0, 0.0, 0.0, nan, 0.0, 0.0, 'T', 'T', 'F', 'T', 1.0, 1.0, 1.0, 1.0], 
        [3957841, 28585809, 80.95, 'W', 6550, nan, 150.0, 'visa', 226.0, 'debit', 204.0, 87.0, 2.0, nan, nan, nan, 5.0, 0.0, 308.0, 308.0, 214.0, 660.0, 'T', 'T', 'T', 'T', 1.0, 1.0, 1.0, 1.0], 
        [4159782, 33933039, 422.5, 'W', 15066, 170.0, 150.0, 'mastercard', 102.0, 'credit', 220.0, 87.0, nan, nan, 'gmail.com', nan, 1.0, 0.0, 0.0, nan, 0.0, 0.0, 'T', 'T', 'F', 'T', 1.0, 1.0, 1.0, 1.0], 
        [4038340, 31198520, 226.0, 'W', 9749, 181.0, 150.0, 'visa', 226.0, 'credit', 299.0, 87.0, 7.0, nan, 'sbcglobal.net', nan, 3.0, 0.0, 0.0, nan, 215.0, 237.0, 'T', 'T', 'F', 'T', 1.0, 1.0, 1.0, 1.0], 
        [3761682, 21688009, 39.0, 'W', 12695, 490.0, 150.0, 'visa', 226.0, 'debit', 204.0, 87.0, nan, nan, 'aol.com', nan, 1.0, 0.0, 0.0, nan, 0.0, nan, nan, nan, nan, nan, nan, nan, 0.0, 0.0],
        [4013230, 30384978, 201.0, 'W', 8117, 490.0, 150.0, 'visa', 226.0, 'debit', 181.0, 87.0, 9.0, nan, 'gmail.com', nan, 1.0, 0.0, 611.0, 611.0, 611.0, 289.0, 'T', 'F', nan, nan, 1.0, 1.0, 0.0, 1.0], 
        [3702264, 19709146, 9.283, 'C', 15885, 545.0, 185.0, 'visa', 138.0, 'debit', nan, nan, nan, 199.0, 'gmail.com', 'gmail.com', 2.0, 2.0, 0.0, nan, 0.0, nan, nan, nan, nan, nan, nan, nan, 0.0, 0.0], 
        [3756671, 21531371, 62.95, 'W', 10185, nan, 150.0, 'visa', 226.0, 'debit', 110.0, 87.0, 2.0, nan, 'msn.com', nan, 4.0, 0.0, 168.0, 168.0, 488.0, 518.0, 'T', 'T', 'T', 'T', 1.0, 1.0, 1.0, 1.0], 
        [3935681, 27821250, 107.0, 'W', 16075, 514.0, 150.0, 'mastercard', 102.0, 'credit', 181.0, 87.0, 7.0, nan, 'hotmail.com', nan, 2.0, 0.0, 29.0, 29.0, 748.0, 231.0, 'T', 'F', 'F', 'T', 1.0, 1.0, 0.0, 0.0]
    ]

    train_columns = [
        'TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD', 
        'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 
        'P_emaildomain', 'R_emaildomain', 'C1', 'C10','D1', 'D2','D10', 'D11', 'M1', 'M2', 'M8', 
        'M9', 'V1', 'V2', 'V12', 'V13'
    ]

    test_columns = [
        'TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD', 
        'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 
        'P_emaildomain', 'R_emaildomain', 'C1', 'C10','D1', 'D2','D10', 'D11', 'M1', 'M2', 'M8', 
        'M9', 'V1', 'V2', 'V12', 'V13'
    ]

    cont_vars = ["TransactionAmt"]
    cat_vars = ["ProductCD","addr1","addr2","P_emaildomain","R_emaildomain",'card1', 'card2', 'card3', 'card4', 'card5', 'card6']

    train = pd.DataFrame(train_data,columns=train_columns)
    test = pd.DataFrame(test_data,columns=test_columns)

    x_train, y_train, x_test = dt.transform_pipeline(train,test,cont_vars,cat_vars)

    #Function has not altered original dataframe in place
    pd.testing.assert_frame_equal(train, pd.DataFrame(train_data,columns=train_columns))
    pd.testing.assert_frame_equal(test, pd.DataFrame(test_data,columns=test_columns))

    print(train.head())
    print(x_train)

    assert 1 == 0

