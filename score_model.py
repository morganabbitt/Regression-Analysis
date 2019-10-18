import sys
import numpy as np
import pandas as pd



def rmsle_score(target, predictions):
    log_diff = np.log(predictions+1) - np.log(target+1)
    return np.sqrt(np.mean(log_diff**2))

if __name__=='__main__':
    infile = sys.argv[1]
    predictions = pd.read_csv(infile)
    predictions.set_index('SalesID')
    test_solution = pd.read_csv('data/do_not_open/test_soln.csv')
    test_solution.set_index('SalesID')
    rmsle = rmsle_score(predictions.SalePrice, test_solution.SalePrice)
    print(rmsle)
