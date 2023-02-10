import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



#------------------------------------------------------------------------------------------------------------

# Rolling average of last 12 months


def rolling(df, val):

    rolling = df['avg_temp'].rolling(12).mean()[-1]

    yhat_df = pd.DataFrame({'avg_temp': rolling}, index= val.index)
    
    return yhat_df
