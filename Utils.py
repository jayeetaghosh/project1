# Auto insurance model - Jayeeta Ghosh
from time import time
from itertools import combinations
from sklearn import preprocessing
import matplotlib.pyplot as plt
import scipy as sp, numpy as np, pandas as pd


#import seaborn as sns


# Return concatenated fields in a dataframe
# [1,2,3,4,5,6] => '123456'
def concat(df, columns):
    return np.array([''.join(x) for x in np.array(
        [np.array(df[col].values, dtype=str) for col in columns]).T])
 



def prepare_data(shuffle=True):
    alldataraw = pd.read_csv('Input/training_autoinsurance.csv')#,nrows=100000)
    alldata = alldataraw.set_index('CustomerID')
    print("Initial training data: ",alldata.shape)

    ## Data Exploration
    ## Take a look into the columns and their dtypes
    # print(alldata.columns.values)
    # print(alldata.dtypes)
    print(alldata.info())
    ## now take a look into the statistics for alldata
    # print (alldata.describe())

     # Are there any missing values?
    print(alldata.isnull().sum().to_string())
    # 98 missing CarValue, 128818 missing RiskFactor, 10111 PrevC and 10111 PrevDuration

    # Consider only numeric variables - impute missing values with mean
    # numerics = ['float64', 'int32', 'int64']
    # numData = alldata.select_dtypes(include=numerics)
    # numCols = numData.columns
    # alldata[numCols] = alldata[numCols].fillna(alldata.mean().iloc[0])
    # alldata['RiskFactor'].fillna((alldata['RiskFactor'].mean()), inplace=True)
    # alldata['PrevC'].fillna((alldata['PrevC'].mean()), inplace=True)
    # alldata['PrevDuration'].fillna((alldata['PrevDuration'].mean()), inplace=True)


    # Consider string/object type columns
    # strCols = alldata.columns[alldata.dtypes == object]
    # print(strCols)
    # alldata[strCols] = alldata[strCols].fillna('.')
    # alldata.fillna(0, inplace=True)
    # alldata['CarValue'].fillna("Nothing", inplace=True)
    # print(alldata.isnull().sum().to_string())

    # Impute numerical missing values by mean, and categorical missing values by a different string
    # https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.dtype.kind.html
    # biufc means bool, signed integer, unsigned integer, float, complex float

    alldata = alldata.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x.fillna('ActuallyMissing'))

    # Check if missing values are gond
    print("Check missing values after imputations")
    print(alldata.isnull().sum().to_string())


    ##################################################
    # Take a closer look at Location
    ##################################################
    loc = np.asarray(alldata.Location)
    #sns.set_palette("deep", desat=.6)
    #sns.set_context(rc={"figure.figsize": (12, 7)})
    
    max_data = np.r_[loc].max()
    min_data = np.r_[loc].min()
    bins = np.linspace(min_data, max_data, max_data/500 + 1)
    plt.hist(loc, bins=20, color="#6495ED", alpha=.5);
    #plt.hist(loc,bins=20)
    plt.title("Histogram of Location")
    # plt.show()

    # Look at cost vs location but nothing interesting
    # cost = np.asarray(alldata.Cost)
    # plt.plot(loc,cost)
    # plt.show()


    # Feature Engineering
    # types of features - continuous and categorical
    # So far location is simple categorical ordered - not good - do not use location as a variable: jayeeta do something
    con = ['GroupSize','CarAge','AgeOldest','AgeYoungest','PrevDuration','Cost'] # continuous variables
    #cat = ['ShoppingPt','Homeowner','CarValue','RiskFactor','Married','PrevC','State', 'Location'] # catagorical variables
    
    cat_int = ['ShoppingPt','Homeowner','Married','PrevC', 'Location', 'RiskFactor'] # catagorical variables
    cat_txt = ['CarValue','State'] # catagorical variables

    conf = ['A','B','C','D','E','F','G'] # options presented to the user

    # final options (A...G) chosen by the user
    # corresponds to row with RecordType = 1 for the same user
    conf_f = [col+'_f' for col in conf] 

    # filter all rows where a purchase was made
    final_purchase = alldata[alldata.RecordType == 1]

    loc = np.asarray(final_purchase.Location)
    cost = np.asarray(final_purchase.Cost)
    groupsz = np.asarray(final_purchase.GroupSize)
    agediff =  np.asarray(final_purchase.AgeOldest-final_purchase.AgeYoungest)
    carage = np.asarray(final_purchase.CarAge)
    a = np.asarray(final_purchase.A)
    b = np.asarray(final_purchase.B)
    c = np.asarray(final_purchase.C)
    d = np.asarray(final_purchase.D)
    e = np.asarray(final_purchase.E)
    f = np.asarray(final_purchase.F)
    g = np.asarray(final_purchase.G)

    #fig = plt.figure(figsize=(50,80))
    #ax1 = fig.add_subplot(4,2,1)

    fig, axes = plt.subplots(nrows=4, ncols=2)
    fig.tight_layout()

    plt.subplot(421); plt.plot(a,cost,'bs'); plt.title("A")
    plt.subplot(422); plt.plot(b,cost,'rs'); plt.title("B")
    plt.subplot(423); plt.plot(c,cost,'gs'); plt.title("C")
    plt.subplot(424); plt.plot(d,cost,'bs'); plt.title("D")
    plt.subplot(425); plt.plot(e,cost,'rs'); plt.title("E")
    plt.subplot(426); plt.plot(f,cost,'gs'); plt.title("F")
    plt.subplot(427); plt.plot(g,cost,'bs'); plt.title("G")

    # plt.show()



    # Set up the plots
    #f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    #c1, c2 = sns.color_palette("husl", 3)[:2]

    plt.plot(loc,cost,'ro'); plt.title("Cost by Location")
    # plt.show()


    plt.plot(cost,groupsz,'bs',cost,agediff,'g^',cost,carage,'ro')
    plt.title("Variation with Cost")
    plt.xlabel('Cost')
    plt.ylabel('group size / age diff / car age')

    # plt.show()
    plt.plot(g,groupsz,'r^')
    plt.xlabel('option G')
    plt.ylabel('group size')
    # plt.show()

    # for each customer id appends columns (A_f...G_f) from final_purchase

    data = alldata.join(final_purchase[conf], rsuffix='_f')
    print("Added 7 extra columns with final purchase plan ",data.shape)
    #print data.shape

    # filter rows where no purchase was made: for now I need all columns
    # create a new column by string joining all the options, e.g., "1234567", on which a purchase was made
    data['plan_purchased'] = concat(data,conf_f)

    # create a new column by string joining all the options, e.g., "0164985", on which a purchase was NOT made
    data['plan_offered'] = concat(data,conf)

    print("Added 2 more columns by combining options ",data.shape)


    # Fix missing values for PrevC and PrevDuration with zeros, refer original problem statement 
    # data['PrevC'].fillna(0, inplace=1)
    # data['PrevDuration'].fillna(0, inplace=1)
    # data['PrevC'].fillna(0, inplace=True)
    # data['PrevDuration'].fillna(0, inplace=True)

    # for three columns CarValue, RiskFactor, State that have categorial values, map them to numbers
    # missing = 99 - can use later to identify missing and populate using customer id from original data
    ###################################
    from sklearn.preprocessing import LabelEncoder

    class MultiColumnLabelEncoder:
        def __init__(self, columns=None):
            self.columns = columns  # array of column names to encode

        def fit(self, X, y=None):
            return self  # not relevant here

        def transform(self, X):
            '''
            Transforms columns of X specified in self.columns using
            LabelEncoder(). If no columns specified, transforms all
            columns in X.
            '''
            output = X.copy()
            if self.columns is not None:
                for col in self.columns:
                    output[col] = LabelEncoder().fit_transform(output[col])
            else:
                for colname, col in output.iteritems():
                    output[colname] = LabelEncoder().fit_transform(col)
            return output

        def fit_transform(self, X, y=None):
            return self.fit(X, y).transform(X)

    ##################################

    data = MultiColumnLabelEncoder(columns=['CarValue','State']).fit_transform(data)
    print(data.info())

    # cost per car_age; cost per person; cost per state
    data['car_age_Cost'] = 1.0 * data.Cost / (data.CarAge + 1)
    data['group_Cost'] = 1.0 * data.Cost / data.GroupSize
    data['state_Cost'] = data.State.map(data.groupby('State')['Cost'].mean())

    # new list to store derived columns
    extra = []
    extra.extend(['car_age_Cost','group_Cost','state_Cost'])

    # difference between age oldest and youngest
    data['ageDiff'] = data.AgeOldest - data.AgeYoungest
    extra.extend(['ageDiff'])

    #print data.caCost,data.ppCost,data.stCost
    for col in ['A','B','C','D','E','F','G']:
        newColName = 'AvgCost_' + col
        data[newColName] = data[col].map(data.groupby(col)['Cost'].mean())
        extra.append(newColName)

    #print data.caCost,data.ppCost,data.stCost
    prev = []
    for col in ['A','B','C','D','E','F','G']:
        newColName = 'LastOptionOffered_' + col
        data[newColName] = data[col].shift(1)
        data.ix[data.ShoppingPt == 1, newColName] = data.ix[data.ShoppingPt == 1, col]
        prev.append(newColName)


    ## previous cost
    data['prev_cost'] = data.Cost.shift(1)
    extra.append('prev_cost')
    data.ix[data.ShoppingPt == 1,'prev_cost'] = data.ix[data.ShoppingPt==1,'Cost']


    # cost lower from previous option might have affect on final choice
    # more positive is better
    data['IncrCost_prev'] = data['prev_cost'] - data['Cost']
    prev.append('IncrCost_prev')

    
    # SHUFFLE THE DATASET, keeping the same customers transaction in order
    # if shuffle:
    #     print("Shuffling dataset...",)
    #     np.random.seed(9); ids = np.unique(data.index.values)
    #     rands = pd.Series(np.random.random_sample(len(ids)),index=ids)
    #     data['rand'] = data.reset_index()['CustomerID'].map(rands).values
    #     data.sort(['rand','ShoppingPt'],inplace=True); print("DONE!")
    #print data
    print(data.isnull().sum().to_string())

    # If nothing final purchase then assign conf_f=0
    data[conf_f] = data[conf_f].fillna(0)
    data[prev] = data[prev].fillna(0)
    data[extra] = data[extra].fillna(0)
    # Are there any missing values?
    print(data.isnull().sum().to_string())

    return data, con, cat_int, cat_txt, extra, prev, conf, conf_f#, encoders

