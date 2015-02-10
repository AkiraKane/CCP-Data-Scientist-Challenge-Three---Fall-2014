
# coding: utf-8

# #Problem 1: SmartFly
# You have been contacted by a new online travel service called SmartFly. SmartFly’s business is providing its customers with timely travel information and notifications about flights, hotels, destination weather, traffic getting to the airport, and anything else that can help make the travel experience smoother. Their product team has come up with the idea of using the flight data that they have been collecting to predict whether customers’ flights will be delayed so that they can respond proactively. They’ve now contacted you to help them test out the viability of the idea.
# 
# SmartFly has only been operating for a short while, so their data set only goes back to late last year. As a test of the viability of the idea, they’d like you to use the data they have thus far collected to predict all flight delays for early next year, ** January 1st through January 31st **, with the accuracy to be verified against the actual flights, once they happen. The goal is to proactively offer vouchers to users booked on flights that are very likely to be delayed, that can be used to purchase services at the connecting airport. Because there is a cost associated with offering the vouchers, SmartFly has asked that you give the list of delayed flights sorted in order from most likely to be delayed to least likely to be delayed. For the purposes of this challenge, a flight is considered delayed if and only if its actual departure time is after its scheduled departure time, i.e. a positive departure delay.
# 
# SmartFly Data
# The SmartFly historic data set and flight plans for January can be downloaded here. The total data size is 685MB with 7.3 million historic records and 0.5 million scheduled flights. Each record is comma-delimited and has the following fields:
# 
# - Unique flight ID
# - Year
# - Month (1–12)
# - Day of month (1–31)
# - Day of week (1–7)
# - Scheduled departure time (HHMM)
# - Scheduled arrival time (HHMM)
# - Airline
# - Flight number
# - Tail number
# - Plane model
# - Seat configuration
# - Departure delay (minutes)
# - Origin airport
# - Destination airport
# - Distance travelled (miles)
# - Taxi time in (minutes)
# - Taxi time out (minutes)
# - Whether the flight was cancelled
# - Cancellation code
# 
# The flight plan data has the same fields as the historic data, but
#  some fields in the flight plans data do not contain values.
# 
# SmartFly Deliverables
# ---------------------
# The predictions for delay probabilities for all scheduled flights must
#  be placed into a file called problem1.csv, with a single unique flight 
#  ID per line. The IDs must be in order of most likely to be delayed 
#  to least likely to be delayed, i.e. the first line of the file should
#  contain the ID of the flight that is most likely to be delayed, 
#  and the last line should contain the ID of the flight that is least
#  likely to be delayed. A flight that is canceled is not considered delayed 
#  for the purposes of this challenge.
#  
#  --------------------------------------------------------------------------------
#  

# Set up the environment
# ----------------------

import os
os.chdir('yourdatafolderhere')

import pandas as pd
import numpy as np
from sklearn import naive_bayes, ensemble, svm, linear_model,metrics, cross_validation, grid_search, decomposition, preprocessing
import pylab as pl
import  scipy.stats as stats
import pickle
import itertools
from operator import itemgetter
#get_ipython().magic(u'matplotlib inline')

#define a random seed
SEED=42


# Define Functions
# ----------------


def cvloop(X, y, model,n_iter=5,test_size=0.1):
    kf = cross_validation.ShuffleSplit(n=y.shape[0],n_iter=n_iter,
                                       test_size=test_size, random_state=SEED)   
    k=1
    METRIC = []
    #X = X.values
    for train_index, test_index in kf:
        print "Fold %i of %i" %(k,n_iter)
        model.fit(X[train_index], y[train_index])
        pred = model.predict_proba(X[test_index])[:,1]
        metric = metrics.roc_auc_score(y[test_index],pred)  
        print "    ROC: %.4f" %(metric)
        METRIC.append(metric)
        plot_roc(y[test_index], pred)
        k += 1
    METRIC = np.array(METRIC)
    print "----------------------"
    print "FINAL ROC"
    print "Avg: %.4f, std: %.4f" %(METRIC.mean(),METRIC.std())
    
    return METRIC
    
def plot_roc(y,preds):
    fpr, tpr, thresholds = metrics.roc_curve(y, preds)    
    roc_auc = metrics.auc(fpr, tpr)
    
    # Plot ROC curve    
    pl.clf()    
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)    
    pl.plot([0, 1], [0, 1], 'k--')    
    pl.xlim([0.0, 1.0])    
    pl.ylim([0.0, 1.0])    
    pl.xlabel('False Positive Rate')    
    pl.ylabel('True Positive Rate')    
    pl.title('ROC Curve')    
    pl.legend(loc="lower right")    
    pl.show()


# Import Data
# -----------


headers = [

'Unique_flight_ID',
'Year',
'Month',
'Day',
'DOW',
'Scheduled_departure_time',
'Scheduled_arrival_time',
'Airline',
'Flight_number',
'Tail_number',
'Plane_model',
'Seat_configuration',
'Departure_delay',
'Origin_airport',
'Destination_airport',
'Distance_travelled',
'Taxi_time_in',
'Taxi_time_out',
'Cancelled',
'Cancellation_code'
]                                                                                 


#import the data
#train = pd.read_csv('smartfly_historic.csv',names=headers,low_memory=False,dtype={'Scheduled_departure_time': object, 
#'Scheduled_arrival_time': object})


#add a binary target for classification
train['target'] = 0
train.target[train.Departure_delay>0] = 1


# Data Exploration
# ----------------

train.head()

train.columns.values

train.groupby(['Year','Month']).Year.count().plot(kind='bar')
pl.title('Number of Flights by Month')
pl.xlabel('Months')

train.groupby(['Year','Month']).target.mean().plot(kind='bar')
pl.title('% delayed flights by Month')
pl.xlabel('Months')


train.groupby(['Plane_model']).target.mean().plot(kind='bar')

#type of cancellation codes
train.Cancellation_code.value_counts()

train.shape


# Get all the data in shape for next step
# ---------------------------------------


test = pd.read_csv('smartfly_scheduled.csv',names=headers,dtype={'Scheduled_departure_time': object, 
                                                                'Scheduled_arrival_time': object,
                                                                'Unique_flight_ID': object})


test_id = test.Unique_flight_ID.values

test.shape

#drop cancelled flights
train = train[train.Cancelled==0]
train = train[train.Scheduled_arrival_time != '2096']
rows_train = train.shape[0]

#append train and target
df = pd.concat((train,test),axis=0)

df.shape

train.shape


# 
# Feature Engineering
# --------------------


df['Airline_fn'] = df['Airline']+"_"+df['Flight_number'].map(str)


train[train.Scheduled_departure_time=='26']


#work with time
k=0
dic = {}
for i in range (0,25):
    for j in ['0'+str(t) if t<10 else str(t) for t in range(60)]:
        #print int(str(i)+str(j))
        dic[str(int(str(i)+str(j)))]=k
        k += 1
 

df['Departure_time'] =   df.Scheduled_departure_time.map(dic)
df['Arrival_time'] =   df.Scheduled_arrival_time.map(dic)

df['Departure_hour'] = df.Departure_time.map(lambda x: int(x) / 60)
df['Arrival_hour'] = df.Arrival_time.map(lambda x: int(x) / 60)


#save to disk
df.to_csv('df.csv',index=False)


# Transform data for modelling
# ----------------------------


rows_train = 7270237

df = pd.read_csv('df.csv')

#target and weights
y = df.iloc[:rows_train].target.values
w = df.iloc[:rows_train].Departure_delay.values

#drop columns not used in modelling
df.drop(['Cancelled','Cancellation_code','Departure_delay',
         'Scheduled_arrival_time','Scheduled_departure_time',
         'Unique_flight_ID','target','Taxi_time_out','Taxi_time_in'],inplace=True,axis=1)


df.Tail_number.fillna('-999',inplace=True)


df.head(1)


#encode categorical to number
categorical = [
'Airline',
'Flight_number',
'Tail_number',
'Plane_model',
'Seat_configuration',
'Origin_airport',
'Destination_airport',
'Airline_fn'
]

le = preprocessing.LabelEncoder()

for cat in categorical:
    print cat
    df[cat] = le.fit_transform(df[cat])


X_train = df.iloc[:rows_train]
X_test = df.iloc[rows_train:]


# Random Forest Model
# ------


rf_cv = ensemble.RandomForestClassifier(n_estimators=200,min_samples_leaf=10,n_jobs=-1,random_state=SEED,verbose=1)

#assess performance with cross validation
cvloop(X=X_train.values, y=y, model=rf_cv,n_iter=1,test_size=0.2)


feat_imp = rf_cv.feature_importances_
pd.DataFrame(np.column_stack((X_train.columns.values,feat_imp)),columns=['feat','imp']).set_index(['feat']).sort(['imp']).plot(kind='bar')

rf = ensemble.RandomForestClassifier(n_estimators=200,min_samples_leaf=10,n_jobs=-1,random_state=SEED,verbose=1)

#now fit to the whole data
rf.fit(X_train.values, y)

#predict test
preds = rf.predict_proba(X_test.values)

#assemble
ids = pd.DataFrame(test_id,columns=['ID'])
predictions = pd.DataFrame(preds[:,1],columns=['prediction'])
solution = pd.concat((ids,predictions),axis=1)

solution.sort(columns=['prediction'],ascending=[False],inplace=True)


solution.ID.to_csv('problem1.csv',index=False,header=False)

