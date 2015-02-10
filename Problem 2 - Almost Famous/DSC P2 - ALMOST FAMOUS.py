
# coding: utf-8

# ALMOST FAMOUS
# ==============

# Congratulations! You have just published your first book on data science, advanced analytics, and predictive modeling. You’ve decided to use your skills as a data scientist to create and optimize a website promoting your book, and you have started several ad campaigns on a popular search engine in order to drive traffic to your site.
# 
# Almost Famous Data set
# There are three actions that visitors can take once they arrive at the site: they can order a copy of your book (which earns you \$4), they can sign up for your data science email newsletter (which earns you \$0.40-per-signup), or they can click on one or more ads you have placed on your site for data science training materials (\$0.10 per ad-click). You are logging each event that visitors do on your website, including the keyword and the ID of the ad campaign that brought your visitor to the website and any subsequent actions (adclick, signup, or order) that they performed during their visit. These events are written in JSON format and are available here.
# 
# Along with your ad campaigns, you are running a pair of simultaneous A/B tests on the site. The first pair (experiment one vs. experiment two) tests two different color schemes for the site: a blue one and a green one. The second pair (experiment three vs. experiment four) tests two different promotional blurbs for your book: one from Josh Wills, and one from Sean Owen. Visitors to the site are diverted into the experiments independently using a per-experiment hashing function on a cookie that is placed on the visitor’s browser, so you can assume that visitors will always be in the same experiment no matter how many times they come back to the site. There are no preexisting expectations that either experiment in either pair is more likely to be effective than the other.
# 
# **Almost Famous Deliverables**
# Using your skills in data munging and statistical analysis, answer the following questions about the performance of your site
# using the log data as your source of truth.
# 
# 1. Your advertisers have notified you that there is a bot network which has been plaguing your site. You can find a representative sample of logs from bot activity in the spam.log file. Calculate the number of distinct visitors (not distinct visits) who are bots in the logs. Exclude all events from those bot users from your answers to all of the following questions.
#  
# 2. What is the overall clickthrough rate of the ads (ad clicks per visit)?
# 
# 3. Which combination of query string and campaign had the highest mean value of orders per visitor? For each combination of query string and campaign, what is the standard deviation of the number of orders? (You should calculate n x m  standard deviations, where n is the number of query strings, and m is the number of campaigns.)
#  
# 4. Compute the overall newsletter signup rate (defined as the number of users who signed up to the newsletter divided by the total number of users) for each of the experiments. Use a G-test to compare the performance of experiment one vs. experiment two and experiment three vs. experiment four overall. How many full days of data, starting from the first day, are required to determine that the newsletter signup rate for experiment one is better than experiment two at the 99% confidence level? For example, if you can claim that experiment one is better with 99% confidence using only the first day’s data (9/15/14) and half the second day’s data (9/16/14), then 2 full days are required.
#  
# 5. Given the revenue-per-action values above for buying a copy of the book, signing up for the newsletter, and clicking on an ad, calculate the mean revenue earned per visit for each experiment. Using a z-test, determine how many full days of data, starting from the first full day, are needed to confirm that experiment four earns more revenue than experiment three at the 99% confidence level. For example, if you can claim that experiment four is better with 99% confidence using only half the first day’s data (9/15/14), then one full day is required.

# Set up the environment
# ---------------------


import os
os.chdir('yourdatafolderhere')
import pandas as pd
import numpy as np
import json
import pylab as pl
get_ipython().magic(u'matplotlib inline')


# Import Data
# -----------


import all logs
data = []
k=0
with open('web.log') as f:
    for line in f:
        print k
        data.append(json.loads(line))
        k += 1
        
logs=pd.DataFrame(data)

import spam
data = []
k=0
with open('spam.log') as f:
    for line in f:
        print k
        data.append(json.loads(line))
        k += 1
        
spam=pd.DataFrame(data)

#save to disk
logs.to_csv('logs_df.csv',index=False)
spam.to_csv('spam_df.csv',index=False)

#load from disk
logs = pd.read_csv('logs_df.csv')
spam = pd.read_csv('spam_df.csv')


# PART 1
# ------
# 
# Your advertisers have notified you that there is a bot network which has been plaguing your site. You can find a representative sample of logs from bot activity in the spam.log file. Calculate the number of distinct visitors (not distinct visits) who are bots in the logs. Exclude all events from those bot users from your answers to all of the following questions.


#add a couple of pieces of data

logs['tstamp']=pd.to_datetime(logs.tstamp)
logs.sort(['uid','visit_id','tstamp'],inplace=True)
logs.reset_index(drop=True,inplace=True)
logs['t_last'] = [np.nan]+[(logs.tstamp[i]-logs.tstamp[i-1]).seconds 
                           if logs.visit_id[i]==logs.visit_id[i-1] 
                           else np.nan for i in range(1,logs.shape[0])]




spam['tstamp']=pd.to_datetime(spam.tstamp)
spam.sort(['uid','visit_id','tstamp'],inplace=True)
spam.reset_index(drop=True,inplace=True)
spam['t_last'] = [np.nan]+[(spam.tstamp[i]-spam.tstamp[i-1]).seconds
                           if spam.visit_id[i]==spam.visit_id[i-1] 
                           else np.nan for i in range(1,spam.shape[0])]


#compare distribution of adclicks in logs and spam
logs[logs.action=='adclick'].t_last.value_counts(sort=False).plot()


spam[spam.action=='adclick'].t_last.value_counts(sort=False).plot()


# -----> spam max time between action is 10
# --------------------------------------------
# -----> logs time between action appears inflated before 10
# ---------------------------------------

#lets look at adclick data for legit people (people who have signedup or ordered)
legit = logs[logs.action.isin(['signup','order'])].uid.unique()
legit_activity = pd.merge(left=logs,right=pd.DataFrame(legit,columns=['uid']),
                     on=['uid'])


legit_activity[legit_activity.action=='adclick'].t_last.value_counts(sort=False).plot()



logs[logs.action=='adclick'].t_last.value_counts(sort=False).plot()


# -------->>>> legit people never click on an add before 10 seconds
# --------------------
# ------------->>>>> it appears as though 10 is an artificial threshold between spam and legit
# -----------------------


#remove all spam form logs
bot_ids = logs[(logs.action == 'adclick') & (logs.t_last <= 10)].uid.unique()
good_logs = logs[logs.uid.isin(bot_ids)==False]
print len(bot_ids)


#save good logs to disk
good_logs.to_csv('good_logs.csv',index=False)


# ###Problem 1 Solution: 32116
# 

# ###IMPORT DATA FROM HERE< BOTS ALREADY REMOVED
# 
# 


logs = pd.read_csv('good_logs.csv',date_parser=['tstamp'])
logs['tstamp']=pd.to_datetime(logs.tstamp)

# Problem 2
# -----------
# 
# What is the overall clickthrough rate of the ads (ad clicks per visit)?


logs[logs.action == 'adclick'].groupby(['visit_id']).visit_id.count().order(ascending=False).head()


# **---------> it looks like there is at most one click per visit**


count_clicks_visits = len(logs[logs.action == 'adclick'].visit_id.unique())
count_clicks = len(logs[logs.action == 'adclick'].visit_id)
count_visits = len(logs.visit_id.unique())

ctr = count_clicks/(count_visits+0.0)

print "Visits with at least one click", count_clicks_visits
print "Total number of clicks", count_clicks
print "Count of visits", count_visits
print "CTR", ctr


# ###Problem 2 Solution: 0.010022915283
# 

# Problem 3.1
# ---------
# Which combination of query string and campaign had the highest mean value of orders per visitor?


logs.campaign.fillna(method='ffill',inplace=True)

logs['query'].fillna(method='ffill',inplace=True)

logs.head()


#denominator
orders = logs[logs.action == 'order'].groupby(['query','campaign']).uid.count().reset_index()


visitors = logs.groupby(['query','campaign']).uid.nunique().reset_index()


#inner join logs and order
visitor_orders = pd.merge(left=visitors,right=orders, on=['query','campaign'])

visitor_orders['mean_ord_visitor'] = visitor_orders['uid_y']/visitor_orders['uid_x']

visitor_orders.head(20)

visitor_orders.sort(['mean_ord_visitor'],ascending=[False],inplace=True)


visitor_orders


# ### Problem 3 Solution: campaign 14, query "building predictive models"

# Problem 3.2
# ---------
# For each combination of query string and campaign, what is the standard deviation of the number of orders? (You should calculate n x m standard deviations, where n is the number of query strings, and m is the number of campaigns.)

std_1 = logs.groupby(['query','campaign','uid']).action.count().reset_index()



std_1['action']=0


std_2 = logs[logs.action=='order'].groupby(['query','campaign','uid']).action.count().reset_index()


std_3 = pd.merge(left=std_1,right=std_2, how='left', on=['query','campaign','uid'])


std_3.action_y.fillna(0,inplace=True)


std_3.action_y.max()

std_3.groupby(['query','campaign']).action_y.std()



# Problem 4
# ---------
# Compute the overall newsletter signup rate (defined as the number of users who signed up to the newsletter divided by the total number of users) for each of the experiments. Use a G-test to compare the performance of experiment one vs. experiment two and experiment three vs. experiment four overall. How many full days of data, starting from the first day, are required to determine that the newsletter signup rate for experiment one is better than experiment two at the 99% confidence level? For example, if you can claim that experiment one is better with 99% confidence using only the first day’s data (9/15/14) and half the second day’s data (9/16/14), then 2 full days are required.


(logs[logs.action == 'signup'].uid.nunique()+0.0)/logs.uid.nunique()

def experiment(exp_n=1):
    logs['exp_%i'%exp_n] = [1 if exp_n in map(int,k.strip('[]').split(",")) else 0 for k in logs.experiments]

experiment(exp_n=1)
experiment(exp_n=2)
experiment(exp_n=3)
experiment(exp_n=4)

logs.head()


def signup_rate(df=logs,exper=1):
    tot = len(df[df['exp_%i'%exper]==1].uid.unique())
    signup = len(df[(df['exp_%i'%exper]==1) & (df.action=='signup')].uid.unique())
    print "Experiment %i: %.4f"%(k, signup/(tot+0.0)  )  
    return signup, tot, signup/(tot+0.0)
    
signups = [signup_rate(df=logs,exper=k) for k in range(1,5)]



def create_contingency(v):
    m12 = np.zeros((2,2))
    m34 = np.zeros((2,2))
    
    m12[0,0] = v[0][0]
    m12[0,1] = v[0][1] - v[0][0]
    m12[1,0] = v[1][0]
    m12[1,1] = v[1][1] - v[1][0]
    
    m34[0,0] = v[2][0]
    m34[0,1] = v[2][1] - v[2][0]
    m34[1,0] = v[3][0]
    m34[1,1] = v[3][1] - v[3][0]
    
    return m12, m34

conting_table = create_contingency(signups)



conting_table

from scipy.stats import chi2_contingency
g, p, dof, expctd = chi2_contingency(conting_table[0], lambda_="log-likelihood")
g2, p2, dof2, expctd2 = chi2_contingency(conting_table[1], lambda_="log-likelihood")
print "Confidence for exp1 vs exp2",p,g
print "Confidence for exp3 vs exp4",p2,g2


# **NOW FIND OUT THE EARLIEST TIME THE EXPERIMENT COULD BE STOPPED**


logs['tstamp']=pd.to_datetime(logs.tstamp)
logs['day'] = [k.day for k in logs.tstamp]


logssort = logs.sort(['tstamp','visit_id'])


logssort.head()


logssort_unique = logssort.groupby(['action','uid']).first()

logssort_unique=logssort_unique.reset_index()

logssort_unique.head()


logssort_unique = logssort_unique.sort(['tstamp','uid'])


logssort_unique.head()


logssort2 = pd.concat((logssort_unique,pd.DataFrame(np.zeros((logssort_unique.shape[0],8)),
                                            columns=['l1','l2','l3','l4','o1','o2','o3','o4'])),axis=1)


logssort2 = logssort2.sort(['tstamp','uid'])

logssort2.head()


logssort2['l1'] = np.where((logssort2.action=='landed')&(logssort2.exp_1==1),1,0)
logssort2['l2'] = np.where((logssort2.action=='landed')&(logssort2.exp_2==1),1,0)
logssort2['l3'] = np.where((logssort2.action=='landed')&(logssort2.exp_3==1),1,0)
logssort2['l4'] = np.where((logssort2.action=='landed')&(logssort2.exp_4==1),1,0)

logssort2['o1'] = np.where((logssort2.action=='signup')&(logssort2.exp_1==1),1,0)
logssort2['o2'] = np.where((logssort2.action=='signup')&(logssort2.exp_2==1),1,0)
logssort2['o3'] = np.where((logssort2.action=='signup')&(logssort2.exp_3==1),1,0)
logssort2['o4'] = np.where((logssort2.action=='signup')&(logssort2.exp_4==1),1,0)


logssort2['csl1'] = logssort2['l1'].cumsum()
logssort2['csl2'] = logssort2['l2'].cumsum()
logssort2['csl3'] = logssort2['l3'].cumsum()
logssort2['csl4'] = logssort2['l4'].cumsum()

logssort2['cso1'] = logssort2['o1'].cumsum()
logssort2['cso2'] = logssort2['o2'].cumsum()
logssort2['cso3'] = logssort2['o3'].cumsum()
logssort2['cso4'] = logssort2['o4'].cumsum()


logssort2.tail()


def gstat(x):
    cm = [[x.csl1-x.cso1,x.cso1],[x.csl2-x.cso2,x.cso2]]
    try:
        g, p, dof, expctd = chi2_contingency(cm, lambda_="log-likelihood")
    except:
        p = 0
    return p

def gstat3(x):
    cm = [[x.csl3-x.cso3,x.cso3],[x.csl4-x.cso4,x.cso4]]
    try:
        g, p, dof, expctd = chi2_contingency(cm, lambda_="log-likelihood")
    except:
        p = 0
    return p
    

logssort2['g_test_pvalue'] = logssort2.apply(gstat,axis=1)

logssort2['g_test_pvalue34'] = logssort2.apply(gstat3,axis=1)

logssort2.tail()

logssort2[(logssort2.g_test_pvalue !=0)&(1-logssort2.g_test_pvalue >.99)].head(1)

logssort2[(logssort2.g_test_pvalue34 !=0)&(1-logssort2.g_test_pvalue34 >.99)].head(1)

for d in range(15,31):
    print "Day", d
    #subset based on day
    sub = logs[logs['day']<=d]
    
    #calculate signups
    signups = [signup_rate(df=sub,exper=k) for k in range(1,5)]
    
    #calculate contingecy tables
    conting_table = create_contingency(signups)
    
    #run G-test
    g, p, dof, expctd = chi2_contingency(conting_table[0], lambda_="log-likelihood")
    print "p-value for experiment 1 & 2:", 1-p    
    g, p, dof, expctd = chi2_contingency(conting_table[1], lambda_="log-likelihood")
    print "p-value for experiment 3 & 4:", 1-p 
    print "---------------------------------------------"


# Problem 5
# ---------
# Given the revenue-per-action values above for buying a copy of the book, signing up for the newsletter, and clicking on an ad, calculate the mean revenue earned per visit for each experiment. Using a z-test, determine how many full days of data, starting from the first full day, are needed to confirm that experiment four earns more revenue than experiment three at the 99% confidence level. For example, if you can claim that experiment four is better with 99% confidence using only half the first day’s data (9/15/14), then one full day is required.

l

logs.head()


def experiment(exp_n=1):
    logs['exp_%i'%exp_n] = [1 if exp_n in map(int,k.strip('[]').split(",")) else 0 for k in logs.experiments]

experiment(exp_n=1)
experiment(exp_n=2)
experiment(exp_n=3)
experiment(exp_n=4)


logs = pd.concat((logs,pd.DataFrame(np.zeros((logs.shape[0],8)),
                                            columns=['l1','l2','l3','l4','o1','o2','o3','o4'])),axis=1)



"""
order your book (which earns you $4),
sign up newsletter (which earns you $0.40-per-signup)
click on one or more ads ($0.10 per ad-click)
"""
logs['l1'] = np.where((logs.action=='landed')&(logs.exp_1==1),1,0)
logs['l2'] = np.where((logs.action=='landed')&(logs.exp_2==1),1,0)
logs['l3'] = np.where((logs.action=='landed')&(logs.exp_3==1),1,0)
logs['l4'] = np.where((logs.action=='landed')&(logs.exp_4==1),1,0)

logs['o1'] = np.where((logs.action=='signup')&(logs.exp_1==1),0.4, 
                      np.where((logs.action=='order')&(logs.exp_1==1),4,
                               np.where((logs.action=='adclick')&(logs.exp_1==1),0.1, 0)))
logs['o2'] = np.where((logs.action=='signup')&(logs.exp_2==1),0.4, 
                      np.where((logs.action=='order')&(logs.exp_2==1),4,
                               np.where((logs.action=='adclick')&(logs.exp_2==1),0.1, 0)))
logs['o3'] = np.where((logs.action=='signup')&(logs.exp_3==1),0.4, 
                      np.where((logs.action=='order')&(logs.exp_3==1),4,
                               np.where((logs.action=='adclick')&(logs.exp_3==1),0.1, 0)))
logs['o4'] = np.where((logs.action=='signup')&(logs.exp_4==1),0.4, 
                      np.where((logs.action=='order')&(logs.exp_4==1),4,
                               np.where((logs.action=='adclick')&(logs.exp_4==1),0.1, 0)))


logs[logs.action=='adclick'].head()



logs.sort(['tstamp'],inplace=True)


logs['csl1'] = logs['l1'].cumsum()
logs['csl2'] = logs['l2'].cumsum()
logs['csl3'] = logs['l3'].cumsum()
logs['csl4'] = logs['l4'].cumsum()

logs['cso1'] = logs['o1'].cumsum()
logs['cso2'] = logs['o2'].cumsum()
logs['cso3'] = logs['o3'].cumsum()
logs['cso4'] = logs['o4'].cumsum()


logs['avginc1'] = logs['cso1']/logs['csl1']
logs['avginc2'] = logs['cso2']/logs['csl2']
logs['avginc3'] = logs['cso3']/logs['csl3']
logs['avginc4'] = logs['cso4']/logs['csl4']


for i in range(1,5):
    print "Experiment", i
    print len(logs[logs['exp_%i'%i]==1].visit_id.value_counts())
    print logs['cso%i'%i].tail(1)/logs['csl%i'%i].tail(1)
    print ""


from statsmodels.stats.weightstats import ztest

a =[]
for i in range(0,logs.shape[0],100):
    try:
        a.append([logs.tstamp.iloc[i],logs.avginc3.iloc[i],logs.avginc4.iloc[i]]+
                 list(ztest(logs.o3.iloc[:i],logs.o4.iloc[:i])))#,alternative='smaller'
    except:
        1==1

    if i % 10000==0:
        print i,


from statsmodels.stats.weightstats import ztest

a2 =[]
for i in range(0,logs.shape[0],1000):
    try:
        a2.append([logs.tstamp.iloc[i],logs.avginc3.iloc[i],logs.avginc4.iloc[i]]+
                 list(ztest(logs.o3.iloc[:i],logs.o4.iloc[:i])))#,alternative='smaller'
    except:
        1==1

    if i % 10000==0:
        print i,


zt = pd.DataFrame(a2,columns=['time','e3','e4','z','p'])


zt['day'] = [k.day for k in zt.time]


pl.title('% Z test p-value')
pl.plot(zt.p[zt.day>=26],'b',np.ones(len(zt.p[zt.day>=26]))*0.01,'r--')


zt[zt.p<0.01].day.value_counts()


zt[zt.p<0.01].head()


pl.plot(zt.p.ix[11247:],'b',np.ones(len(zt.p.ix[11247:]))*0.01,'r--')

logs.day.value_counts(sort=False)



