import csv
import random
import smtplib, ssl, email, imaplib
import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy


print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('Scipy: {}'.format(scipy.__version__))

# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the csv file using pandas
data = pd.read_csv('creditcard.csv')

# Print the shape of the data
data = data.sample(frac=0.1, random_state = 1)
print(data.shape)
print(data.describe())

# Start exploring the dataset
print(data.columns)

# V1 - V28 are the results of a PCA Dimensionality reduction to protect user identities and sensitive features

# Plot histograms of each parameter 
data.hist(figsize = (20, 20))
plt.show()

# Determine number of fraud cases in dataset

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

# Get all the columns from the dataFrame
columns = data.columns.tolist()

# Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["Class"]]

# Store the variable we'll be predicting on
target = "Class"

X = data[columns]
Y = data[target]

# Print shapes
print(X.shape)
print(Y.shape)

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# define random states
state = 1

# define outlier detection tools to be compared
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        contamination=outlier_fraction)}

# Fit the model
plt.figure(figsize=(9, 7))
n_outliers = len(Fraud)


for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    
    # Reshape the prediction values to 0 for valid, 1 for fraud. 
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    # Run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))



def dataloader(path, mode='n'):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # print(csv_reader)
        n = random.randint(1, 284000)

        #print(n)
        for i, row in enumerate(csv_reader):
            if i == n and mode=='r':
                data = row
                print(data)
                #print(data[30])
                break
            elif (i == 624 and mode=='f'):
                data = row
                print(data)
                break
            elif (i == 732 and mode == 'n'):
                data = row
                print(data)
                break


        return data

def transaction(data):
    print("ID-"+str(data[0])+"has initiated transaction amount of "+str(data[29])+" USD")

def Mlmodel(input):
    network_output=30
    #print(input[network_output])
    if(input[network_output]=='1'):
        print("Fraud is detected")
        fraud=True
    else:
        print("No Fraud is detected")
        fraud= False
    return fraud

def sms(data):
    id=data[0]
    amount=data[29]
    
    final_message_no="""Subject: Credit Card Security Alert
Transaction Blocked."""
    
    final_message_yes="""Subject: Credit Card Security Alert
You may proceed with your transaction."""
    
    final_message_unknown="""Subject: Credit Card Security Alert
Couldn't Process Reply. Reply with either Yes or No."""

    sender="chiefwarlock1707@gmail.com"
    password="dementors"
    receiver="aritrakhan0@gmail.com"
    context =ssl.create_default_context()
    port=465
    
    mail=imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(sender, password)
    
    mail.select("inbox")
    
    import time,sys
    start=time.time()
    (retcode, messages) = mail.search(None, '(UNSEEN)')
    while b'' in messages:
        mail.select("inbox")
        (retcode, messages) = mail.search(None, '(UNSEEN)')
        end=time.time()
        if end-start>120.0:
            print("Transaction cancelled due to no reply from user")
            sys.exit()
    
    print("Reply received from user")
    
    result, data=mail.uid("search", None, "ALL")
    inbox_item_list=data[0].split()
    most_recent=inbox_item_list[-1]
    
    result2, email_data=mail.uid("fetch", most_recent, "(RFC822)")
    raw_email=email_data[0][1].decode("utf-8")
    
    email_message=email.message_from_string(raw_email)
    
    '''To=email_message["To"]
    From=email_message["From"]
    Subject=email_message["Subject"]'''
    
    text_content=email_message.get_payload()
    text_content2=text_content[0].get_payload()
    
    if text_content2[0:2].find("No")>=0:
        print("Starting to send reply to No..")
        with smtplib.SMTP_SSL("smtp.gmail.com",port,context=context) as server:
            server.login(sender,password)
            server.sendmail(sender,receiver,final_message_no)
        print("Sent. Yay!")
    elif text_content2[0:3].find("Yes")>=0:
        print("Starting to send reply to Yes..")
        with smtplib.SMTP_SSL("smtp.gmail.com",port,context=context) as server:
            server.login(sender,password)
            server.sendmail(sender,receiver,final_message_yes)
        print("Sent. Yay!")
    else:
        print("Starting to send reply to unrecognised..")
        with smtplib.SMTP_SSL("smtp.gmail.com",port,context=context) as server:
            server.login(sender,password)
            server.sendmail(sender,receiver,final_message_unknown)
        print("Sent. Yay!")

def transaction_gateway(transaction_data):
    print(transaction_data)
    print("data loading complete")
    transaction(transaction_data)
    print("ML model start")
    fraud_factor= Mlmodel(transaction_data)
    print(fraud_factor)
    if fraud_factor:
        security_message="""Subject: Credit Card Security Alert
We have detected a possible fraudulent transaction on your credit card XXXX-XXXX-XXXX-XXXX.
Transaction amount: $____ at xyz.com
If you haven't initiated the transaction, reply with: No
and if you have initiated the transaction, then to proceed further, reply with: Yes
        """
        port=465
        
        sender="chiefwarlock1707@gmail.com"
        password="dementors"
        #receiver=sender
        receiver="aritrakhan0@gmail.com"
        
        context =ssl.create_default_context()
        
        print("Starting to send security message..")
        with smtplib.SMTP_SSL("smtp.gmail.com",port,context=context) as server:
            server.login(sender,password)
            server.sendmail(sender,receiver,security_message)
        print("Security message sent. Yay!")
        sms(transaction_data)
        print("okay")
    else:
        print("Transaction successful")



if __name__=="__main__":
    
    path = r"/home/aritra/creditcard.csv"
    
    #random test case mostly non-fraud as the data is unbalanced
    print("Transaction data loading started")
    transaction_data=dataloader(path, 'r')
    transaction_gateway(transaction_data)

    #fraud test case
    print("Transaction data loading started")
    transaction_data=dataloader(path, 'f')
    transaction_gateway(transaction_data)

    #Non-fraud test case
    print("Transaction data loading started")
    transaction_data=dataloader(path, 'n')
    transaction_gateway(transaction_data)