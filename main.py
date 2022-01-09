# D:\Users\kozgen\PROJECT\log_channel.log file contains logs of discord server
# D:\Users\kozgen\PROJECT\bans-filtered.json file contains filtered logs of log_channel.log
# D:\Users\kozgen\PROJECT\welcome.log file contains member join logs of discord server

# Read files and import them to dataframes and visualize them
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime

# print stacktrace, so import
import traceback

# Read bans-filtered.json file and convert it to a dataframe
print("Reading bans-filtered.json file...")

# 2 searial welcome log timestamp difference:
diff = 1300
# Threshold to trigger:
threshold = 15

bf = None
wf = None
bf_f = None
with open('D:\\Users\\kozgen\\PROJECT\\bans-filtered.json', encoding="utf8") as f:
    #Also add welcome.log file
    with open('D:\\Users\\kozgen\\PROJECT\\welcome.log', encoding="utf8") as f2:
        try:
            # File content is: {"1": {"channelId":"772827177804365885","guildId":"534402283149197319","deleted":false,"id":"923517752239620156","createdTimestamp":1640254190264,"type":"GUILD_MEMBER_JOIN","system":true,"content":"","authorId":"739610256614883439","pinned":false,"tts":false,"nonce":null,"embeds":[],"components":[],"attachments":[],"stickers":[],"editedTimestamp":null,"webhookId":null,"groupActivityApplicationId":null,"applicationId":null,"activity":null,"flags":0,"reference":null,"interaction":null,"cleanContent":""},
            #"2": {"channelId":"772827177804365885","guildId":"534402283149197319","deleted":false,"id":"923495624203198485","createdTimestamp":1640248914529,"type":"GUILD_MEMBER_JOIN","system":true,"content":"","authorId":"836724442658897950","pinned":false,"tts":false,"nonce":null,"embeds":[],"components":[],"attachments":[],"stickers":[],"editedTimestamp":null,"webhookId":null,"groupActivityApplicationId":null,"applicationId":null,"activity":null,"flags":0,"reference":null,"interaction":null,"cleanContent":""}}
            # Load the json file into a dataframe
            bf = pd.read_json(f, orient='index')
            wf = pd.read_json(f2, orient='index')

            # Columns are #Index(['channelId', 'guildId', 'deleted', 'id', 'createdTimestamp', 'type','system', 'content', 'authorId', 'pinned', 'tts', 'nonce', 'embeds','components', 'attachments', 'stickers', 'editedTimestamp', 'webhookId','groupActivityApplicationId', 'applicationId', 'activity', 'flags','reference', 'interaction', 'cleanContent'],dtype='object')
            # Filter welcome logs by this algorithm: If there is more than threshold number of logs coming after each other after diff seconds, then add them to the filtered dataframe. createdTimestamp is in milliseconds.
            bf_f = pd.DataFrame()
            # Add a new column createdTimestamp_formatted to the dataframe. It is in format of MM/DD/YYYY.
            bf['createdTimestamp_formatted'] = pd.to_datetime(bf['createdTimestamp'], unit='ms').dt.strftime('%m/%d/%Y')
            # Print createdTimestamp_formatted column
            wf['createdTimestamp_formatted'] = pd.to_datetime(wf['createdTimestamp'], unit='ms').dt.strftime('%m/%d/%Y')
            print(bf['createdTimestamp_formatted'])

        except Exception as e:
            print("Error: " + str(e))
            traceback.print_exc()

# Create a histogram and show createdTimestamp_formatted column with counts
plt.figure(figsize=(30,10))
plt.hist(bf['createdTimestamp_formatted'], bins=len(bf['createdTimestamp_formatted'].unique()))
plt.xticks(rotation=90)
plt.show()

# Create a histogram and show createdTimestamp_formatted 
plt.figure(figsize=(30,10))
plt.hist(wf['createdTimestamp_formatted'], bins=len(wf['createdTimestamp_formatted'].unique()))
plt.xticks(wf['createdTimestamp_formatted'][::400], rotation=90)
plt.show()

# Print banned user count for each day
print(bf['createdTimestamp_formatted'].value_counts())


combined = pd.DataFrame()
# Add empty date, bans, and welcome logs columns to the combined dataframe
combined['date'] = ""
combined['bans'] = 0
combined['welcome'] = 0

# Visualize both dataframes together in a line chart
# For all unique dates in both dataframes, count the number of bans and welcome logs.
for i in wf['createdTimestamp_formatted'].unique():
    # Insert row to combined dataframe. Date, banned count and welcome logs count.
    combined.loc[len(combined)] = [i, bf[bf['createdTimestamp_formatted'] == i]['id'].count(), wf[wf['createdTimestamp_formatted'] == i]['id'].count()]


# Visualize combined dataframe in a stacked bar chart. X axis is date, Y axis is bans and welcome logs.
plt.figure(figsize=(30,10))
plt.bar(combined['date'], combined['bans'], color='red', label='Bans')
plt.bar(combined['date'], combined['welcome'], bottom=combined['bans'], color='green', label='Welcome logs')
plt.xticks(combined['date'][::20], rotation=90)
plt.legend()
plt.show()

# Create a new wf_f dataframe with same columns as wf dataframe.
wf_f = pd.DataFrame( columns=wf.columns )



# Print column names
print(wf_f.columns)
print(wf.columns)



current_date = ""
# Print 13th element of createdTimestamp_formatted column

# For all rows unique MM/DD/YYYY in createdTimestamp_formatted column filter 2 consecutive rows with same date. If there is more than threshold number of logs coming after each other after diff seconds, then add them to the filtered dataframe. createdTimestamp is in milliseconds.
for i in wf['createdTimestamp_formatted'].unique():
    # Create a temporary dataframe to store the consecutive logs that come after each other.
    tmp = pd.DataFrame( columns=wf.columns )
    # For all rows in wf that have same createdTimestamp_formatted as i
    for j in wf[wf['createdTimestamp_formatted'] == i].index:
        # If this is the last row in wf, then break
        if j == len(wf) - 1:
            break
        # If the difference between the current row and the next row is less than diff seconds, then add the next row to the list.
        if abs(wf.iloc[j]['createdTimestamp'] - wf.iloc[j+1]['createdTimestamp']) < diff:
            # Append elements to temporary dataframe
            tmp.loc[len(tmp)] = wf.loc[j]
    # If the length of the temporary dataframe is more than threshold number of logs, then add the temporary dataframe to the filtered dataframe.
    if len(tmp) > threshold:
        # Add the temporary dataframe to the filtered dataframe.
        wf_f = wf_f.append(tmp)
        # Print length of tmp for date i
        print("Date: " + str(i) + " Length: " + str(len(tmp)))


# Create a new dataframe and store filtered frequency column with date information and actual frequency column and ban count for each date.
wf_f_f = pd.DataFrame( columns= ['date', 'frequency', 'filtered_frequency', 'bans'] )

# For all unique dates in wf_f dataframe, count the number of logs.
for i in wf_f['createdTimestamp_formatted'].unique():
    # Insert row to wf_f_f dataframe. Date, frequency, filtered frequency and ban count.
    frequency = wf[wf['createdTimestamp_formatted'] == i]['id'].count()
    filtered_frequency = wf_f[wf_f['createdTimestamp_formatted'] == i]['id'].count()
    bans = bf[bf['createdTimestamp_formatted'] == i]['id'].count()
    wf_f_f.loc[len(wf_f_f)] = [i, frequency, filtered_frequency, bans]



wf_f_f.loc[len(wf_f_f)] = ['11/06/2021', wf[wf['createdTimestamp_formatted'] == '11/06/2021']['id'].count(), wf_f[wf_f['createdTimestamp_formatted'] == '11/06/2021']['id'].count(), bf[bf['createdTimestamp_formatted'] == '11/06/2021']['id'].count()]

try:
# Add 00:00:00 to the end of each date in date column
    wf_f_f['date'] = wf_f_f['date'] + ' 00:00:00'
# Convert date column to datetime format then sort by date
    wf_f_f['date'] = pd.to_datetime(wf_f_f['date'])
    wf_f_f = wf_f_f.sort_values(by='date')
# Convert date column to string format as MM/DD/YYYY
    wf_f_f['date'] = wf_f_f['date'].dt.strftime('%m/%d/%Y')
except Exception as e:
    print("Error: " + str(e))
    traceback.print_exc()



# Visualize as a bar chart. X axis is date, Y axis is filtered frequency, frequency and ban count. Write 1 to red, 2 to green and 3 to blue.
fig, ax = plt.subplots()
fig.set_size_inches(30,10)

x = np.arange(len(wf_f_f['date']))
width = 0.3
ax.bar(x - width, wf_f_f['filtered_frequency'], width, label='Filtered frequency', color='r')
ax.bar(x, wf_f_f['frequency'], width, label='Frequency', color='g')
ax.bar(x + width, wf_f_f['bans'], width, label='Bans', color='b')
ax.set_xticks(x)
ax.set_xticklabels(wf_f_f['date'], rotation=90)
ax.legend()
plt.show()

# A dataframe to store banned user names
banned_users = pd.DataFrame( columns= ['username', 'label'] )

# Get user ids usernames that have banned on 11/02/2021, 11/03/2021 and 11/06/2021.
# username is at "embeds":[{"type":"rich","title":null,"description":"<@373940551818412035> godxdchan#3351","url":null,"color":16729871,"timestamp":1633375294360,"fields":[],"image":null,"video":null,"provider":null}] godxdchan#3351 is username in this case and 373940551818412035 is user id.

try:
    for bans in bf['createdTimestamp_formatted'].unique():
        if bans == '11/02/2021' or bans == '11/03/2021' or bans == '11/06/2021':
            # Convert  createdTimestamp in milliseconds epoch to time and print id and username of users that have banned on 11/02/2021, 11/03/2021 and 11/06/2021.
            desc = bf[bf['createdTimestamp_formatted'] == bans]['embeds'].apply(lambda x: x[0]['description'])
            # Print Date user id and username from decription inside embeds "<@373940551818412035> godxdchan#3351" to 373940551818412035 godxdchan#3351
            # Get description from embeds column
            for i in desc:
                # From "<@373940551818412035> godxdchan#3351" get 373940551818412035
                id = i.split('@')[1].split('>')[0]
                # Get username from description
                name = i.split('>')[1].split('#')[0]
                banned_users.loc[len(banned_users)] = [name, 1]
                # Print user id and username
                print(bans + " " + id + " " + banned_users.loc[len(banned_users)-1]['username'])
except Exception as e:
    print("Error: " + str(e))
    traceback.print_exc()

# Print 5 top and bottom banned users.
print(banned_users.sort_values(by='label', ascending=False).head(5))
print(banned_users.sort_values(by='label', ascending=True).head(5))


try:
    all_users = pd.DataFrame( columns= ['username', 'label'] )

    # Read allusers.txt from D:\Users\kozgen\PROJECT\allusers.txt
    # File contents is like [{"id":"772805089035419678","username":"Yasin başar","discriminator":"3912","createdTimestamp":1604321491250,"avatarURL":null,"defaultAvatarURL":"https://cdn.discordapp.com/embed/avatars/2.png"},{"id":"324880771393519618","username":"Irresistable","discriminator":"0189","createdTimestamp":1497528011893,"avatarURL":"https://cdn.discordapp.com/avatars/324880771393519618/450ce5b6dcda02e993ca7f4f2bcfb0a2.webp","defaultAvatarURL":"https://cdn.discordapp.com/embed/avatars/4.png"}]
    with open('D:\\Users\\kozgen\\PROJECT\\allusers.txt', 'r', encoding="utf8") as f:
        # Read allusers.txt
        data = json.load(f)
        # Convert array to dataframe
        df = pd.DataFrame(data)
        # Check defaultAvatarURL and avatarURL and if they are not null and not equal  print id and username
        
        print("Finished reading allusers.txt")
    # print(all_users)
except Exception as e:
    print("Error: " + str(e))
    traceback.print_exc()

# Print unique usernames from df
print(df['username'].unique())

# Import scikit-learn and use it to create the classifiers.
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from IPython.display import Javascript


# Print size of df 
print(len(df))
# Create a training set.
# We will create a training set by creating a dataframe with user names. Add a label column to the dataframe.
train_users = pd.DataFrame( columns= ['username', 'label'] )
# Create a test set with remaining user names.
test_users = pd.DataFrame( columns= ['username', 'label'] )

regular_ones = pd.DataFrame( columns= ['username', 'label'] )
irregular_ones = pd.DataFrame( columns= ['username', 'label'] )

# Add user names from pd dataframe to all_users dataframe with label 0.
for i in df['username'].unique():
    # Insert userid to all_users dataframe
    regular_ones.loc[len(regular_ones)] = [i, 0]
# Add banned user names from bf dataframe to all_users dataframe with label 1.
for i in banned_users['username'].unique():
    # Insert userid to all_users dataframe
    irregular_ones.loc[len(irregular_ones)] = [i, 1]


try:
    # Print length of all_users dataframe
    print(len(all_users))
    # Print length of irregular_ones dataframe
    #print("Irregular ones: " + str(len(irregular_ones)))
    rat_increase = 0.05
    rat = 0.05
    mispreds_df = pd.DataFrame(columns=['classifier','array', 'training_ratio', 'array_size'] )
    # Save scores to scores.txt
    scores = open('D:\\Users\\kozgen\\PROJECT\\scores.txt', 'w')
    while rat <= 1.0:

        regular_split = int(len(regular_ones) * rat)
        irregular_split = int(len(irregular_ones) * rat)

        train_users = regular_ones.sample(n=regular_split, random_state=1)
        # Add from irregular_ones to train_users from 0 to regular_split
        train_users = train_users.append(irregular_ones.sample(n=irregular_split, random_state=1))

        # Add from regular_ones to test_users from regular_split to len(regular_ones)
        test_users = regular_ones.iloc[regular_split:len(regular_ones)]
        # Add from irregular_ones to test_users from irregular_split to len(irregular_ones)
        test_users = test_users.append(irregular_ones.iloc[irregular_split:len(irregular_ones)])

        
        # Print size of train_users dataframe and test_users dataframe
        print(len(train_users))
        print(len(test_users))

        # Shuffle train and test dataframes
        train_users = train_users.sample(frac=1).reset_index(drop=True)
        test_users = test_users.sample(frac=1).reset_index(drop=True)

        # Train the classifiers.
        sklearn.utils.shuffle(train_users)
        sklearn.utils.shuffle(test_users)

        # Combined dataframe
        combined_df = pd.DataFrame( columns= ['username', 'label'] )
        # Get index of train_users dataframe
        train_users_index = train_users.index
        # Append train_users dataframe to combined_df
        combined_df = combined_df.append(train_users)
        # Get index of test_users dataframe
        test_users_index = test_users.index
        # Append test_users dataframe to combined_df
        combined_df = combined_df.append(test_users)
        # LabelEncoder for combined_df
        le = preprocessing.LabelEncoder()
        # Fit username column of combined_df to le
        le.fit(combined_df['username'])
        # Transform username column of combined_df to le
        combined_df['username'] = le.transform(combined_df['username'])
        # Reconstruct train_users dataframe and test_users dataframe
        train_users = combined_df.iloc[train_users_index]
        test_users = combined_df.iloc[test_users_index]

        # Convert label column to int
        train_users['label'] = train_users['label'].astype(int)
        test_users['label'] = test_users['label'].astype(int)

        # Numpy array for train username column
        train_users_username = train_users['username'].values
        # Numpy array for train label column
        train_users_label = train_users['label'].values
        # Numpy array for test username column
        test_users_username = test_users['username'].values
        # Numpy array for test label column
        test_users_label = test_users['label'].values

        # reshape(-1, 1)
        train_users_username = train_users_username.reshape(-1, 1)
        test_users_username = test_users_username.reshape(-1, 1)


        # Create a random forest classifier
        rfc = RandomForestClassifier(n_estimators=100, random_state=1)
        # Train the classifier
        rfc.fit(train_users_username, train_users_label)
        # Predict the class of test_users
        predictions_rfc = rfc.predict(test_users_username)
        # Print f1 score, precision, recall, accuracy, and confusion matrix
        scores.write("Random Forest Classifier"+ "\n")
        scores.write("F1 score: " + str(f1_score(test_users_label, predictions_rfc))+ "\n")
        scores.write("Precision: " + str(precision_score(test_users_label, predictions_rfc))+ "\n")
        scores.write("Recall: " + str(recall_score(test_users_label, predictions_rfc))+ "\n")
        scores.write("Accuracy: " + str(accuracy_score(test_users_label, predictions_rfc))+ "\n")
        scores.write("Confusion matrix: " + str(confusion_matrix(test_users_label, predictions_rfc))+ "\n")
        scores.write("Classification report: " + str(classification_report(test_users_label, predictions_rfc))+ "\n")
        

        # Create a naive bayes classifier
        nbc = MultinomialNB()
        # Train the classifier
        nbc.fit(train_users_username, train_users_label)
        # Predict the class of test_users
        predictions_nbc = nbc.predict(test_users_username)
        # Print f1 score, precision, recall, accuracy, and confusion matrix
        scores.write("Naive Bayes Classifier"+ "\n")
        scores.write("F1 score: " + str(f1_score(test_users_label, predictions_nbc))+ "\n")
        scores.write("Precision: " + str(precision_score(test_users_label, predictions_nbc))+ "\n")
        scores.write("Recall: " + str(recall_score(test_users_label, predictions_nbc))+ "\n")
        scores.write("Accuracy: " + str(accuracy_score(test_users_label, predictions_nbc))+ "\n")
        scores.write("Confusion matrix: " + str(confusion_matrix(test_users_label, predictions_nbc))+ "\n")
        scores.write("Classification report: " + str(classification_report(test_users_label, predictions_nbc))+ "\n")

        # Create a logistic regression classifier
        lrc = LogisticRegression()
        # Train the classifier
        lrc.fit(train_users_username, train_users_label)
        # Predict the class of test_users
        predictions_lrc = lrc.predict(test_users_username)
        # Print f1 score, precision, recall, accuracy, and confusion matrix
        scores.write("Logistic Regression Classifier"+ "\n")
        scores.write("F1 score: " + str(f1_score(test_users_label, predictions_lrc))+ "\n")
        scores.write("Precision: " + str(precision_score(test_users_label, predictions_lrc))+ "\n")
        scores.write("Recall: " + str(recall_score(test_users_label, predictions_lrc))+ "\n")
        scores.write("Accuracy: " + str(accuracy_score(test_users_label, predictions_lrc))+ "\n")
        scores.write("Confusion matrix: " + str(confusion_matrix(test_users_label, predictions_lrc))+ "\n")
        scores.write("Classification report: " + str(classification_report(test_users_label, predictions_lrc))+ "\n")

        # Create a support vector machine classifier
        svc = SVC()
        # Train the classifier
        svc.fit(train_users_username, train_users_label)
        # Predict the class of test_users
        predictions_svc = svc.predict(test_users_username)
        # Print f1 score, precision, recall, accuracy, and confusion matrix
        scores.write("Support Vector Machine Classifier"+ "\n")
        scores.write("F1 score: " + str(f1_score(test_users_label, predictions_svc))+ "\n")
        scores.write("Precision: " + str(precision_score(test_users_label, predictions_svc))+ "\n")
        scores.write("Recall: " + str(recall_score(test_users_label, predictions_svc))+ "\n")
        scores.write("Accuracy: " + str(accuracy_score(test_users_label, predictions_svc))+ "\n")
        scores.write("Confusion matrix: " + str(confusion_matrix(test_users_label, predictions_svc))+ "\n")
        scores.write("Classification report: " + str(classification_report(test_users_label, predictions_svc))+ "\n")

        # Create a decision tree classifier
        dtc = DecisionTreeClassifier()
        # Train the classifier
        dtc.fit(train_users_username, train_users_label)
        # Predict the class of test_users
        predictions_dtc = dtc.predict(test_users_username)
        # Print f1 score, precision, recall, accuracy, and confusion matrix
        scores.write("Decision Tree Classifier"+ "\n")
        scores.write("F1 score: " + str(f1_score(test_users_label, predictions_dtc))+ "\n")
        scores.write("Precision: " + str(precision_score(test_users_label, predictions_dtc))+ "\n")
        scores.write("Recall: " + str(recall_score(test_users_label, predictions_dtc))+ "\n")
        scores.write("Accuracy: " + str(accuracy_score(test_users_label, predictions_dtc))+ "\n")
        scores.write("Confusion matrix: " + str(confusion_matrix(test_users_label, predictions_dtc))+ "\n")
        scores.write("Classification report: " + str(classification_report(test_users_label, predictions_dtc))+ "\n")

        # Create a k-nearest neighbors classifier
        knn = KNeighborsClassifier()
        # Train the classifier
        knn.fit(train_users_username, train_users_label)
        # Predict the class of test_users
        predictions_knn = knn.predict(test_users_username)
        # Print f1 score, precision, recall, accuracy, and confusion matrix
        scores.write("K-Nearest Neighbors Classifier"+ "\n")
        scores.write("F1 score: " + str(f1_score(test_users_label, predictions_knn))+ "\n")
        scores.write("Precision: " + str(precision_score(test_users_label, predictions_knn))+ "\n")
        scores.write("Recall: " + str(recall_score(test_users_label, predictions_knn))+ "\n")
        scores.write("Accuracy: " + str(accuracy_score(test_users_label, predictions_knn))+ "\n")
        scores.write("Confusion matrix: " + str(confusion_matrix(test_users_label, predictions_knn))+ "\n")
        scores.write("Classification report: " + str(classification_report(test_users_label, predictions_knn))+ "\n")

        # Create neural network classifier
        nnc = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        # Train the classifier
        nnc.fit(train_users_username, train_users_label)
        # Predict the class of test_users
        predictions_nnc = nnc.predict(test_users_username)
        # Print f1 score, precision, recall, accuracy, and confusion matrix
        scores.write("Neural Network Classifier"+ "\n")
        scores.write("F1 score: " + str(f1_score(test_users_label, predictions_nnc))+ "\n")
        scores.write("Precision: " + str(precision_score(test_users_label, predictions_nnc))+ "\n")
        scores.write("Recall: " + str(recall_score(test_users_label, predictions_nnc))+ "\n")
        scores.write("Accuracy: " + str(accuracy_score(test_users_label, predictions_nnc))+ "\n")
        scores.write("Confusion matrix: " + str(confusion_matrix(test_users_label, predictions_nnc))+ "\n")
        scores.write("Classification report: " + str(classification_report(test_users_label, predictions_nnc))+ "\n")

        # Create a deep neural network classifier with a hidden layer of size 10
        dnnc = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,), random_state=1)
        # Train the classifier
        dnnc.fit(train_users_username, train_users_label)
        # Predict the class of test_users
        predictions_dnnc = dnnc.predict(test_users_username)
        # Print f1 score, precision, recall, accuracy, and confusion matrix
        scores.write("Deep Neural Network Classifier"+ "\n")
        scores.write("F1 score: " + str(f1_score(test_users_label, predictions_dnnc))+ "\n")
        scores.write("Precision: " + str(precision_score(test_users_label, predictions_dnnc))+ "\n")
        scores.write("Recall: " + str(recall_score(test_users_label, predictions_dnnc))+ "\n")
        scores.write("Accuracy: " + str(accuracy_score(test_users_label, predictions_dnnc))+ "\n")
        scores.write("Confusion matrix: " + str(confusion_matrix(test_users_label, predictions_dnnc))+ "\n")
        scores.write("Classification report: " + str(classification_report(test_users_label, predictions_dnnc))+ "\n")

        # For all mispredictions, print the predicted class and the actual class and actual username. (predictions_rfc predictions_nbc predictions_lrc predictions_svc predictions_dtc predictions_knn)
        #print("Random Forest Classifier")
        mispred_rfc = []
        # For predictions_rfc, print the predicted class and the actual class and actual username.
        for i in range(len(predictions_rfc)):
            if predictions_rfc[i] != test_users_label[i]:
                mispred_rfc.append([predictions_rfc[i], test_users_label[i], test_users_username[i]])
                #print("Predicted class: " + str(predictions_rfc[i]) + " Actual class: " + str(test_users_label[i]) + " Actual username: " + str(le.inverse_transform(test_users_username[i])))
        
        #print("Naive Bayes Classifier")
        mispred_nbc = []
        for i in range(len(predictions_nbc)):
            if predictions_nbc[i] != test_users_label[i]:
                mispred_nbc.append([predictions_nbc[i], test_users_label[i], test_users_username[i]])
                #print("Predicted class: " + str(predictions_nbc[i]) + " Actual class: " + str(test_users_label[i]) + " Actual username: " + str(le.inverse_transform(test_users_username[i])))

        #print("Logistic Regression Classifier")
        mispred_lrc = []
        for i in range(len(predictions_lrc)):
            if predictions_lrc[i] != test_users_label[i]:
                mispred_lrc.append([predictions_lrc[i], test_users_label[i], test_users_username[i]])
                #print("Predicted class: " + str(predictions_lrc[i]) + " Actual class: " + str(test_users_label[i]) + " Actual username: " + str(le.inverse_transform(test_users_username[i])))

        #print("Support Vector Classifier")
        mispred_svc = []
        for i in range(len(predictions_svc)):
            if predictions_svc[i] != test_users_label[i]:
                mispred_svc.append([predictions_svc[i], test_users_label[i], test_users_username[i]])
                #print("Predicted class: " + str(predictions_svc[i]) + " Actual class: " + str(test_users_label[i]) + " Actual username: " + str(le.inverse_transform(test_users_username[i])))

        #print("Decision Tree Classifier")
        mispred_dtc = []
        for i in range(len(predictions_dtc)):
            if predictions_dtc[i] != test_users_label[i]:
                mispred_dtc.append([predictions_dtc[i], test_users_label[i], test_users_username[i]])
                #print("Predicted class: " + str(predictions_dtc[i]) + " Actual class: " + str(test_users_label[i]) + " Actual username: " + str(le.inverse_transform(test_users_username[i])))

        #print("K-Nearest Neighbors Classifier")
        mispred_knn = []
        for i in range(len(predictions_knn)):
            if predictions_knn[i] != test_users_label[i]:
                mispred_knn.append([predictions_knn[i], test_users_label[i], test_users_username[i]])
                #print("Predicted class: " + str(predictions_knn[i]) + " Actual class: " + str(test_users_label[i]) + " Actual username: " + str(le.inverse_transform(test_users_username[i])))

        #print("Neural Network Classifier")
        mispred_nnc = []
        for i in range(len(predictions_nnc)):
            if predictions_nnc[i] != test_users_label[i]:
                mispred_nnc.append([predictions_nnc[i], test_users_label[i], test_users_username[i]])
                #print("Predicted class: " + str(predictions_nnc[i]) + " Actual class: " + str(test_users_label[i]) + " Actual username: " + str(le.inverse_transform(test_users_username[i])))

        #print("Deep Neural Network Classifier")
        mispred_dnnc = []
        for i in range(len(predictions_dnnc)):
            if predictions_dnnc[i] != test_users_label[i]:
                mispred_dnnc.append([predictions_dnnc[i], test_users_label[i], test_users_username[i]])
                #print("Predicted class: " + str(predictions_dnnc[i]) + " Actual class: " + str(test_users_label[i]) + " Actual username: " + str(le.inverse_transform(test_users_username[i])))

        
        mispreds_df = mispreds_df.append({'classifier': 'Random Forest Classifier', 'array': mispred_rfc, 'training_ratio': rat,'array_size': len(mispred_rfc)}, ignore_index=True)
        mispreds_df = mispreds_df.append({'classifier': 'Naive Bayes Classifier', 'array': mispred_nbc, 'training_ratio': rat,'array_size': len(mispred_nbc)}, ignore_index=True)
        mispreds_df = mispreds_df.append({'classifier': 'Logistic Regression Classifier', 'array': mispred_lrc, 'training_ratio': rat,'array_size': len(mispred_lrc)}, ignore_index=True)
        mispreds_df = mispreds_df.append({'classifier': 'Support Vector Classifier', 'array': mispred_svc, 'training_ratio': rat,'array_size': len(mispred_svc)}, ignore_index=True)
        mispreds_df = mispreds_df.append({'classifier': 'Decision Tree Classifier', 'array': mispred_dtc, 'training_ratio': rat,'array_size': len(mispred_dtc)}, ignore_index=True)
        mispreds_df = mispreds_df.append({'classifier': 'K-Nearest Neighbors Classifier', 'array': mispred_knn, 'training_ratio': rat,'array_size': len(mispred_knn)}, ignore_index=True)
        mispreds_df = mispreds_df.append({'classifier': 'Neural Network Classifier', 'array': mispred_nnc, 'training_ratio': rat,'array_size': len(mispred_nnc)}, ignore_index=True)
        mispreds_df = mispreds_df.append({'classifier': 'Deep Neural Network Classifier', 'array': mispred_dnnc, 'training_ratio': rat,'array_size': len(mispred_dnnc)}, ignore_index=True)

        rat += rat_increase
        if rat >= 1.0:
            #print("End of training")
            break
        #print("Next training ratio: " + str(rat))
        scores.write("Next training ratio: " + str(rat) + "\n")
    

except Exception as e:
    #print("Error: " + str(e))
    traceback.print_exc()


# Sort by len(mispreds_df.iloc[i]['array'])
mispreds_df = mispreds_df.sort_values(by=['array_size'], ascending=False)

# Print classifier, training ratio, and misprediction array size
for i in range(len(mispreds_df)):
    print("Classifier: " + str(mispreds_df.iloc[i]['classifier']) + " Training ratio: " + str(mispreds_df.iloc[i]['training_ratio']) + " Array size: " + str(mispreds_df.iloc[i]['array_size']) + " Length: " + str(len(mispreds_df.iloc[i]['array'])))

# Sort by classifier and training ratio
mispreds_df = mispreds_df.sort_values(by=['classifier', 'training_ratio'], ascending=True)

# For each classifier, visualize array_size for each training ratio with a bar chart
# Training ratio starts from 0.05 and ends before 1.0
for i in mispreds_df['classifier'].unique():
    plt.figure()
    plt.title(str(i))
    plt.xlabel('Training Ratio')
    plt.ylabel('Misclassified Users')
    plt.ylim(0, max(mispreds_df.loc[mispreds_df['classifier'] == i, 'array_size']) + 1)
    plt.xlim(0, 1)
    plt.xticks(np.arange(0.05, 1.0, 0.05), rotation=90)
    plt.bar(mispreds_df.loc[mispreds_df['classifier'] == i, 'training_ratio'], mispreds_df.loc[mispreds_df['classifier'] == i, 'array_size'])
    plt.show()

    
    
