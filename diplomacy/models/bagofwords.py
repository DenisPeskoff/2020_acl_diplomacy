#BoW Version
import jsonlines
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import RFE
import numpy as np
from scipy.sparse import csr_matrix



#change the format from list of lists into a single list
def aggregate(dataset):
    messages = []
    rec = []
    send = []
    power = []
    for dialogs in dataset:
        messages.extend(dialogs['messages'])
        rec.extend(dialogs['receiver_labels'])
        send.extend(dialogs['sender_labels'])
        #ONLY FOR POWER VERSION
        power.extend(dialogs['game_score_delta'])
    #print(len(rec), len(send), len(messages))
    merged = []
    for i, item in enumerate(messages):
        merged.append({'message':item, 'sender_annotation':send[i], 'receiver_annotation':rec[i], 'score_delta':int(power[i])})
    return merged

def convert_to_binary(dataset):
    binary_data = []
    matrix = []
    for message in dataset:
        #drop the instances that were not annotated
        if message['receiver_annotation'] == True or message['receiver_annotation'] == False:
            pass
        else:
            #CHANGE BACK IF YOU WANT TO DROP UN-ANNOTATED RECEIVER MESSAGES
            pass
            #continue
            
        binary = []
        
        #a severe power skew (a difference of 5 or more supply centers) has the best result
        if message['score_delta'] > 4:
            binary.append(1)
        else:
            binary.append(0)
            
        if message['score_delta'] < -4:
            binary.append(1)
        else:
            binary.append(0)
                
        #add class label
        if message['sender_annotation'] == False:
            binary.append(0)
        else:
            binary.append(1)

        binary_data.append(binary)
    return binary_data

#split up x and y label in data
def split_xy(data):
    X, y = [], []
    for line in data:
        x = line[:len(line)-1]
        single_y = line[len(line)-1]
        X.append(x)
        y.append(single_y)
    return(X, y)


if __name__ == '__main__':
    #import data.  Specify directory path
    data_path = '../../data/'#'diplomacy_model/data/sep11/by_game/'

    with jsonlines.open(data_path+'train.jsonl', 'r') as reader:
        train = list(reader)
    with jsonlines.open(data_path+'validation.jsonl', 'r') as reader:
        dev = list(reader)
    with jsonlines.open(data_path+'test.jsonl', 'r') as reader:
        test = list(reader)


    vectorizer = CountVectorizer()
    corpus = [message['message'].lower() for message in aggregate(train)] #if message['receiver_annotation'] != None]
    X = vectorizer.fit_transform(corpus)
    newVec = CountVectorizer(vocabulary=vectorizer.vocabulary_)
    y = newVec.fit_transform([message['message'].lower() for message in aggregate(test)]) #if message['receiver_annotation'] != None]) #

    train = convert_to_binary(aggregate(train))
    #validation set not used for consistency with neural
    test = convert_to_binary(aggregate(test))

    train = split_xy(train)
    test = split_xy(test)

    append_power_x = np.append(X.toarray(), train[0], axis = 1)
    append_power_y = np.append(y.toarray(), test[0], axis = 1)

    X = csr_matrix(append_power_x)
    y = csr_matrix(append_power_y)


    logmodel = LogisticRegression(class_weight = 'balanced', max_iter=1000)

    #RFE VERSION
#    rfe = RFE(logmodel, 20)
#    rfe = rfe.fit(X, train[1])
#    print(rfe.support_)
#    print(rfe.ranking_)
#    predictions = fit.predict(y)
#    print(fit.score(y, test[1]))
#    print(classification_report(test[1],predictions))

    logmodel.fit(X, train[1])
    predictions = logmodel.predict(y)
    #print out top words
    print ("Examples of words that skew towards a lie are:")
    for index,a in enumerate(logmodel.coef_[0]):
        if a > 1.75:
            print(vectorizer.get_feature_names()[index], a)

    print(classification_report(test[1],predictions, digits=3))

