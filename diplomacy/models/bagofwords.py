#BoW Version
import jsonlines
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import RFE
import numpy as np
from scipy.sparse import csr_matrix
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
import sys
import warnings
warnings.filterwarnings("ignore")

def is_number(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False

def spacy_tokenizer(text):
    return [tok.text if not is_number(tok.text) else '_NUM_' for tok in nlp(text)]



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
            if TASK == "SENDER":
                pass
            elif TASK == "RECEIVER":
                continue
            
        binary = []
        
        #a severe power skew (a difference of 5 or more supply centers) has the best result
        if POWER == "y":
            if message['score_delta'] > 4:
                binary.append(1)
            else:
                binary.append(0)

            if message['score_delta'] < -4:
                binary.append(1)
            else:
                binary.append(0)
#binary.append(message['score_delta'])
        if TASK == "SENDER":
            annotation ='sender_annotation'
        elif TASK == "RECEIVER":
            annotation ='receiver_annotation'
        #add class label
        if message[annotation] == False:
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


def log_reg(train, test):
    #Convert train data into a vector
    vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, stop_words=STOP_WORDS, strip_accents = 'unicode')
    if TASK == "SENDER":
        corpus = [message['message'].lower() for message in aggregate(train)]
    elif TASK == "RECEIVER": #for receivers, drop all missing annotations
        corpus = [message['message'].lower() for message in aggregate(train) if message['receiver_annotation'] != None]
    X = vectorizer.fit_transform(corpus)

    #Convert test data into a vector, only based on train vocab
    newVec = CountVectorizer(tokenizer=spacy_tokenizer, vocabulary=vectorizer.vocabulary_, stop_words=STOP_WORDS, strip_accents = 'unicode')
    if TASK == "SENDER":
        y = newVec.fit_transform([message['message'].lower() for message in aggregate(test)])
    elif TASK == "RECEIVER": #for receivers, drop all missing annotations
        y = newVec.fit_transform([message['message'].lower() for message in aggregate(test) if message['receiver_annotation'] != None])

    #only used for getting lie/not lie labels
    train = convert_to_binary(aggregate(train))
    #validation set not used for consistency with neural
    test = convert_to_binary(aggregate(test))
    train = split_xy(train)
    test = split_xy(test)

    #append power to the matrix
    append_power_x = np.append(X.toarray(), train[0], axis = 1)
    append_power_y = np.append(y.toarray(), test[0], axis = 1)

    #code to scale features, if power is added as a raw value, not binary feature.  Worse than binary so not sued
    #    from sklearn.preprocessing import StandardScaler
    #    scaler = StandardScaler()
    #    append_power_x = scaler.fit_transform(append_power_x)
    #    append_power_y = scaler.fit_transform(append_power_y)

    #convert matrix back to sparse format
    X = csr_matrix(append_power_x)
    y = csr_matrix(append_power_y)

    #balanced class weight is important, since otherwise it will only learn majority class
    logmodel = LogisticRegression(class_weight = 'balanced', max_iter=1000)


    #RFE VERSION.  Worse than log regression and long run time so not used
#    rfe = RFE(logmodel)
#    rfe = rfe.fit(X, train[1])
#    print(rfe.support_)
#    print(rfe.ranking_)
#    predictions = fit.predict(y)
#    print(fit.score(y, test[1]))
#    print(classification_report(test[1],predictions))
    
    logmodel.fit(X, train[1])
    predictions = logmodel.predict(y)
    #code to print out top words
    #    print ("Examples of words that skew towards a lie are:")
    #    for index,a in enumerate(logmodel.coef_[0]):
    #        if a > 1.75:
    #            print(vectorizer.get_feature_names()[index], a)

    print(classification_report(test[1],predictions, digits=3))


if __name__ == '__main__':
    if len(sys.argv) == 2 or len(sys.argv) == 3:
        POWER = "y"
        if sys.argv[1] == 's':
            TASK = "SENDER"
        elif sys.argv[1] == 'r':
            TASK = "RECEIVER"
        else:
            print("Specify s for sender or r for receiver")
            exit()
        if len(sys.argv) == 3:
            if sys.argv[2] == 'n':
                POWER = sys.argv[2]
            elif sys.argv[2] == 'y':
                POWER = sys.argv[2]
            else:
                print("Specify y for including power and n for not including it e.g.: python bagofwords.py s n")
                exit()
    else:
        print("Specify s for sender or r for receiver e.g.:  python bagofwords.py s")
        exit()
    #import data.  Specify directory path
    data_path = '../../data/'#'diplomacy_model/data/sep11/by_game/'

    with jsonlines.open(data_path+'train.jsonl', 'r') as reader:
        train = list(reader)
    #VAL NOT USED IN LOG REG FOR CONSISTENCY WITH NEURAL
    #with jsonlines.open(data_path+'validation.jsonl', 'r') as reade
        #dev = list(reader)
    with jsonlines.open(data_path+'test.jsonl', 'r') as reader:
        test = list(reader)

    #spacy used for tokenization 
    nlp = English()

    log_reg(train, test)


