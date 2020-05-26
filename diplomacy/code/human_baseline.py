import json
from sklearn.metrics import f1_score, accuracy_score
from random import uniform
import jsonlines

test_set = 'data/test.jsonl'

total_tt, total_tn, total_nt, total_nn = 0, 0, 0, 0
sender_labels = []
receiver_labels = []


def aggregate(dataset):
    messages = []
    rec = []
    send = []
    for dialogs in dataset:
        messages.extend(dialogs['messages'])
        rec.extend(dialogs['receiver_labels'])
        send.extend(dialogs['sender_labels'])
    merged = []
    for i, item in enumerate(messages):
        merged.append({'message':item, 'sender_annotation':send[i], 'receiver_annotation':rec[i]})
    return merged

with jsonlines.open(test_set, 'r') as reader:
    train = list(reader)

    for msg in aggregate(train):
        
        #this means the denominator for sent lies is slightly different from the rest of the sent ones, as no other calculation depends on both SENDER and RECEIVER labels in our table.  However, we suspect that the NOANNOTATIONS are done at random.  Assuming majority class in this calculation leads to a human baseline of 0.207 Lie F1/ 0.572 macro, which is ~.02 Lie and ~0.1 macro F1 lower than than our reported results.  
        
        #remove this if statement and uncomment MAJORITY CLASS ASSUMPTION below to calculate a human baseline that assumes NOANNOTATION are majority class of "True".
        #this ensures NOANNOTATION is not included
        if msg['receiver_annotation'] == True or msg['receiver_annotation'] == False :
            #collect sender annotation options
            if msg['sender_annotation'] == True:
                sender_labels.append(0)
                if msg['receiver_annotation'] == True:
                    total_tt +=1
                else:
                    total_tn +=1
            else:
                if msg['receiver_annotation'] == True:# MAJORITY CLASS ASSUMPTION# or msg['receiver_annotation'] == "NOANNOTATION":
                    total_nt +=1
                else:
                    total_nn +=1
                sender_labels.append(1)

            #this can only be true or false due to earlier if condition
            if msg['receiver_annotation'] == True: #MAJORIY CLASS ASSUMPTION #or msg['receiver_annotation'] == "NOANNOTATION":
                receiver_labels.append(0)
            else:
                receiver_labels.append(1)

print('Human baseline, macro:', f1_score(sender_labels, receiver_labels, pos_label=1, average= 'macro'))
print('Human baseline, lie F1:', f1_score(sender_labels, receiver_labels, pos_label=1, average= 'binary'))
print('Overall Accuracy is, ', accuracy_score(sender_labels, receiver_labels))




