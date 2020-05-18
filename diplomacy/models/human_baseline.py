import json
from sklearn.metrics import f1_score, accuracy_score
from random import uniform
import jsonlines

test_set = '../../data/finalsplit_gameid/test.jsonl'

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
                if msg['receiver_annotation'] == True:
                    total_nt +=1
                else:
                    total_nn +=1
                sender_labels.append(1)

            #this can only be true or false due to earlier if condition
            if msg['receiver_annotation'] == True:
                receiver_labels.append(0)
            else:
                receiver_labels.append(1)

print('Human baseline, macro:', f1_score(sender_labels, receiver_labels, pos_label=1, average= 'macro'))
print('Human baseline, lie F1:', f1_score(sender_labels, receiver_labels, pos_label=1, average= 'binary'))



#type = [ 'macro', 'binary'] #'weighted', 'micro',
#    #for metric in type:
#    print("The human baseline is:")
#    print(metric)
#    print('Sender Human F1', f1_score(sender_labels, receiver_labels, pos_label=1, average= metric))

#print('Overall accurcay is, ', accuracy_score(sender_labels, receiver_labels))
#print (total_tt, total_tn, total_nt, total_nn)
