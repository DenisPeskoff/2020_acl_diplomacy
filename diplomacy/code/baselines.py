import json
from sklearn.metrics import f1_score
from random import uniform
import jsonlines

test_set = 'data/test.jsonl'
repeats = 500

#convert conversations into single messages
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

sender_labels = []
receiver_labels = []

if __name__ == '__main__':
    # loop through all messages that aren't "NOANNOTATION" and calculate sender and receiver metrics
    with jsonlines.open(test_set, 'r') as reader:
        test = list(reader)
        for msg in aggregate(test):
            #don't drop for sender
            if msg['sender_annotation'] == True:
                sender_labels.append(0)
            elif msg['sender_annotation'] == False:
                sender_labels.append(1)
            #only annotated ones for receiver
            if msg['receiver_annotation'] == True or msg['receiver_annotation'] == False:
                if msg['receiver_annotation'] == True:
                    receiver_labels.append(0)
                elif msg['receiver_annotation'] == False:
                    receiver_labels.append(1)


    print("Total sender samples:", len(sender_labels))
    print("Total receiver samples:", len(receiver_labels))
    type = ['macro', 'binary']
    for metric in type:
        sender_f1s, sender_majority_f1s = [], []
        receiver_f1s, receiver_majority_f1s = [], []
        for _ in range(repeats):
            sender_preds, sender_majority = [], []
            receiver_preds, receiver_majority = [], []
            for _ in range(len(sender_labels)):
                if uniform(0,1) < .5 :
                    sender_preds.append(0)
                else:
                    sender_preds.append(1)
                sender_majority.append(0)

            for _ in range(len(receiver_labels)):
                if uniform(0,1) < .5 :
                    receiver_preds.append(0)
                else:
                    receiver_preds.append(1)
                receiver_majority.append(0)


            sender_f1s.append(f1_score(sender_labels, sender_preds, pos_label=1, average = metric))
            receiver_f1s.append(f1_score(receiver_labels, receiver_preds, pos_label=1, average= metric))
            sender_majority_f1s.append(f1_score(sender_labels, sender_majority, pos_label=1, average = metric))
            receiver_majority_f1s.append(f1_score(receiver_labels, receiver_majority, pos_label=1, average= metric))

        print(metric)
        print('Sender Random F1', sum(sender_f1s)/repeats)
        print('Receiver Random F1', sum(receiver_f1s)/repeats)
        print('Sender Majority Class F1', sum(sender_majority_f1s)/repeats)
        print('Receiver Majority Class F1', sum(receiver_majority_f1s)/repeats)
