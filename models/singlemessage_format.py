import json
from os.path import join
from random import shuffle, sample


def to_single_message_format(gamefile):
    messages = []
    with open(gamefile) as inh:
        for ln in inh:
            conversation = json.loads(ln)
            for msg, sender_label, receiver_label, score_delta \
                in zip(conversation['messages'],conversation['sender_labels'], \
                    conversation['receiver_labels'], conversation['game_score_delta']):
                messages.append({'message': msg, 'receiver_annotation': receiver_label,\
                    'sender_annotation':sender_label, 'score_delta': int(score_delta)})
    shuffle(messages)    
    return messages

def write_single_messages(messages, outfile):
    with open(outfile, "w") as outh:
        for msg in messages:
            outh.write(json.dumps(msg)+'\n')

if __name__ == '__main__':
    ROOT = '../data/'
    
    write_single_messages(to_single_message_format(join(ROOT, 'validation.jsonl')), 
                                                        join(ROOT, 'validation_sm.jsonl'))
    write_single_messages(to_single_message_format(join(ROOT, 'train.jsonl')), 
                                                        join(ROOT, 'train_sm.jsonl'))
    write_single_messages(to_single_message_format(join(ROOT, 'test.jsonl')), 
                                                        join(ROOT, 'test_sm.jsonl'))
