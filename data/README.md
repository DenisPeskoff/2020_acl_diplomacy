Each file is a jsonlines file containing an entire game dialog per line. Each object has the following fields: 
An alternate representation of the data, by message not by conversation, is available as part of ConvoKit: https://convokit.cornell.edu/documentation/diplomacy.html

**speakers:** the sender of the message (string format.  russia, turkey, england...)

**receivers:** the receiver of the message (string format.  russia, turkey, england...)

**messages:** the raw message string (string format.  ranges in length from one word to paragraphs in length)

**sender_labels:** indicates if the sender of the message selected that the message is truthful, true, or deceptive, false.  This is used for our ACTUAL_LIE calculation (true/false which can be bool or string format) 

**receiver_labels:** indicates if the receiver of the message selected that the message is perceived as truthful, true, or deceptive, false.  In <10% of the cases, no annotation was received.  This is used for our SUSPECTED_LIE calculation (string format.  true/false/"NOANNOTATION" ) 

**game_score:** the current game score---supply centers---of the sender  (string format that ranges can range from  0 to 18)

**score_delta:** the current game score---supply centers---of the sender minus the game score of the recipient (string format that ranges from -18 to 18)

**absolute_message_index:** the index the message is in the entire game, across all dialogs (int format)

**relative_message_index:**  the index of the message in the current dialog (int format)

**seasons:** the season in Diplomacy, associated with the year (string format. Spring, Fall, Winter)

**years:** the year in Diplomacy, associated with the season (string format.  1901 through 1918)

**game_id:** which of the 12 games the dialog comes from (int format ranging from 1 to 12)

*UPDATE* We additionally have all game data (moves) available at https://github.com/DenisPeskov/2020_acl_diplomacy/blob/master/utils/ExtraGameData.zip
