There experiments are set up to understand the role of context vs the role of prosody when predicting the dialog act. 
# Creating the data #
bash humanSubjectEval.sh `<no-files>` <br />
<br />
This will create csv files for the utterance transcripts and audio files for utterance segments. The corresponding ground truth labels are also stored in a csv file. All the dataset is generated for sorted version and shuffled version of the utterances from the files. For the shuffled version, all the utterances from all the given files are shuffled . 

# Guidelines for evaluation #
Each utterance (audio/transcript) has to be labelled into one of the following 9 classes described below with examples. <br />
1. statement <br />
A non-opinion statement; descriptive and/or narrative (listener has no basis to dispute)
2. opinion <br />
Opinion statement; viewpoint, from personal opinions to proposed general facts  (listener could have basis to dispute)
3. backchannel <br />
Continuers in a conversation. Includes backchannel placed in the form of questions as well.
4. ynq <br />
Yes-no questions, including the tag questions. 

utt_id        | transcript        | label
------------- | ----------------- | -------------
sw3411_A_75     | So you say you 've always have preferred General Motors products | *ynq*
sw3411_B_76    | Yeah              | other
sw2027_B_177     | I do  | statement

5. question <br />
Questions (which are not yes-no questions), could be open ended or declarative
6. apprec <br />
A backchannel/continuer which functions to express slightly more emotional involvement and support than just "uh-huh". 

utt_id        | transcript        | label
------------- | ----------------- | -------------
sw2027_A_175     | but you know I mean it was just completely miserable for her  | statement
sw2027_B_176     | Yeah              | backchannel
sw2027_B_177     | that 's terrible  | *apprec*

7. other <br />
Anything that does not fit into one of the above classes should be marked as 'other'. 
