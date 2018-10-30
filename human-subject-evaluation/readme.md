There experiments are set up to understand the role of context vs the role of prosody when predicting the dialog act. 
# Creating the data #
bash humanSubjectEval.sh <no-files> <br />
<br />
This will create csv files for the utterance transcripts and audio files for utterance segments. The corresponding ground truth labels are also stored in a csv file. All the dataset is generated for sorted version and shuffled version of the utterances from the files. For the shuffled version, all the utterances from all the given files are shuffled . 

# Guidelines for evaluation #
Each utterance (audio/transcript) has to be labelled into one of the following 9 classes described below with examples. <br />
1. statement <br />
A non-opinion statement; descriptive and/or narrative (listener has no basis to dispute)
2. backchannel <br />
Continuers in a conversation. Does not include backchannel questions.
3. opinion <br />
Opinion statement; viewpoint, from personal opinions to proposed general facts  (listener could have basis to dispute)
4. yn_q <br />
These are the yes-no questions - not declarative.
5. close <br />
6. wh_q <br />
These are the wh-questions - not open ended or declarative.
7. agree <br />
This marks the degree to which speaker accepts some previous proposal, plan, opinion, or statement.
8. apprec <br />
A backchannel/continuer which functions to express slightly more emotional involvement and support than just "uh-huh". An example transcript clip for a single conversation side - 

utt_id        | transcript        | label
------------- | ----------------- | -------------
sw2027_96     | Oh yuck           | *apprec*
sw2027_97     | Yeah              | *backchannel*
sw2027_98     | that 's terrible  | *apprec*
sw2027_98     | that 's terrible  | *apprec*
sw2027_100    | gosh              | *apprec*
9. other <br />
Anything that does not fit into one of the above classes should be marked as 'other' 
