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

utt_id        | transcript        | label
------------- | ----------------- | -------------
sw3408_A_28     | You know I read a study once | statement
sw3408_A_29    | nd it said that uh like thirty-four percent of uh college students actually graduate in four years from a four year program             | statement
sw3408_B_30     | Oh really  | backchannel
sw3408_B_31     | *That sounds* | *opinion*
sw3408_A_32    | It took me it took me five year            | statement
sw3408_A_33    | Yeah  | backchannel
sw3408_A_34    | *So you just you get started* | *opinion*
sw3408_A_35    | *and you change your mind*              | *opinion*
sw3408_A_36     | *or you want to pick up a second major*  | *opinion*
sw3408_A_37    | *if you 're management you you thought I I 'll take that marketing*             | *opinion*
sw3408_A_38     | it 's just only was three more classes  | statement

3. backchannel <br />
Continuers in a conversation. Includes backchannel placed in the form of questions as well.
4. ynq <br />
Yes-no questions, including the tag questions. 

utt_id        | transcript        | label
------------- | ----------------- | -------------
sw3411_A_75     | *So you say you 've always have preferred General Motors products* | *ynq*
sw3411_B_76    | Yeah              | other
sw2027_B_177     | I do  | statement

This is an example transcript clip with a tag question

utt_id        | transcript        | label
------------- | ----------------- | -------------
sw4171_B_102     | *all that stuff huh* | *ynq*
sw4171_A_103    | Yeah              | other
sw4171_A_104     | but they no no longer  | statement

5. question <br />
Questions (which are not yes-no questions), could be open ended or declarative.

utt_id        | transcript        | label
------------- | ----------------- | -------------
sw3411_A_82     | *What kind of uh General Motors cars have you had in the past* | *question*
sw3411_B_83    | Mostly Oldsmobiles              | statement
sw3411_A_84    | Oldsmobiles              | other
sw3411_A_85     | Those are real nice riding cars too  | opinion

utt_id        | transcript        | label
------------- | ----------------- | -------------
sw3503_A_1     | *Well how do you feel about capital punishment* | *question*
sw3503_B_2    | Well I I just last Friday got off a capital murder case              | statement
sw3503_A_3    | Oh you did              | backchannel

6. apprec <br />
A backchannel/continuer which functions to express slightly more emotional involvement and support than just "uh-huh". 

utt_id        | transcript        | label
------------- | ----------------- | -------------
sw2027_A_175     | but you know I mean it was just completely miserable for her  | statement
sw2027_B_176     | Yeah              | backchannel
sw2027_B_177     | *that 's terrible*  | *apprec*

7. other <br />
Anything that does not fit into one of the above classes should be marked as 'other'. 
