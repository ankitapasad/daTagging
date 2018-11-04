'''
This script generates dataset for human-subject evaluation experiments
Dialog acts clubbed:
statement 
backchannel: bachchannel, acknowledge, backchannel_q
opinion
# agree: agree, yes # remove this since we do not have a separate class for "no"
ynq: yn_q, yn_decl_q, tag_q
# close # remove this since we are not involvong 'open' and this doesn't 
        # fit well with other tags in the list
q: wh_q, open_q, decl_q # questions which expect an answer
apprec
other: other, abandon, uninterp, no, hedge, excluded, quote, sum, affirm, directive, 
       repeat, completion, hold, reject, neg, answer, repeat_q, open, ans_dispref,
       or, commit, maybe, agree, yes, third_pty, self_talk, apology, downplay, thank,
       rhet_q, close
'''

import xml.etree.ElementTree as ET
import os, sys
from random import shuffle
import getDatafromXML as getDict
import copy
import scipy.io.wavfile as wav

def updateText(convIds,utt,labels,idDict,GTfile,file):
	for i in range(len(sortedConvIds)):
		dataPoint = convIds[i]+','+utt[i]+'\n'
		labelPoint = convIds[i]+','+labels[i]+','+str(idDict[labels[i]])+'\n'
		GTfile.write(labelPoint)
		file.write(dataPoint)

def updateAudio(startDir,convIds,startTimes,endTimes):
	startDir1 = "/share/data/speech/ankitap/topicID/"
	currentWav = 'temp'
	count = 1
	for i in range(len(convIds)):
		newWav = convIds[i].split('_')[0]
		if(currentWav != newWav):
			currentWav = copy.deepcopy(newWav)
			f = currentWav[:2]+'0'+currentWav[2:]+'.wav'
			(rate,sig) = wav.read(startDir1+"data/audio/all/"+f)
			sigA = sig.T[0] # channel A
			sigB = sig.T[1] # channel B
			count = 1
		if(sortedConvIds[i].split('_')[1]=='A'): sig = copy.deepcopy(sigA)
		else: sig = copy.deepcopy(sigB)
		try:
			st = float(startTimes[i])
			et = float(endTimes[i])
			uttAudio = sig[int(st*rate):int(et*rate)]
			filename = startDir+'audio/sorted/'+str(currentWav)+'/'+str(count)+'.wav'
			wav.write(filename, rate, uttAudio)
		except: pass
		count += 1


if __name__=="__main__":
	noFiles = int(sys.argv[1])

	dataDir = '/share/data/speech/Datasets/nxt_switchboard_ann/xml/'
	startDir = '/share/data/speech/ankitap/daTagging/humanSubjectEval/sample/'

	if(sys.argv[2]=='dataprep'):
		fileListComplete = open(startDir+'../../dataset/daFiles/trainFiles.txt').readlines()
		shuffle(fileListComplete)
		fileList = fileListComplete[:noFiles]
		for i in range(noFiles):
			fileList[i] = fileList[i].strip('\n')
		# returnList = " ".join([fileName for fileName in fileList])
		returnList = " ".join(fileList)
		print(returnList)
		outfile = open(startDir+'fileList.txt',"w")
		outfile.write("\n".join(fileList))
		outfile.close()

	elif(sys.argv[2]=='gendata'):
		fileList = open(startDir+'fileList.txt').readlines()
		namespaceIdentifier = '{http://nite.sourceforge.net/}'
		sortedFile = open(startDir+'sorted.csv',"w")
		shuffledFile = open(startDir+'shuffled.csv',"w")
		sortedGTFile = open(startDir+'sortedGT.csv',"w")
		shuffledGTFile = open(startDir+'shuffledGT.csv',"w")

		truncDAs = ['statement','backchannel','acknowledge','backchannel_q','opinion','yn_q','yn_decl_q','tag_q','wh_q','open_q','decl_q','apprec']
		idDict = getDict.labelToId(truncDAs)
		sortedLabels =[]
		sortedUtt = []
		sortedConvIds = []
		sortedStartTime = []
		sortedEndTime = []
		name = str(noFiles)+'_'+fileList[0].strip('\n')
		for fileName in fileList:
			fileName = fileName.strip('\n')
			if(fileName==fileList[0]): name += '_'
			else: name += (fileName + '_')
			fileName1 = fileName+'.A.dialAct.xml'
			fileName2 = fileName+'.B.dialAct.xml'
			# updated with utterances from both sides
			dialActDict, uttDict, sideDict = getDict.getDialAct(fileName1,namespaceIdentifier,dataDir,truncDAs=truncDAs)
			dialActDict, uttDict, sideDict = getDict.getDialAct(fileName2,namespaceIdentifier,dataDir,dialActDict,uttDict,sideDict,truncDAs=truncDAs)
			wordDict = getDict.getUtt(fileName1,namespaceIdentifier,dataDir)
			wordDict = getDict.getUtt(fileName2,namespaceIdentifier,dataDir,wordDict)
			
			keys = list(range(1,len(uttDict)+1))
			for key in keys: 
				convId = fileName1.split('.')[0]+'_'+sideDict[key] # fileName_side
				utt = []
				label = dialActDict[key]
				for wordId in uttDict[key]:
					if(wordId in wordDict):
						if(len(utt)==0): startTime = wordDict[wordId][1]
						endTime = wordDict[wordId][2]
						utt.append(wordDict[wordId][0])
				utt = ' '.join(utt)

				sortedConvIds.append(convId)
				sortedLabels.append(label)
				sortedUtt.append(utt)
				sortedStartTime.append(startTime)
				sortedEndTime.append(endTime)

		shuffledData = list(zip(sortedConvIds,sortedUtt,sortedLabels,sortedStartTime,sortedEndTime))
		shuffle(shuffledData)
		shuffledConvIds,shuffledUtt,shuffledLabels,shuffledStartTime,shuffledEndTime = zip(*shuffledData)
		
		updateText(sortedConvIds,sortedUtt,sortedLabels,idDict,sortedGTFile,sortedFile)
		updateText(shuffledConvIds,shuffledUtt,shuffledLabels,idDict,shuffledGTFile,shuffledFile)
		
		updateAudio(startDir,sortedConvIds,sortedStartTime,sortedEndTime)
		updateAudio(startDir,shuffledConvIds,shuffledStartTime,shuffledEndTime)

		print(name[:-1])
