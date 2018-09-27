'''
Generates dataset.csv which is then sorted from the largest to smallest fraction
and saved.
label and the fraction of times that particular label appears in the dataset (train/dev/test)
'''
startDir = '/Users/ankita/TTIC/code/DA-tagging/'

def getFreq(label='train',tagFreq = None):
	dataLines = open(startDir+'dataSplit/fromPaper/dataset_'+label+'.txt').readlines()
	dur = 0
	numNonalignedUtt = 0
	if(not tagFreq): tagFreq = {}
	for line in dataLines:
		try:
			startTime = float(line.split('\t')[3])
			endTime = float(line.split('\t')[4].strip('\n'))
			dur += (endTime - startTime)
		except:
			numNonalignedUtt += 1
		tag = line.split('\t')[2]
		if(label=='train'):
			if(tag not in tagFreq): tagFreq[tag] = [0,0,0]
			tagFreq[tag][0] += 1
		elif(label=='dev'): tagFreq[tag][1] += 1
		else: tagFreq[tag][2] += 1

	total = len(dataLines)

	return tagFreq, total, (dur/3600), numNonalignedUtt

def getStats():
	tagFreq, trainTot, trainDur, trainNum = getFreq('train')
	tagFreq, devTot, devDur, devNum = getFreq('dev',tagFreq)
	tagFreq, testTot, testDur, testNum = getFreq('test',tagFreq)
	print("# of non-aligned utterances: ",trainNum,devNum,testNum)
	print("# of hours: ",trainDur,devDur,testDur)
	dataStatsFile = open('dataStats.csv',"w")
	for key in tagFreq:
		# dataline = key+','+"{0:.2f}".format(100*tagFreq[key][0]/trainTot)+','+"{0:.2f}".format(100*tagFreq[key][1]/devTot)+','+"{0:.2f}".format(100*tagFreq[key][2]/testTot)
		dataline = key+','+str(tagFreq[key][0]/trainTot)+','+str(tagFreq[key][1]/devTot)+','+str(tagFreq[key][2]/testTot)
		dataStatsFile.write(dataline+'\n')

	dataStatsFile.close()

getStats()