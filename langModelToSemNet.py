# -*- coding: utf-8 -*-
"""langModelToSemNet.py

# Requirements:
Python 3.8+ and PyTorch 1.7+

Anaconda installation instructions:
conda create -n pytorchenv python=3.8
source activate pytorchenv
conda install -c pytorch pytorch=1.7
pip install pytorch-transformers
pip install transformers
pip install nltk

# License:
MIT License

# Usage:
python3 langModelToSemNet.py

# Requirements:

# Description

Language Model (BERT) to Semantic Network (triplet relations) extraction

- Author: Richard Bruce Baxter - Copyright (c) 2021 Baxter AI (baxterai.com)

"""

import torch
from transformers import AutoTokenizer, BertForMaskedLM
import nltk
#nltk.download()	#required to download Models during initialisation (punkt)
#nltk.download('all')	#required to download Models during initialisation (wordnet data)
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import wordnet as wn
import numpy as np

tok = AutoTokenizer.from_pretrained("bert-base-cased")
bert = BertForMaskedLM.from_pretrained("bert-base-cased")

def mainFunction():

	#constructs tuples from noun/verb dictionary and finds probability of their associations using pretrained bert
	
	testHarness = True
	saveTopPredictionsOnly = True
	
	saveMaskProbabilities = True
	printInputText = True
	addMaskProbabilitiesToSemanticNet = False	#not yet coded
	topPredictionsNumber = 1000
	
	if(saveMaskProbabilities):
		saveInputTextSummary = True
		firstLineHeader = True
		maskProbabilitiesTextFileName = "maskProbabilitiesTextFile.txt"
		maskProbabilitiesTextFileDelimiter = ","	#csv format
		maskProbabilitiesTextFile = open(maskProbabilitiesTextFileName, "w")
	
	nounList, verbList, prepositionList = createWordLists()	#createLemmaLists()

	predictionIndexSubject = "subject"
	predictionIndexRelationship = "relationship"
	predictionIndexObject = "object"
	numberOfPredictionIndices = 3	#number of masks #subject, relationship, object

	if(testHarness):
		subjectList = ["dog"]
		relationshipList = ["eats"]
		objectList = nounList
		predictionIndices = [predictionIndexObject]
	else:
		subjectList = nounList
		relationshipList = verbList + prepositionList
		objectList = nounList
		predictionIndices = [predictionIndexSubject, predictionIndexRelationship, predictionIndexObject]
	
	addDeterminers = True #adds "the/a" to nouns
	addAuxiliary = False	#adds "is" to verb/preposition
	determinerText = "the"
	auxiliaryText = "is"
			
	tupleTriplets = []
	
	for predictionIndex in predictionIndices:
		
		subjectListPrediction = subjectList
		relationshipListPrediction = relationshipList
		objectListPrediction = objectList
		
		if(predictionIndex == predictionIndexSubject):
			subjectListPrediction = ["NA"]
		elif(predictionIndex == predictionIndexRelationship):
			relationshipListPrediction = ["NA"]
		elif(predictionIndex == predictionIndexObject):
			objectListPrediction = ["NA"]
		
		for i1, w1 in enumerate(subjectListPrediction):	#subject
			for i2, w2 in enumerate(relationshipListPrediction):	#relationship
				for i3, w3 in enumerate(objectListPrediction):	#object

					#if(i3 != i1):	#optional (do not allow relationships with identical subject and object)

					subjectText = w1 + " "
					relationshipText = w2 + " "
					objectText = w3

					startTextToken = nltk.word_tokenize(subjectText)
					relationshipTextToken = nltk.word_tokenize(relationshipText)
					objectTextToken = nltk.word_tokenize(objectText)

					if((len(startTextToken) == 1) and (len(relationshipTextToken) == 1) and (len(objectTextToken) == 1)):

						if(addDeterminers):
							subjectText = determinerText + " " + subjectText
							objectText = determinerText + " " + objectText
						if(addAuxiliary):
							relationshipText = auxiliaryText + " " + relationshipText

						startTextTokens = nltk.word_tokenize(subjectText)
						relationshipTextTokens = nltk.word_tokenize(relationshipText)
						objectTextTokens = nltk.word_tokenize(objectText)

						if(predictionIndex == predictionIndexSubject):
							endText = relationshipText + objectText
							inputText = f"{tok.mask_token} {endText}."
							maskIndex = 0
						elif(predictionIndex == predictionIndexRelationship):
							startText = subjectText
							endText = objectText
							inputText = f"{startText} {tok.mask_token} {endText}."
							maskIndex = len(startTextTokens)
						elif(predictionIndex == predictionIndexObject):
							startText = subjectText + relationshipText
							inputText = f"{startText} {tok.mask_token}."
							maskIndex = len(startTextTokens)
							
						inputTextSummary = inputText.replace(" ", "_")
						
						if(printInputText):
							print(inputText)

						predictionProbabilities, headerList = getProbabilitiesOfMaskedWord(inputText, maskIndex, firstLineHeader, saveTopPredictionsOnly, topPredictionsNumber)

						if(addMaskProbabilitiesToSemanticNet):
							addTuplesToSemanticNet(predictionProbabilities, subjectText, relationshipText, objectText)
						
						if(saveMaskProbabilities):
							addMaskProbabilitiesToFile(maskProbabilitiesTextFile, predictionProbabilities, maskProbabilitiesTextFileDelimiter, firstLineHeader, headerList, saveInputTextSummary, inputTextSummary)

						if(firstLineHeader):
							if(not saveTopPredictionsOnly):
								firstLineHeader = False
										
	
	
def addMaskProbabilitiesToFile(maskProbabilitiesTextFile, predictionProbabilities, maskProbabilitiesTextFileDelimiter, firstLineHeader, headerList, saveInputTextSummary, inputTextSummary):

	if(firstLineHeader):
		predictionProbabilitiesTextFileLine = ""
		if(saveInputTextSummary):
			predictionProbabilitiesTextFileLine = predictionProbabilitiesTextFileLine + maskProbabilitiesTextFileDelimiter
		for predictedWord in headerList:
			predictionProbabilitiesTextFileLine = predictionProbabilitiesTextFileLine + predictedWord + maskProbabilitiesTextFileDelimiter
		#predictionProbabilitiesTextFileLine = predictionProbabilitiesTextFileLine + "\n"
		maskProbabilitiesTextFile.write(predictionProbabilitiesTextFileLine + "\n")

	predictionProbabilitiesTextFileLine = ""
	if(saveInputTextSummary):
		predictionProbabilitiesTextFileLine = predictionProbabilitiesTextFileLine + inputTextSummary + maskProbabilitiesTextFileDelimiter
	for predictionProbability in predictionProbabilities:			
		predictionProbabilityString = "{:.2f}".format(predictionProbability)
		predictionProbabilitiesTextFileLine = predictionProbabilitiesTextFileLine + str(predictionProbabilityString) + maskProbabilitiesTextFileDelimiter
	maskProbabilitiesTextFile.write(predictionProbabilitiesTextFileLine + "\n")
	
																	
							
def getProbabilitiesOfMaskedWord(inputText, maskIndex, firstLineHeader, saveTopPredictionsOnly, topPredictionsNumber):
	
	input_idx = tok.encode(inputText)
	
	logits = bert(torch.tensor([input_idx]))[0]
	predictionProbabilities = logits[0][maskIndex]
	
	headerList = []
	
	if(saveTopPredictionsOnly):
		predictionProbabilitiesTopTensor = predictionProbabilities.topk(topPredictionsNumber)
		predictionProbabilitiesTopValues = predictionProbabilitiesTopTensor.values.tolist()
		predictionProbabilitiesTopIndices = predictionProbabilitiesTopTensor.indices.tolist()
		predictionProbabilitiesList = predictionProbabilitiesTopValues
		headerList = tok.convert_ids_to_tokens(predictionProbabilitiesTopIndices)
	else:
		predictionProbabilitiesList = predictionProbabilities.detach().numpy().tolist()
		if(firstLineHeader):
			for maskedWordIndex, predictionProbability in enumerate(predictionProbabilitiesList):
				predictedWord = tok.convert_ids_to_tokens(maskedWordIndex)
				headerList.append(predictedWord)

	return predictionProbabilitiesList, headerList
			

def createWordLists():

	nounList = []
	verbList = []
	prepositionList = []
	
	LRPdataFolder = "LRPdata/"
	
	nounListFileName = LRPdataFolder + "wordlistNoun.txt"
	verbListFileName = LRPdataFolder + "wordlistVerb.txt"
	prepositionListFileName = LRPdataFolder + "wordlistPreposition.txt"
	
	fileContents = open(nounListFileName, "r")
	nounList = fileContents.read().split('\n')
	fileContents = open(verbListFileName, "r")
	verbList = fileContents.read().split('\n')
	fileContents = open(prepositionListFileName, "r")
	prepositionList = fileContents.read().split('\n')
		
	return nounList, verbList, prepositionList

def createLemmaLists():

	#dict ensures no redundancy (vs list)
	nounDict = {}	
	verbDict = {}
	
	for synset in list(wn.all_synsets(wn.NOUN)):
		#print("synset = ", synset.name())
		for lemma in synset.lemmas():
			print("lemma = ", lemma)
			nounDict[lemma] = lemma
			#nounList.append(lemma)
	for synset in list(wn.all_synsets(wn.VERB)):
		#print("synset = ", synset.name())
		for lemma in synset.lemmas():
			print("lemma = ", lemma)
			verbDict[lemma] = lemma
			#verbList.append(lemma)
		
	nounList = list(nounDict.keys())
	verbList = list(verbDict.keys())
	
	return nounList, verbList


def addTuplesToSemanticNet(predictionProbabilities, subjectText, relationshipText, objectText):

	minPredictionProbability = 0.1	#min probability required to add connection to knowledge graph

	for maskedWordIndex, predictionProbability in enumerate(predictionProbabilities):
	
		if(predictionProbability > minPredictionProbability):

			#add the tuple to the semantic net

			predictedWord = tok.convert_ids_to_tokens(maskedWordIndex)

			if(predictionIndex == predictionIndexSubject):
				tupleTriplet = (predictedWord, relationshipText, objectText)
			if(predictionIndex == predictionIndexRelationship):
				tupleTriplet = (subjectText, predictedWord, objectText)
			if(predictionIndex == predictionIndexObject):
				tupleTriplet = (subjectText, relationshipText, predictedWord)

			tupleTriplets.append(tupleTriplet)


if __name__ == "__main__":

	mainFunction()
