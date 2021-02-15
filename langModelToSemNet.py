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
pip install pattern

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

from pattern.en import conjugate, lemma, lexeme
import pattern.en as pattern

tok = AutoTokenizer.from_pretrained("bert-base-cased")
bert = BertForMaskedLM.from_pretrained("bert-base-cased")



testHarness = True
saveTopPredictionsOnly = True

saveMaskProbabilities = True
printInputText = True
addMaskProbabilitiesToSemanticNet = False	#not yet coded
topPredictionsNumber = 1000

feedMultipleInstanceRelationshipVerbForms = True
feedMultipleInstanceSubjectObjectDeterminerForms = True
feedMultipleInstanceRelationshipPrepositionAuxiliaryForms = True
feedExplicitConceptsForms = True
if(feedMultipleInstanceRelationshipVerbForms):
	instanceRelationshipVerbTenseFormList = [pattern.PRESENT, pattern.PAST]	#rides, rode, [NOT: will ride] 
else:
	instanceRelationshipVerbTenseFormList = [pattern.PRESENT]	#eg. the dog eats the pie
if(feedMultipleInstanceSubjectObjectDeterminerForms):
	instanceSubjectObjectDeterminerList = ["the", "a"]
else:
	instanceSubjectObjectDeterminerList = ["the"]	
instanceRelationshipPrepositionAuxiliary = "be"
if(feedMultipleInstanceRelationshipPrepositionAuxiliaryForms):
	instanceRelationshipPrepositionAuxiliaryFormList = [pattern.PRESENT, pattern.PAST]	#is, was, [NOT: will be] 
else:
	instanceRelationshipPrepositionAuxiliaryFormList = [pattern.PRESENT]	#"is"

if(feedExplicitConceptsForms):
	conceptRelationshipVerbTenseForm = pattern.INFINITIVE	#dogs eat pies
	conceptSubjectObjectNumberForm = pattern.PLURAL	#dogs eat pies


if(saveMaskProbabilities):
	saveInputTextSummary = True
	firstLineHeader = True
	maskProbabilitiesTextFileName = "maskProbabilitiesTextFile.txt"
	maskProbabilitiesTextFileDelimiter = ","	#csv format
	maskProbabilitiesTextFile = open(maskProbabilitiesTextFileName, "w")

predictionIndexSubject = "subject"
predictionIndexRelationship = "relationship"
predictionIndexObject = "object"
numberOfPredictionIndices = 3	#number of masks #subject, relationship, object


def mainFunction():

	#constructs tuples from noun/verb dictionary and finds probability of their associations using pretrained bert

	
	nounList, verbList, prepositionList = createWordLists()	#createLemmaLists()


	if(testHarness):
		subjectList = ["dog"]
		verbList = ["eat"]
		objectList = nounList
		predictionIndices = [predictionIndexObject]
	else:
		subjectList = nounList
		objectList = nounList
		predictionIndices = [predictionIndexSubject, predictionIndexRelationship, predictionIndexObject]

	tupleTriplets = []
	
	for predictionIndex in predictionIndices:
		
		subjectListPrediction = subjectList
		verbListPrediction = verbList
		prepositionListPrediction = prepositionList
		objectListPrediction = objectList
		
		if(predictionIndex == predictionIndexSubject):
			subjectListPrediction = ["NA"]
		elif(predictionIndex == predictionIndexRelationship):
			verbListPrediction = ["NA"]
			prepositionListPrediction = ["NA"]
		elif(predictionIndex == predictionIndexObject):
			objectListPrediction = ["NA"]
			
		if(testHarness):
			prepositionListPrediction = []
		
		for i1, w1 in enumerate(subjectListPrediction):	#subject
			for i3, w3 in enumerate(objectListPrediction):	#object

				w1morph = w1
				w3morph = w3
									
				#explicit instances;
				for i2, w2 in enumerate(verbListPrediction):	#relationship
					w2morph = w2
					for instanceSubjectObjectDeterminer in instanceSubjectObjectDeterminerList:
						determinerText = instanceSubjectObjectDeterminer
						for verbForm in instanceRelationshipVerbTenseFormList:
							w2morph = convertVerbForm(w2, verbForm, subjectNumber=pattern.SINGULAR)
							getMaskPossibilities(w1, w2, w3, w1morph, w2morph, w3morph, predictionIndex, explicitConcept=False, addDeterminers=True, determinerText=determinerText)
				for i2, w2 in enumerate(prepositionListPrediction):	#relationship
					w2morph = w2
					for instanceSubjectObjectDeterminer in instanceSubjectObjectDeterminerList:
						determinerText = instanceSubjectObjectDeterminer
						for instanceRelationshipPrepositionAuxiliaryForm in instanceRelationshipPrepositionAuxiliaryFormList:
							auxiliaryText = convertVerbForm(instanceRelationshipPrepositionAuxiliary, instanceRelationshipPrepositionAuxiliaryForm, subjectNumber=pattern.SINGULAR)
							getMaskPossibilities(w1, w2, w3, w1morph, w2morph, w3morph, predictionIndex, explicitConcept=False, addDeterminers=True, determinerText=determinerText, addAuxiliary=True, auxiliaryText=auxiliaryText)	
				
				#explicit concepts;		
				if(feedExplicitConceptsForms):
					for i2, w2 in enumerate(verbListPrediction):	#relationship
						w2morph = w2
						if(conceptRelationshipVerbTenseForm == pattern.INFINITIVE):
							w2morph = convertVerbForm(w2, pattern.INFINITIVE, subjectNumber=pattern.SINGULAR)
						if(conceptSubjectObjectNumberForm == pattern.PLURAL):
							w1morph = convertNounPlural(w1)
							w3morph =  convertNounPlural(w3)
						getMaskPossibilities(w1, w2, w3, w1morph, w2morph, w3morph, predictionIndex, explicitConcept=True)							

def convertVerbForm(verbInfinitive, verbForm, subjectNumber):	#pattern.SG
	wordMorp = (conjugate(verb=verbInfinitive,tense=verbForm,number=subjectNumber))
	return wordMorp

def convertNounPlural(nounSingular):
	nounMorp = pattern.pluralize(nounSingular)
	return nounMorp

	
def getMaskPossibilities(w1, w2, w3, w1morph, w2morph, w3morph, predictionIndex, explicitConcept, addDeterminers=False, determinerText=None, addAuxiliary=False, auxiliaryText=None):

	global firstLineHeader
	
	#if(i3 != i1):	#optional (do not allow relationships with identical subject and object)

	subjectText = w1morph
	relationshipText = w2morph
	objectText = w3morph

	startTextToken = nltk.word_tokenize(subjectText)
	relationshipTextToken = nltk.word_tokenize(relationshipText)
	objectTextToken = nltk.word_tokenize(objectText)

	if((len(startTextToken) == 1) and (len(relationshipTextToken) == 1) and (len(objectTextToken) == 1)):	#currently only allow single-word subject/relationship/objects

		subjectTextContextual = subjectText
		relationshipTextContextual = relationshipText
		objectTextContextual = objectText
		
		if(addDeterminers):
			subjectTextContextual = determinerText + " " + subjectText
			objectTextContextual = determinerText + " " + objectText
		if(addAuxiliary):
			relationshipTextContextual = auxiliaryText + " " + relationshipText

		startTextContextualTokens = nltk.word_tokenize(subjectTextContextual)
		relationshipTextContextualTokens = nltk.word_tokenize(relationshipTextContextual)
		objectTextContextualTokens = nltk.word_tokenize(objectTextContextual)

		if(predictionIndex == predictionIndexSubject):
			endText = relationshipTextContextual + " " + objectTextContextual
			inputText = f"{tok.mask_token} {endText}."
			maskIndex = 0
			#subjectText = tok.mask_token
			w1 = tok.mask_token
		elif(predictionIndex == predictionIndexRelationship):
			startText = subjectTextContextual
			endText = objectTextContextual
			inputText = f"{startText} {tok.mask_token} {endText}."
			maskIndex = len(startTextContextualTokens)
			#relationshipText = tok.mask_token
			w2 = tok.mask_token
		elif(predictionIndex == predictionIndexObject):
			startText = subjectTextContextual + " " + relationshipTextContextual
			inputText = f"{startText} {tok.mask_token}."
			maskIndex = len(startTextContextualTokens)
			#objectText = tok.mask_token
			w3 = tok.mask_token

		if(printInputText):
			print("inputText = ", inputText)

		predictionProbabilities, headerList = getProbabilitiesOfMaskedWord(inputText, maskIndex, firstLineHeader, saveTopPredictionsOnly, topPredictionsNumber)

		if(addMaskProbabilitiesToSemanticNet):
			#addTuplesToSemanticNet(predictionProbabilities, subjectText, relationshipText, objectText)
			addTuplesToSemanticNet(predictionProbabilities, w1, w2, w3)	#semantic dependency relations are defined by lemmas/infinitive

		#inputTextSummary = inputText.replace(" ", "_")
		#inputTextSummary = "(" + subjectText + " " + relationshipText + " " + objectText + ")"
		inputTextSummary = "(" + w1 + " " + w2 + " " + w3 + ")"	#semantic dependency relations are defined by lemmas/infinitive
		if(printInputText):
			print("inputTextSummary = ", inputTextSummary)
		
		if(saveMaskProbabilities):
			addMaskProbabilitiesToFile(maskProbabilitiesTextFile, predictionProbabilities, maskProbabilitiesTextFileDelimiter, firstLineHeader, headerList, saveInputTextSummary, inputTextSummary)

		if(firstLineHeader):
			if(not saveTopPredictionsOnly):
				firstLineHeader = False

								
def addMaskProbabilitiesToFile(maskProbabilitiesTextFile, predictionProbabilities, maskProbabilitiesTextFileDelimiter, firstLineHeader, headerList, saveInputTextSummary, inputTextSummary):

	if(firstLineHeader):
		predictionProbabilitiesTextFileLine = ""
		if(saveInputTextSummary):
			predictionProbabilitiesTextFileLine = predictionProbabilitiesTextFileLine + inputTextSummary + maskProbabilitiesTextFileDelimiter
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
