import math
import csv
from random import shuffle
import numpy as np
from sklearn.naive_bayes import MultinomialNB


data=[]
trainingData=[]
testData=[]
stopWords=[]
punctuations=["’","'","[","]","(",")","{","}",":",",","،", "‒","–","—","!"
                ,"‹","›","«","»","‐","-","?","‘","’","“","”","'","\"",";", "/","\\"
                ,"&","*","@","×","~","÷","%","+","-" ,"=" ,"_",".",">","<"]
vocabulary=[[],[],[]]
vocabForSpam=[]
vocabForHam=[]

truePositive=0
trueNegative=0
falsePositive=0
falseNegative=0

def loadInputFiles():
    global trainingData,testData,stopWords
    datafile = open('sms_spam.csv', 'r',newline="\n")
    dataReader = csv.reader(datafile,delimiter=',')
    data = list(dataReader)
    shuffle(data)
    datafile.close()

    datafile = open('stopwords.csv', 'r',newline='\n')
    dataReader = csv.reader(datafile,delimiter=',')
    stopWords = list(dataReader)
    stopWords=stopWords[0]

    dataSize=len(data)
    trainingDataSize= math.ceil((80*dataSize)/100)
    trainingData=data[1:trainingDataSize]
    testData=data[trainingDataSize:]


def removeSpaceFromStopWords(stopWords):
    newStopWords=[]
    for word in stopWords:
        newStopWords.append(str(word).replace(" ",""))
    stopWords.clear()
    stopWords=newStopWords
    return stopWords



def removeStopWords(inputData):

    for i in range(len(inputData)):
        tempList=[]
        flag=True
        for j in  range(len(inputData[i][1])):
            flag = True
            for p in punctuations:
                inputData[i][1][j]=str(inputData[i][1][j]).replace(p,"")
            if inputData[i][1][j] in stopWords or str(inputData[i][1][j]).isdigit()==True or len(inputData[i][1][j]) <= 1:
                    flag=False
            if flag:
                tempList.append(inputData[i][1][j])
        inputData[i][1]=tempList

    return inputData

def generateVocabulary(inputData):
    vocabulary=[]
    for i in range(len(inputData)):
        for j in range(len(inputData[i][1])):
                if inputData[i][1][j] in vocabulary:
                    continue
                else:
                    vocabulary.append(inputData[i][1][j])
    return vocabulary



def generateVocabularyForClass(inputData,className):
    vocabForClass=[[],[],[],[]]
    for i in range(len(inputData)):
        for j in range(len(inputData[i][1])):
            if inputData[i][0]==className:
                if inputData[i][1][j] in vocabForClass[0]:
                    index=vocabForClass[0].index(inputData[i][1][j])
                    vocabForClass[1][index]+=1
                else:
                    vocabForClass[0].append(inputData[i][1][j])
                    vocabForClass[1].append(1)
                    vocabForClass[2].append(className)
    return  vocabForClass

def prepareDataFormat(inputData):

    for i in range(len(inputData)):
        inputData[i][1]=str(inputData[i][1]).split(" ")

    for i in range(len(inputData)):
        for j in range(len(inputData[i][1])):
            inputData[i][1][j] = str(inputData[i][1][j]).lower()
    return inputData


def makeProperInputForSciKit(input):
    totalBag=[]

    for i in range (len(input)):
        row=[]
        for word in vocabulary:
             row.append(0)
        totalBag.append(row)

    for i in range(len(input)):
        for j in range(len(input[i][1])):
            if input[i][1][j] in  vocabulary:
                totalBag[i][vocabulary.index(input[i][1][j])]+=1

    return totalBag

def makeProperTargets(inputData):
    targetVectors=[]
    for i in range(len(inputData)):
        if inputData[i][0]=="ham":
            targetVectors.append("H")
        else:
            targetVectors.append("S")
    return targetVectors


def calculateAccuracy(prediction,testTargets):
    count=0
    for i in range(len(testTargets)):
        if prediction[i]==testTargets[i]:
            count+=1
    return (count/len(testTargets) )*100



loadInputFiles()
stopWords=removeSpaceFromStopWords(stopWords)

trainingData = prepareDataFormat(trainingData)
testData = prepareDataFormat(testData)

trainingData = removeStopWords(trainingData)
testData = removeStopWords(testData)

vocabulary = generateVocabulary(trainingData)


properInput=makeProperInputForSciKit(trainingData)
trainDataForSciKit=np.asanyarray(properInput)
testDataForSciKit=makeProperInputForSciKit(testData)

trainTargets=makeProperTargets(trainingData)
trainTargetsForSciKit=np.asanyarray(trainTargets)

testTargets=makeProperTargets(testData)

classifier = MultinomialNB()
classifier.fit(trainDataForSciKit, trainTargetsForSciKit)

TEST=np.asanyarray(testDataForSciKit)

prediction=classifier.predict(TEST)

acc = calculateAccuracy(prediction,testTargets)
print(acc)









