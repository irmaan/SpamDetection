import math
import csv
from random import shuffle



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


def searchInVocabulary(key):
    if key in vocabulary:
        return vocabulary.index(key)


def calculatePrior(className):
    count=0
    for i in range(len(trainingData)):
        if trainingData[i][0]==className:
            count+=1
    return count/len(trainingData)


def calculateNumberOfWordsInClass(className):
    count=0
    for i in range(len(trainingData)):
        if trainingData[i][0] == className:
            count+=len(trainingData[i][1])
    return  count


def calculateProbsOfWordsOfVocab(vocab,numberOfWords):
    for j in range(len(vocab[1])):
            vocab[3].append((vocab[1][j]+1)/(numberOfWords+len(vocabulary)))

    return vocab


def calculateWordProb(word, className):
    if className == "spam":
        if word in vocabForSpam[0]:
            return vocabForSpam[3][vocabForSpam[0].index(word)]
        else:  # Laplace estimation to prevent probability of 0
            return 1/(numberOfWordsOfSpam+len(vocabulary))

    elif className == "ham":
        if word in vocabForHam[0]:
            return vocabForHam[3][vocabForHam[0].index(word)]
        else:  # Laplace estimation to prevent probability of 0
            return 1/(numberOfWordsOfHam+len(vocabulary))



def testForPredict(testData):
    global truePositive,trueNegative,falsePositive,falseNegative
    count=0
    sumOfProbHam=0
    sumOfProbSpam=0
    for i in range(len(testData)):
        sumOfProbSpam=1
        sumOfProbHam=1
        for j in range(len(testData[i][1])):
                sumOfProbHam*=calculateWordProb(testData[i][1][j], "ham")
                sumOfProbSpam *= calculateWordProb(testData[i][1][j],"spam")

        spamProb = spamPriorProb* sumOfProbSpam
        hamProb  = hamPriorProb* sumOfProbHam
        if spamProb>hamProb:
            result="spam"
        else:
            result="ham"

        if  testData[i][0] == result:
            count+=1
            if result=="ham":
                truePositive+=1
            else:
                trueNegative+=1
        else:
            if result=="ham":
                falsePositive+=1
            else:
                falseNegative+=1

    return  (count/len(testData)) *100


loadInputFiles()
stopWords=removeSpaceFromStopWords(stopWords)

trainingData = prepareDataFormat(trainingData)
testData = prepareDataFormat(testData)

trainingData = removeStopWords(trainingData)
testData = removeStopWords(testData)

vocabulary = generateVocabulary(trainingData)

vocabForSpam=generateVocabularyForClass(trainingData,"spam")
vocabForHam=generateVocabularyForClass(trainingData,"ham")

spamPriorProb=calculatePrior("spam")
hamPriorProb=calculatePrior("ham")

numberOfWordsOfSpam=calculateNumberOfWordsInClass("spam") #n for spam
numberOfWordsOfHam=calculateNumberOfWordsInClass("ham")   #n  for ham

vocabForHam = calculateProbsOfWordsOfVocab(vocabForHam,numberOfWordsOfHam)
vocabForSpam = calculateProbsOfWordsOfVocab(vocabForSpam,numberOfWordsOfSpam)

accuracy= testForPredict(testData)
print("Accuracy is :   " + str(round(accuracy,2)) + "%")

print("True Positive : " + str(truePositive))
print("True Negative : " + str(trueNegative))
print("False Positive : " + str(falsePositive))
print("False Negative : " + str(falseNegative))








