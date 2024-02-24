import collections
from re import S
import numpy as np
import string
from collections import defaultdict

class Ngram(object):
    """Barebones example of a language model class."""

    def __init__(self,n,lSmoothing):
        self.charMap = defaultdict(list)
        self.counts = collections.Counter()
        self.total_count = 0
        self.n = n
        self.state = ""
       
    def train(self, filename):
        """Train the model on a text file."""
        for line in open(filename,encoding="utf8"):
            line = line.rstrip('\n')
            for index in range(0,len(line)):
                for n in range(0,self.n):
                    window = line[index+n:index+self.n]
                    self.counts[window] += 1
                    self.total_count += 1

    def start(self):
        """Reset the state to the initial state."""
        self.state = " " * (self.n - 1)
        """Reset the state to the initial state."""
        pass

    def read(self, w):
        """Read in character w, updating the state."""
        self.state = self.state[1:]+w
        pass

    def prob(self, w,canidates,n):
        """Return the probability of the next character being w given the
        current state."""
        denom = 0.0000001
        if(n>1):
            for char in canidates:
                denom+=self.counts[self.state[self.n-n:]+char] 
            return (self.counts[self.state[self.n-n:]+w]+10)  / (denom+10)
        return ( self.counts[w]+1) / (self.total_count+1)

    def predict(self,token):
        canidates = self.canidates(token)
        scores = np.zeros(len(canidates))
        n=self.n
        scoresIndex = 0
        for char in canidates:
            if(char == " " and token!="<space>"):
                scores[scoresIndex] = self.prob(char,canidates,n) -1
            else:
                scores[scoresIndex] = self.prob(char,canidates,n)
            scoresIndex+=1
        winners = np.argwhere(scores == np.amax(scores))
        bestScore = np.argsort(scores, axis=0)[-1]
        secondBestScore = np.argsort(scores, axis=0)[-2]
        if(len(winners)>1 or scores[bestScore] - scores[secondBestScore]<.01) :
            scoresIndex = 0
            for char in canidates:
                if(char == " " and token!="<space>"):
                    scores[scoresIndex] = self.prob(char,canidates,n-1) -1
                else:
                    scores[scoresIndex] = self.prob(char,canidates,n-1)
                scoresIndex+=1
        winners = np.argwhere(scores == np.amax(scores))
        bestScore = np.argsort(scores, axis=0)[-1]
        secondBestScore = np.argsort(scores, axis=0)[-2]
        if(len(winners)>1 or scores[bestScore] - scores[secondBestScore]<0):
            scoresIndex = 0
            for char in canidates:
                if(char == " " and token!="<space>"):
                    scores[scoresIndex] = self.prob(char,canidates,n-2) -1
                else:
                    scores[scoresIndex] = self.prob(char,canidates,n-2)
                scoresIndex+=1
        if(canidates[np.argmax(scores)]=="<space>"):
            return " "
        return canidates[np.argmax(scores)]

    def predictWithInterpolation(self,token):
        canidates = self.canidates(token)
        scores = np.zeros(len(canidates))
        n=self.n
        scoresIndex = 0
        for char in canidates:
            if(char == " " and token!="<space>"):
               scores[scoresIndex] = 0
            else:
                scores[scoresIndex] = self.prob(char,canidates,n) + self.prob(char,canidates,n-1) + self.prob(char,canidates,n-2)
            scoresIndex+=1

        if(canidates[np.argmax(scores)]=="<space>"):
            return " "
        return canidates[np.argmax(scores)]
        
    def canidates(self,token):
        canidates = self.charMap[token]
        canidates.append(token)
        canidates.append(" ")
        return canidates
        
    def generateCharMap(self, filename):
        for line in open(filename,encoding="utf8"):
            line = line.split()
            self.charMap[line[1]].append(line[0])

def testNgramOnFile(pinFile,hanFile):
    correct = 0
    totalCharsInTest = 0
    mistakes = collections.Counter()
    mistakenlyPredicted = collections.Counter()
    with open(pinFile,'r',encoding='utf-8') as file:
        with open(hanFile,'r',encoding='utf-8') as file2:
            for question, answer in zip(file,file2):
                question = question.rstrip('\n')
                answer = answer.rstrip('\n')
                for token, char in zip(question.split(),answer):
        
                    state = ngram.state
                    totalCharsInTest+=1
                    predicted = ngram.predict(token)
                    if(char==predicted):
                        correct+=1
                    else:
                        mistakes[char]+=1
                        mistakenlyPredicted[predicted]+=1
                    ngram.read(char)
            # print("MISSED:",mistakes)
            # print("MISTAKENLY PREDICETED", mistakenlyPredicted)
            print("percentage of chars correctly predicted:",correct/totalCharsInTest)


def testNgramOnFileWithInterpolation(pinFile,hanFile):
    correct = 0
    totalCharsInTest = 0
    mistakes = collections.Counter()
    mistakenlyPredicted = collections.Counter()
    with open(pinFile,'r',encoding='utf-8') as file:
        with open(hanFile,'r',encoding='utf-8') as file2:
            for question, answer in zip(file,file2):
                question = question.rstrip('\n')
                answer = answer.rstrip('\n')
                for token, char in zip(question.split(),answer):
                    state = ngram.state
                    totalCharsInTest+=1
                    predicted = ngram.predictWithInterpolation(token)
                    if(char==predicted):
                        correct+=1
                    else:
                        mistakes[char]+=1
                        mistakenlyPredicted[predicted]+=1
                    ngram.read(char)
            # print("MISSED:",mistakes)
            # print("MISTAKENLY PREDICETED", mistakenlyPredicted)
            print("percentage of chars correctly predicted:",correct/totalCharsInTest)

print("Running Chinese ngramV3")
ngram = Ngram(3,1)
ngram.start()
ngram.generateCharMap("chinese\charmap")
ngram.train("chinese/train.han")

testNgramOnFileWithInterpolation("chinese/test.pin","chinese/test.han")