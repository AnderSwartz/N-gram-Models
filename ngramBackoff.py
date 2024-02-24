import collections
import numpy as np
import string

class Unigram(object):
    """Barebones example of a language model class."""

    def __init__(self,n):
        self.counts = collections.Counter()
        self.total_count = 0
        self.n = n
        self.state = ""

    def train(self, filename):
        """Train the model on a text file."""
        for line in open(filename):
            line = line.rstrip('\n')
            for index in range(0,len(line)):
                window = line[index:index+self.n]
                self.counts[window] += 1
                self.total_count += 1

    def start(self):
        """Reset the state to the initial state."""
        self.state = " " * (self.n - 1)
        pass
   
    def read(self, w):
        """Read in character w, updating the state."""
        self.state = self.state[1:]+w
        pass

    def prob(self, w,n):
        """Return the probability of the next character being w given the
        current state."""
        listOfChars = [char for char in string.printable]
        listOfChars.append(" ")
        denom = 0.0000001
        if(n>1):
            for letter in listOfChars:
                denom+=self.counts[self.state[self.n-n:]+letter] 
            return self.counts[self.state[self.n-n:]+w] / denom
        # print("using unigram")
        return self.counts[w] / self.total_count

    def predict(self):
        listOfChars = [char for char in string.printable]
        listOfChars.append(" ")
        scores = np.zeros(len(listOfChars))
        n=self.n
        while(np.sum(scores)==0):
            scoresIndex = 0
            for letter in listOfChars:
                    # start with n gram, backoff if necessary
                scores[scoresIndex] = self.prob(letter,n)
                # print("could not find ngram", self.state[self.n-n:]+letter,"backing off to n=",n-1)
                scoresIndex+=1
            n-=1
        return listOfChars[np.argmax(scores)]

def testUnigramOnFile(fileToTest):
    correct = 0
    totalCharsInTest = 0
    mistakes = collections.Counter()
    for line in open(fileToTest):
        for w in line.rstrip('\n'):
            totalCharsInTest+=1
            if(w==unigram.predict()):
                correct+=1
            else:
                mistakes[w]+=1
            unigram.read(w)
    # print(mistakes)        
    print("percentage of chars correctly predicted:",correct/totalCharsInTest)

print("Running ngramV(with backoff)...")
unigram = Unigram(5)
unigram.start()
print("Training...")
unigram.train("english/train")
print("Testing on test set...")
testUnigramOnFile("english/test")
# print("Testing on dev set...")
# testUnigramOnFile("english/dev")