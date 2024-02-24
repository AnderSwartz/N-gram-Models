import collections
import numpy as np
import string

class Ngram(object):
    """Barebones example of a language model class."""

    def __init__(self,n):
        self.counts = collections.Counter()
        self.total_count = 0
        self.n = n
        self.state = ""
        
    def train(self, filename):
        """Train the model on a text file."""
        
        for line in open(filename):
            # line = line.replace(" ","")
            line = line.rstrip('\n')
            for index in range(0,len(line)):
                window = line[index:index+self.n]
                # print(window)
                self.counts[window] += 1
                self.total_count += 1

    def start(self):
        self.state = " " * (self.n - 1)
        """Reset the state to the initial state."""
        pass
    
    def read(self, w):
        """Read in character w, updating the state."""
        #state is the previous chars
        self.state = self.state[1:]+w
        pass

    def prob(self, w):
        """Return the probability of the next character being w given the
        current state."""
        listOfChars = [char for char in string.printable]
        listOfChars.append(" ")

        denom = 0.0000001
        
        for letter in listOfChars:
            denom+=self.counts[self.state+letter] 
        return self.counts[self.state+w]  / denom 

    def predict(self):
        listOfChars = [char for char in string.printable]
        listOfChars.append(" ")
        scores = np.zeros(len(listOfChars))
        scoresIndex = 0
        for letter in listOfChars:
            scores[scoresIndex] = self.prob(letter)
            scoresIndex+=1
        return listOfChars[np.argmax(scores)]

def testUnigramOnFile(fileToTest):
    correct = 0
    totalCharsInTest = 0
    mistakes = collections.Counter()
    for line in open(fileToTest):
        for w in line.rstrip('\n'):
            totalCharsInTest+=1
            if(w==ngram.predict()):
                correct+=1
            else:
                mistakes[w]+=1
            ngram.read(w)
    print("percentage of chars correctly predicted:",correct/totalCharsInTest)

print("Running ngramV1...")
ngram = Ngram(5)
ngram.start()
print("Training...")
ngram.train("english/train")

print("Testing on dev set...")
testUnigramOnFile("english/dev")

# print("Assessing accuracy on test set...")
# testUnigramOnFile("english/test")