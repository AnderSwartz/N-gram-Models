import collections
import numpy as np
import string

class Unigram(object):
    """Barebones example of a language model class."""

    def __init__(self):
        self.counts = collections.Counter()
        self.total_count = 0
        
    def train(self, filename):
        """Train the model on a text file."""
        for line in open(filename):
            for w in line.rstrip('\n'):
                self.counts[w] += 1
                self.total_count += 1

    def start(self):
        """Reset the state to the initial state."""
        self.state = ""
        pass

    def read(self, w):
        """Read in character w, updating the state."""
        self.state = w
        pass

    def prob(self, w):
        """Return the probability of the next character being w given the
        current state."""
        return self.counts[w] / self.total_count

    def predict(self):
        listOfChars = [char for char in string.printable]
        listOfChars.append(" ")
        scores = np.zeros(len(listOfChars))
        scoresIndex = 0
        for letter in listOfChars:
            scores[scoresIndex] = self.prob(letter)
            scoresIndex+=1
        return listOfChars[np.argmax(scores)]

("Running unigram...")

unigram = Unigram()
unigram.train("english/train")
# unigram.train("english\dev")

#testing:
print("Assessing accuracy on development set...")
correct = 0
totalCharsInTest = 0
for line in open("english\dev"):
    for w in line.rstrip('\n'):
        totalCharsInTest+=1
        if(w==unigram.predict()):
            correct+=1
        unigram.read(w) #doesn't make a difference for unigram
print("percentage of chars correctly predicted:",correct/totalCharsInTest)