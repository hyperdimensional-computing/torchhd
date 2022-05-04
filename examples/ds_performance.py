import time

import torch
import matplotlib.pyplot as plt

start_time = time.time()

# The following two lines are only needed because of this repository organization
import sys, os
sys.path.insert(1, os.path.realpath(os.path.pardir))
from torchhd import structures, functional
import string

# Setup

DIMENSIONS = 10000
itemMemory = structures.Memory()
letters = list(string.ascii_lowercase)
letters_hv = functional.random_hv(len(letters), DIMENSIONS)
list(map(lambda l: itemMemory.add(l[0], l[1]), zip(letters_hv, letters)))

# SETS

exampleSet = ['a', 'b', 'c', 'd', 'e']
hdSet = structures.Multiset(DIMENSIONS)
list(map(lambda l: hdSet.add(letters_hv[letters.index(l)]), exampleSet))
similarity = functional.cosine_similarity(hdSet.value, letters_hv)

'''
plt.rcParams.update({'font.size': 22})
plt.figure(figsize=(20,10))
plt.stem(similarity, use_line_collection=True)
plt.xticks([i for i in range(len(letters))], letters)
plt.grid()
print("Simality of the set to the codebook:")
plt.show()
'''
# SEQUENCE

hdSequence = structures.Sequence(DIMENSIONS)
list(map(lambda l: hdSequence.append(letters_hv[letters.index(l)]), exampleSet))
# Pop last value

last_value = hdSequence[4]
similarity = functional.cosine_similarity(last_value, letters_hv)

hdSequence.pop(hdSequence[4])
hdSequence.append(letters_hv[letters.index('z')])

exampleSetZ = ['a', 'b', 'c', 'd', 'z']
hdSequenceZ = structures.Sequence(DIMENSIONS)
list(map(lambda l: hdSequenceZ.append(letters_hv[letters.index(l)]), exampleSetZ))

similarity = functional.cosine_similarity(hdSequenceZ.value, hdSequence.value.unsqueeze(0))

# SHIFT AND CONCATENATE
hdSequence = structures.Sequence(DIMENSIONS)
list(map(lambda l: hdSequence.append(letters_hv[letters.index(l)]), exampleSet))

concatSet = ['x', 'y', 'z']
hdSequenceConcat = structures.Sequence(DIMENSIONS)
list(map(lambda l: hdSequenceConcat.append(letters_hv[letters.index(l)]), concatSet))

hdSequence.concat(hdSequenceConcat)

similarity = functional.cosine_similarity(letters_hv, hdSequence.value)

# BINDING BASED REPRESENTATION
exSeq = ['a', 'b', 'c', 'd', 'e']  # sequence to represent
hdSequenceOne = structures.Sequence(DIMENSIONS)
list(map(lambda l: hdSequenceOne.append(letters_hv[letters.index(l)]), exSeq))

exSeqTwo = ['a', 'b', 'c', 'd', 'd']  # sequence to represent
hdSequenceTwo = structures.Sequence(DIMENSIONS)
list(map(lambda l: hdSequenceTwo.append(letters_hv[letters.index(l)]), exSeqTwo))

similarity = functional.cosine_similarity(hdSequenceOne.value, hdSequenceTwo.value.unsqueeze(0))

'''

# TUPLES
exSetOne = ['a', 'b', 'c', 'd', 'e']  # sequence to represent
exSetTwo = ['x', 'y', 'z']  # sequence to represent

hdTensorsOne = torch.stack(list(map(lambda l: letters_hv[letters.index(l)], exSetOne)))
hdTensorsTwo = torch.stack(list(map(lambda l: letters_hv[letters.index(l)], exSetTwo)))

hdSetOne = functional.multiset(hdTensorsOne)
hdSetTwo = functional.multiset(hdTensorsTwo)

hdTuples = functional.bind(hdSetOne, functional.permute(hdSetTwo, shifts=1))

axSequence = structures.Sequence(DIMENSIONS)
axSequence.append(letters_hv[letters.index('a')])
axSequence.append(letters_hv[letters.index('x')])

similarity = functional.cosine_similarity(axSequence.value, hdTuples.unsqueeze(0))

asSequence = structures.Sequence(DIMENSIONS)
asSequence.append(letters_hv[letters.index('a')])
asSequence.append(letters_hv[letters.index('s')])
similarity = functional.cosine_similarity(asSequence.value, hdTuples.unsqueeze(0))

'''

# UNDIRECTED GRAPH

edges = [('a', 'b'), ('a', 'e'), ('c', 'b'), ('d', 'c'), ('e', 'd')]
graph = structures.Graph(DIMENSIONS, directed=False)
list(map(lambda l: graph.add_edge(letters_hv[letters.index(l[0])], letters_hv[letters.index(l[1])]), edges))

Aneighbours = graph.node_neighbours(letters_hv[letters.index('a')])

similarity = functional.cosine_similarity(Aneighbours, letters_hv)

# DIRECTED GRAPH

edges = [('a', 'b'), ('a', 'e'), ('c', 'b'), ('d', 'c'), ('e', 'd')]
graph = structures.Graph(DIMENSIONS, directed=True)
list(map(lambda l: graph.add_edge(letters_hv[letters.index(l[0])], letters_hv[letters.index(l[1])]), edges))

# Outgoing a
Aneighbours = graph.node_neighbours(letters_hv[letters.index('a')], outgoing=True)

similarity = functional.cosine_similarity(Aneighbours, letters_hv)

# Incoming b
Bneighbours = graph.node_neighbours(letters_hv[letters.index('b')], outgoing=False)
similarity = functional.cosine_similarity(Bneighbours, letters_hv)

