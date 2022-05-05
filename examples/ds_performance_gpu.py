import time
import torch

start_time = time.time()

# The following two lines are only needed because of this repository organization
import sys, os

sys.path.insert(1, os.path.realpath(os.path.pardir))
from torchhd import structures, functional
import string

device = torch.device("cuda:2")

# Setup
DIMENSIONS = 10000
itemMemory = structures.Memory()
letters = list(string.ascii_lowercase)
letters_hv = functional.random_hv(len(letters), DIMENSIONS, device=device)
list(map(lambda l: itemMemory.add(l[0], l[1]), zip(letters_hv, letters)))


# SETS
exampleSet = ["a", "b", "c", "d", "e"]
hdSet = structures.Multiset(DIMENSIONS, device=device)
list(map(lambda l: hdSet.add(letters_hv[letters.index(l)]), exampleSet))
similarity = functional.cosine_similarity(hdSet.value, letters_hv)


# SEQUENCE
# Bundling based
exampleSet = ["a", "b", "c", "d", "e"]
hdSequence = structures.Sequence(DIMENSIONS, device=device)
list(map(lambda l: hdSequence.append(letters_hv[letters.index(l)]), exampleSet))

last_value = hdSequence[4]
similarity = functional.cosine_similarity(last_value, letters_hv)

# Replacing "e" to "z" at 5th position in  the  sequence
hdSequence.replace(4, letters_hv[letters.index("e")], letters_hv[letters.index("z")])

exampleSetZ = ["a", "b", "c", "d", "z"]
hdSequenceZ = structures.Sequence(DIMENSIONS, device=device)
list(map(lambda l: hdSequenceZ.append(letters_hv[letters.index(l)]), exampleSetZ))

similarity = functional.cosine_similarity(
    hdSequenceZ.value, hdSequence.value.unsqueeze(0)
)


# SHIFT AND CONCATENATE
hdSequence = structures.Sequence(DIMENSIONS, device=device)
list(map(lambda l: hdSequence.append(letters_hv[letters.index(l)]), exampleSet))

concatSet = ["x", "y", "z"]
hdSequenceConcat = structures.Sequence(DIMENSIONS, device=device)
list(map(lambda l: hdSequenceConcat.append(letters_hv[letters.index(l)]), concatSet))

hdSequence.concat(hdSequenceConcat)

similarity = functional.cosine_similarity(letters_hv, hdSequence.value)


# BINDING BASED REPRESENTATION
exSeq = ["a", "b", "c", "d", "e"]  # sequence to represent
hdSequenceOne = structures.DistinctSequence(DIMENSIONS, device=device)
list(map(lambda l: hdSequenceOne.append(letters_hv[letters.index(l)]), exSeq))

exSeqTwo = ["a", "b", "c", "d", "d"]  # sequence to represent
hdSequenceTwo = structures.DistinctSequence(DIMENSIONS, device=device)
list(map(lambda l: hdSequenceTwo.append(letters_hv[letters.index(l)]), exSeqTwo))

similarity = functional.cosine_similarity(
    hdSequenceOne.value, hdSequenceTwo.value.unsqueeze(0)
)

# Replacing "e" to "z" at 5th position in  the  sequence
hdSequenceOne.replace(4, letters_hv[letters.index("e")], letters_hv[letters.index("z")])

exSeqScratch = ["a", "b", "c", "d", "z"]  # sequence to represent
hdSequenceScratch = structures.DistinctSequence(DIMENSIONS, device=device)
list(
    map(lambda l: hdSequenceScratch.append(letters_hv[letters.index(l)]), exSeqScratch)
)

similarity = functional.cosine_similarity(
    hdSequenceOne.value, hdSequenceScratch.value.unsqueeze(0)
)


# TUPLES
exSetOne = ["a", "b", "c", "d", "e"]  # sequence to represent
exSetTwo = ["x", "y", "z"]  # sequence to represent

hdTensorsOne = torch.stack(list(map(lambda l: letters_hv[letters.index(l)], exSetOne)))
hdTensorsTwo = torch.stack(list(map(lambda l: letters_hv[letters.index(l)], exSetTwo)))

hdSetOne = functional.multiset(hdTensorsOne)
hdSetTwo = functional.multiset(hdTensorsTwo)

hdTuples = functional.bind(hdSetOne, functional.permute(hdSetTwo, shifts=1))

axSequence = structures.DistinctSequence(DIMENSIONS, device=device)
axSequence.append(letters_hv[letters.index("a")])
axSequence.append(letters_hv[letters.index("x")])

similarity = functional.cosine_similarity(axSequence.value, hdTuples.unsqueeze(0))

asSequence = structures.DistinctSequence(DIMENSIONS, device=device)
asSequence.append(letters_hv[letters.index("a")])
asSequence.append(letters_hv[letters.index("s")])
similarity = functional.cosine_similarity(asSequence.value, hdTuples.unsqueeze(0))


# UNDIRECTED GRAPH
edges = [("a", "b"), ("a", "e"), ("c", "b"), ("d", "c"), ("e", "d")]
graph = structures.Graph(DIMENSIONS, device=device, directed=False)
list(
    map(
        lambda l: graph.add_edge(
            letters_hv[letters.index(l[0])], letters_hv[letters.index(l[1])]
        ),
        edges,
    )
)

Aneighbours = graph.node_neighbors(letters_hv[letters.index("a")])

similarity = functional.cosine_similarity(Aneighbours, letters_hv)

# DIRECTED GRAPH

edges = [("a", "b"), ("a", "e"), ("c", "b"), ("d", "c"), ("e", "d")]
graph = structures.Graph(
    DIMENSIONS, device=device,
    directed=True,
)
list(
    map(
        lambda l: graph.add_edge(
            letters_hv[letters.index(l[0])], letters_hv[letters.index(l[1])]
        ),
        edges,
    )
)

# Outgoing a
Aneighbours = graph.node_neighbors(letters_hv[letters.index("a")], outgoing=True)

similarity = functional.cosine_similarity(Aneighbours, letters_hv)

# Incoming b
Bneighbours = graph.node_neighbors(letters_hv[letters.index("b")], outgoing=False)
similarity = functional.cosine_similarity(Bneighbours, letters_hv)

# BINARY TREE

tree_list = [
    ["a", ["l", "l", "l"]],
    ["b", ["l", "r", "l"]],
    ["c", ["r", "r", "l"]],
    ["d", ["r", "r", "r", "l"]],
    ["e", ["r", "r", "r", "r"]],
    ["f", ["l", "r", "r", "l", "l"]],
    ["g", ["l", "r", "r", "l", "r"]],
]

tree = structures.Tree(DIMENSIONS, device=device)
list(map(lambda l: tree.add_leaf(letters_hv[letters.index(l[0])], l[1]), tree_list))
d_value = tree.get_leaf(tree_list[3][1])
similarity = functional.cosine_similarity(d_value, letters_hv)

# FREQUENCY

exMul = ["a", "a", "a", "b", "b", "c"]
hd_freq = structures.Multiset(DIMENSIONS, device=device)
list(map(lambda l: hd_freq.add(letters_hv[letters.index(l)]), exMul))

similarity = functional.dot_similarity(hd_freq.value, letters_hv)

# NGRAM

data = list("helloworld")
n = 3
hv_data = torch.stack(list(map(lambda l: letters_hv[letters.index(l)], data)))
hd_gram1 = functional.ngrams(hv_data, n)

data = list("felloworld")
hv_data = torch.stack(list(map(lambda l: letters_hv[letters.index(l)], data)))
hd_gram2 = functional.ngrams(hv_data, n)

similarity = functional.cosine_similarity(hd_gram1, hd_gram2.unsqueeze(0))

data = list("hejvarlden")
hv_data = torch.stack(list(map(lambda l: letters_hv[letters.index(l)], data)))
hd_gram3 = functional.ngrams(hv_data, n)

similarity = functional.cosine_similarity(hd_gram1, hd_gram3.unsqueeze(0))

data = list("ell")
hv_data = torch.stack(list(map(lambda l: letters_hv[letters.index(l)], data)))
hd_gram4 = functional.ngrams(hv_data, n)

similarity = functional.cosine_similarity(hd_gram1, hd_gram4.unsqueeze(0))

data = list("abc")
hv_data = torch.stack(list(map(lambda l: letters_hv[letters.index(l)], data)))
hd_gram5 = functional.ngrams(hv_data, n)

similarity = functional.cosine_similarity(hd_gram1, hd_gram5.unsqueeze(0))

# STACK

exStack = ["b", "c", "d"]
hdStack = structures.Sequence(DIMENSIONS, device=device)
list(map(lambda l: hdStack.append(letters_hv[letters.index(l)]), exStack))
similarity = functional.cosine_similarity(hdStack.value, letters_hv)

hdStack.appendleft(letters_hv[letters.index("a")])
similarity = functional.cosine_similarity(hdStack.value, letters_hv)

# Lookup
first_element = torch.argmax(functional.cosine_similarity(hdStack[0], letters_hv))
hdStack.popleft(letters_hv[first_element])

exStack = ["b", "c", "d"]
hdStackTwo = structures.Sequence(DIMENSIONS, device=device)
list(map(lambda l: hdStackTwo.append(letters_hv[letters.index(l)]), exStack))

similarity = functional.cosine_similarity(hdStackTwo.value, hdStack.value.unsqueeze(0))

# FSA

states = ["L", "U"]
tokens = ["P", "T"]
states_hv = functional.random_hv(len(states), DIMENSIONS, device=device)
tokens_hv = functional.random_hv(len(tokens), DIMENSIONS, device=device)

transitions = [["L", "L", "P"], ["L", "U", "T"], ["U", "U", "T"], ["U", "L", "P"]]

fsa = structures.FiniteStateAutomata(DIMENSIONS, device=device)
list(
    map(
        lambda l: fsa.add_transition(
            tokens_hv[tokens.index(l[2])],
            states_hv[states.index(l[0])],
            states_hv[states.index(l[1])],
        ),
        transitions,
    )
)

hd_approx = fsa.change_state(tokens_hv[tokens.index("P")], states_hv[states.index("L")])
next_state = torch.argmax(functional.cosine_similarity(hd_approx, states_hv))

hd_approx = fsa.change_state(tokens_hv[tokens.index("T")], states_hv[states.index("L")])
next_state = torch.argmax(functional.cosine_similarity(hd_approx, states_hv))

print("Duration", time.time() - start_time)
