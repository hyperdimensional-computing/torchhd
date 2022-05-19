HDC Learning
============

After learning about representing and manipulating information in hyperspace, we can implement our first HDC classification model! We will use as an example the famous MNIST dataset that contains images of handwritten digits.


We start by importing Torchhd and any other libraries we need:

.. code-block:: python

	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	import torchvision
	from torchvision.datasets import MNIST
	import torchmetrics

	from torchhd import functional
	from torchhd import embeddings

Datasets
--------

Next, we load the training and testing datasets: 

.. code-block:: python

	transform = torchvision.transforms.ToTensor()

	train_ds = MNIST("../data", train=True, transform=transform, download=True)
	train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

	test_ds = MNIST("../data", train=False, transform=transform, download=True)
	test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


In addition to the various datasets available in the Torch ecosystem, such as MNIST, the :ref:`datasets` module provides interface to several commonly used datasets in HDC. Such interfaces inherit from PyTorch's dataset class, ensuring interoperability with other datasets.

Training
--------

To perform the training, we start by defining a model. In addition to specifying the basis-hypervectors sets, the core part of the model is the encoding function. In the example below, we use random-hypervectors and level-hypervectors to encode the position and value of each pixel, respectively:

.. code-block:: python

	class Model(nn.Module):
	    def __init__(self, num_classes, size):
	        super(Model, self).__init__()

	        self.flatten = torch.nn.Flatten()

	        self.position = embeddings.Random(size * size, DIMENSIONS)
	        self.value = embeddings.Level(NUM_LEVELS, DIMENSIONS)

	        self.classify = nn.Linear(DIMENSIONS, num_classes, bias=False)
	        self.classify.weight.data.fill_(0.0)

	    def encode(self, x):
	        x = self.flatten(x)
	        sample_hv = functional.bind(self.position.weight, self.value(x))
	        sample_hv = functional.multiset(sample_hv)
	        return functional.hard_quantize(sample_hv)

	    def forward(self, x):
	        enc = self.encode(x)
	        logit = self.classify(enc)
	        return logit


	model = Model(len(train_ds.classes), IMG_SIZE)
	model = model.to(device)


Having defined the model, we iterate over the training samples to create the class-vectors:

.. code-block:: python

    for samples, labels in train_ld:
        samples = samples.to(device)
        labels = labels.to(device)

        samples_hv = model.encode(samples)
        model.classify.weight[labels] += samples_hv

    model.classify.weight[:] = F.normalize(model.classify.weight)

Testing
-------

With the model trained, we can classify the testing samples by encoding them and comparing them to the class-vectors:

.. code-block:: python

    for samples, labels in test_ld:
        samples = samples.to(device)

        outputs = model(samples)
        predictions = torch.argmax(outputs, dim=-1)
        accuracy.update(predictions.cpu(), labels)