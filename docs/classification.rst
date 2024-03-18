HDC Learning
============

After learning about representing and manipulating information in hyperspace, we can implement our first HDC classification model! We will use as an example the famous MNIST dataset that contains images of handwritten digits.


We start by importing Torchhd and the other libraries we need, in addition to specifying the training parameters:

.. code-block:: python

	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	import torchvision
	from torchvision.datasets import MNIST
	# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
	import torchmetrics
	
	import torchhd
	from torchhd.models import Centroid
	from torchhd import embeddings

	# Use the GPU if available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Using {} device".format(device))

	DIMENSIONS = 10000
	IMG_SIZE = 28
	NUM_LEVELS = 1000
	BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones

Datasets
--------

Next, we load the training and testing datasets: 

.. code-block:: python

	transform = torchvision.transforms.ToTensor()

	train_ds = MNIST("../data", train=True, transform=transform, download=True)
	train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

	test_ds = MNIST("../data", train=False, transform=transform, download=True)
	test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


In addition to the various datasets available in the Torch ecosystem, such as MNIST, the :ref:`datasets` module provides an interface to several commonly used datasets in HDC. Such interfaces inherit from PyTorch's dataset class, ensuring interoperability with other datasets.

Training
--------

To perform the training, we start by defining an encoding. In addition to specifying the basis-hypervectors sets, a core part of learning is the encoding function. In the example below, we use random-hypervectors and level-hypervectors to encode the position and value of each pixel, respectively:

.. code-block:: python

	class Encoder(nn.Module):
		def __init__(self, out_features, size, levels):
			super(Encoder, self).__init__()
			self.flatten = torch.nn.Flatten()
			self.position = embeddings.Random(size * size, out_features)
			self.value = embeddings.Level(levels, out_features)

		def forward(self, x):
			x = self.flatten(x)
			sample_hv = torchhd.bind(self.position.weight, self.value(x))
			sample_hv = torchhd.multiset(sample_hv)
			return torchhd.hard_quantize(sample_hv)

	encode = Encoder(DIMENSIONS, IMG_SIZE, NUM_LEVELS)
	encode = encode.to(device)

	num_classes = len(train_ds.classes)
	model = Centroid(DIMENSIONS, num_classes)
	model = model.to(device)

Having defined the model, we iterate over the training samples to create the class-vectors:

.. code-block:: python

	with torch.no_grad():
		for samples, labels in tqdm(train_ld, desc="Training"):
			samples = samples.to(device)
			labels = labels.to(device)

			samples_hv = encode(samples)
			model.add(samples_hv, labels)

Testing
-------

With the model trained, we can classify the testing samples by encoding them and comparing them to the class-vectors:

.. code-block:: python

	accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

	with torch.no_grad():
		model.normalize()

		for samples, labels in tqdm(test_ld, desc="Testing"):
			samples = samples.to(device)

			samples_hv = encode(samples)
			outputs = model(samples_hv, dot=True)
			accuracy.update(outputs.cpu(), labels)

	print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")