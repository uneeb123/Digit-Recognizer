Kaggle: Digit Recognizer
----------------------

This code solves the Digit Recognizer problem from Kaggle (source: www.kaggle.com/c/digit-recognizer)

Open Python console and type the following:
>>> import kaggle_loader
>>> kaggle_loader.train_network()

This will load data from the train.csv and test.csv files. Set up a neural network with 3 layers (784 neurons; 30 neurons; 10 neurons). And then apply stochastic gradient descent to learn the network. Then, it tests the data from test.csv and saves its output to submission.csv. Everything is pretty much modifiable in "kaggle_loader.py"

Adapted from Michael Nielsen's "Neural Networks and Deep Learning" book
