import csv
import numpy as np

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data(training_total = 10000, testing_total = 2000):
	with open('train.csv', 'rb') as f_training:
		reader = csv.reader(f_training)
		reader.next()

		count = 0

		training_inputs = []
		training_results = []
		testing_inputs = []
		testing_results = []

		for row in reader:
			# CSV reader is strings
			row_2_floats = [float(element) for element in row]

			if count < training_total:
				y = row_2_floats[0]
				row_2_floats.pop(0)
				x = [val/255.0 for val in row_2_floats]
				y = vectorized_result(y)
				training_inputs.append(x)
				training_results.append(y)


			elif count < testing_total+training_total:
				y = row_2_floats[0]
				row_2_floats.pop(0)
				x = [val/255.0 for val in row_2_floats]
				testing_inputs.append(x)
				testing_results.append(int(y))

			else:
				break

			count = count + 1

		training_inputs = [np.reshape(x, (784, 1)) for x in training_inputs]
		training_results = np.array(training_results)

		testing_inputs = [np.reshape(x, (784, 1)) for x in testing_inputs]
		testing_results = np.array(testing_results)

		training_data = zip(training_inputs,training_results)
		testing_data = zip(testing_inputs,testing_results)

	return training_data, testing_data

import network

def train_network():
	training_data, test_data = load_data(41000, 100)
	net = network.Network([784, 30, 10])
	net.SGD(training_data, 20, 10, 3.0, test_data=test_data)

	testing_inputs = []

	with open('test.csv', 'rb') as f_testing:
		reader = csv.reader(f_testing)
		reader.next()
		for row in reader:
			row_2_floats = [float(element) for element in row]
			x = [val/255.0 for val in row_2_floats]
			testing_inputs.append(x)
		testing_inputs = [np.reshape(x, (784, 1)) for x in testing_inputs]
		
	outputs = []
	for x in testing_inputs:
		y = np.argmax(net.feedforward(x))
		outputs.append(y)

	with open('submission.csv', 'wb') as f:
	    writer = csv.writer(f)
	    outputs = np.reshape(outputs,(len(outputs),1))
	    writer.writerows(outputs)
	    f.close()


# net = network.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 1.0, test_data=test_data)
# net.print_all()