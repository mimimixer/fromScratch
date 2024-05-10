import sys
import numpy as np
import matplotlib

# 1. our first neuron! actually just doung the dot-rpoduct (=Skalarprodukt, inneres Produkt)
inputs1 = [1, 2, 3, 2.5]  # unique inputs = outputs from 3 differnt neurons in previous layers
weights1 = [0.2, 0.8, -0.5, 1.0]  # every input has a unique weight associated with it
# = every value coming from a neuron in previous layer has unique weight
bias1 = 2  # every neuron has its unique bias - for the neuron, not for the incoming input each
# (the weights are for this)

output1a = inputs1[0] * weights1[0] + inputs1[1] * weights1[1] + inputs1[2] * weights1[2] + inputs1[3] * \
           weights1[3] + bias1

# also possible to do it as dot-product with numpy
output1b = np.dot(inputs1, weights1) + bias1

print("the output of our first neuron is ", output1a)
print("the output of our first neuron is ", output1b, "as np-dot-product\n")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 2. now we have a neuron layer, for example the output layer with 3 neurons. the previous layer had 4 neurons
inputs2 = [1, 2, 3, 2.5]  # unique inputs = outputs from 4 differnt neurons in previous layers (representing diff. features)
weights2a = [0.2, 0.8, -0.5, 1.0]  # every input has a unique weight associated with it
weights2b = [0.5, -0.91, 0.26, -0.5]  # = every value coming from a neuron in previous layer has unique weight
weights2c = [-0.26, -0.27, 0.17, 0.87]
bias2a = 2  # every neuron has its unique bias - for the neuron, not for the incoming input each
bias2b = 3  # (the weights are for this)
bias2c = 0.5

output2a = [inputs2[0] * weights2a[0] + inputs2[1] * weights2a[1] + inputs2[2] * weights2a[2] + inputs2[3] * weights2a[
    3] + bias2a,
            inputs2[0] * weights2b[0] + inputs2[1] * weights2b[1] + inputs2[2] * weights2b[2] + inputs2[3] * weights2b[
                3] + bias2b,
            inputs2[0] * weights2c[0] + inputs2[1] * weights2c[1] + inputs2[2] * weights2c[2] + inputs2[3] * weights2c[
                3] + bias2c]

print("the output of our output layer consisting of 3 neurons is ", output2a, "\n")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 3. neuron layer, but with modularized code - cleaner, more dynamic
inputs3 = inputs2
weights3 = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases3 = [2, 3, 0.5]

layer_outputs = []  # output-array of current layer
for neuron_weights, neuron_bias in zip(weights3,biases3):  # zip fun combines two lists elementwise into a list of lists
    # like: 1st element list 1 and 1st element list2 = 1st list in zip
    neuron_output = 0  # output of given neuron
    for n_input, weight in zip(inputs3, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print("3 neuron layer outputs with function calculation: ", layer_outputs)

# again we can calculate this using numpy's dot-product

output3b = np.dot(weights3, inputs3) + biases3  # regard the changed order of the arrays, this is because of
# matrix multiplication depending on dimentions!

# using numpy
print("the output of our output layer consisting of 3 neurons is ", output3b, "using dot product with numpy\n")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# concept of shape:
# at each dimention: what is the size of that dimension?
# ex: 1D array(=vector): list 4 elements: shape=(4)
# ex: 2D array(=matrix) : list of lists (=[[1,1,1,1],[3,2,1,2]]: shape (2,4)
#     here: lol has to be homologous, like: both inside lists need same size
# ex: 3D array(=tensor) : lolol (=[[[3,4,5,6],[3,2,1,2]], [[1,1,1,1],[3,5,6,2]], [[1,7,7,1],[3,2,9,2]]]
#     shape (3,2,4)
# tensor is an object, that CAN be represented as an array (ONLY CAN!)

# dot-product (= inneres Produkt, Skalarprodukt zweier Vektoren)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 4a. one layer of 3 neurons, but now using a batch of 3 inputs (each input consisting of the data of 4 sensors (eg))
# and converting the layers to objects
# batches help with generalisation: we have now 4 features (4 different imputs: can be sensordata from 4 diff sensors. or
# return values from the previous layer with 4 neurons)

inputs4 =   [[1, 2, 3, 2.5],  # features: 4 values from diff. sensors. or readings from different times)
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]]  # now we offer batches: data from 4 sensors measured 3 times a day - dataOfDay
weights4 =   weights3

biases4 = biases3

##  ATTENTION! we won't be able to multiply inputs3 with weights as bots are (3x4)-matrices! shape problem!
# take care of this, so we need to transpone one of the matrices!


# therefore we will transpose the weights, so we keep the same order an use numpy. (we could have done this before too).

'''
#layer_outputs = [] #output-array of current layer
#for neuron_weights, neuron_bias in zip(weights, biases):#zip function combines two lists into a list of lists, elementwise
#                                                        # like: 1st element list 1 and 1st element list2 = 1st list in zip
#    neuron_output = 0 #output of given neuron
#    for n_input, weight in zip(inputs2, neuron_weights):
#        neuron_output += n_input*weight
#    neuron_output += neuron_bias
#    layer_outputs.append(neuron_output)

#print("layeroutputs: ", layer_outputs)
'''

# again we can calculate this using numpy's dot-product. we will need to transpose the weights matrix,
# that is currently saved as a list of lists. Numpy will automatially convert lists of lists to arrays to be able to do
# the required calculations. to be able to transpose the list of lists, we need to convert it ourselves to an array.

output4a = np.dot(inputs4,np.array(weights4).T) + biases4
# converting into array np.array(x) and transposing it with array.T
# matrix multiplication depending on dimentions!

print("the output of the 3 batches of our output layer consisting of 3 neurons using dot product with numpy is \n",
      output4a)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 4b. two layers of 3 neurons, but now using a batch of 3 inputs (each input consisting of the data of 4 sensors (eg))
# and converting the layers to objects
# batches see 3a

# input will be called X (capital x) in the future, as input feature sets are called X as a convention
# input -> X, Xtrain, Xtest: training data. input data. we will try to scale input data between [-1, 1]

# the weights matrix + biases will be called hidden layer - hidden as we (=programmer) do not specify/define
# the changes (behaviour, weights) of this layer, so it'#'s called hidden
# we define optimizer, hyperparameter etc. but nn defines weights

# here we add a 2nd hidden layer = weights-matrix + biases:
weights4b = [[0.1, -0.14, 0.5],  # hidden layer 2
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases4b = [-1, 2, -0.5]

##  ATTENTION! we won't be able to multiply inputs3 with weights as bots are (3x4)-matrices! shape problem!
# take care of this, so we need to transpone one of the matrices!

# therefore we will transpose the weights, so we keep the same order an use numpy. (we could have done this before too).

# again we can calculate this using numpy's dot-product. we will need to transpose the weights matrix,
# that is currently saved as a list of lists. Numpy will automatially convert lists of lists to arrays to be able to do
# the required calculations. to be able to transpose the list of lists, we need to convert it ourselves to an array.

layer4a_outputs = np.dot(inputs4, np.array(weights4).T) + biases4
# converting into array np.array(x) and transposing it with array.T
# matrix multiplication depending on dimensions!
layer4b_outputs = np.dot(layer4a_outputs, np.array(weights4b).T) + biases4b

print("the output of the 3 batches of our 2 layers consisting of 3 neurons each using dot product with numpy is "
      "\n", layer4b_outputs)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 4c. convert the 2-layer-neura-network to an object so that the code doesn't look so crowded

# initialising layer:
#   a) loading saved trained model=saved weights+biases.
#   b) initialise weights and biases:
#       weights: random values between [-1,1] but smaller values = better, so hope that they will stay in range [-1,1]
#               because if weights are > |1| they will get bigger and bigger with each training, so model explodes.
#               therefore also inputs are usually normalized
#       biases: usually =0. but might be problematic, so that if weight produces small value and once output is 0,
#       this will propagate through the network, that will have only 0 values => network is dead.

X = inputs4     # will be called X (capital x) in the future, as input feature sets are
                # called X as a convention - X, Xtrain, Xtest -> training data. input data

np.random.seed(0)
'''This is a convenience, legacy function that exists to support older code that uses the singleton RandomState. 
Best practice is to use a dedicated Generator instance rather than the random variate generation methods exposed 
directly in the random module.'''


class Layer_Dense:          #layer-object definition
    def __init__(self, n_inputs, n_neurons):  # initialize the neurons in a form you dont need transpose
        # If positive int_like arguments are provided, randn generates an array of shape (d0, d1, ..., dn), filled with
        # random floats sampled from a univariate “normal” (Gaussian) distribution of mean 0 and variance 1
        self.weight_numbers = 0.10 * np.random.randn(n_inputs, n_neurons)  # here 4 inputs, and 3 neurons = shape. normalize (= *0.10)
        # also invert order of inputs/neurons no neet to transpose
        self.bias_numbers = np.zeros((1, n_neurons))  # here pass shape AS parameters - return tupel of shape

    def forward(self, input_data):  # forward pass: inputs=training data for 1st layer or output of previous layer
        self.output_data = np.dot(input_data, self.weight_numbers) + self.bias_numbers


layer1 = Layer_Dense(4, 5)                          #initialisation 1st layer
layer2 = Layer_Dense(5, 2)                          #initialisation 2nd layer

print()
layer1.forward(X)                                   #sending data trhough 1st layer
print("layer1 output: \n", layer1.output_data)
layer2.forward(layer1.output_data)                  #sending data trhough 1st layer
print("layer2 output: \n", layer2.output_data)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 5. activation functions: rectified linear, step, sigmoid

#5a. unit step function:


















# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

'''
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    print("python: ", sys.version)
    print("numpy: ", np.__version__)
    print("mathplotlib: ", matplotlib.__version__)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
'''
