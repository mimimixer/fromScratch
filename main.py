import sys
import numpy as np
import matplotlib

# 1. our first neuron! actually just doung the dot-rpoduct (=Skalarprodukt, inneres Produkt)
inputs1a = [1, 2, 3, 2.5]           # unique inputs = outputs from 3 differnt neurons in previous layers
weights1a = [0.2, 0.8, -0.5, 1.0]   # every input has a unique weight associated with it
                                    # = every value coming from a neuron in previous layer has unique weight
bias1a = 2                          # every neuron has its unique bias - for the neuron, not for the incoming input each
                                    # (the weights are for this)

output1a = inputs1a[0]*weights1a[0] + inputs1a[1]*weights1a[1] + inputs1a[2]*weights1a[2] + inputs1a[3]*weights1a[3] + bias1a

#also possible to do it as dot-product with numpy
output1b = np.dot(inputs1a, weights1a) + bias1a

print("the output of our first neuron is ", output1a)
print("the output of our first neuron is ", output1b, "as np-dot-product")

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 2. now we have a neuron layer, for example the output layer with 3 neurons. the previous layer had 4 neurons
inputs2 = [1, 2, 3, 2.5]             # unique inputs = outputs from 3 differnt neurons in previous layers
weights1b = [0.2, 0.8, -0.5, 1.0]    # every input has a unique weight associated with it
weights2b = [0.5, -0.91, 0.26, -0.5]    # = every value coming from a neuron in previous layer has unique weight
weights3b = [-0.26, -0.27, 0.17, 0.87]
bias1b = 2                           # every neuron has its unique bias - for the neuron, not for the incoming input each
bias2b = 3                           # (the weights are for this)
bias3b = 0.5

output2a = [inputs2[0]*weights1b[0] + inputs2[1]*weights1b[1] + inputs2[2]*weights1b[2] + inputs2[3]*weights1b[3] + bias1b,
            inputs2[0]*weights2b[0] + inputs2[1]*weights2b[1] + inputs2[2]*weights2b[2] + inputs2[3]*weights2b[3] + bias2b,
            inputs2[0]*weights3b[0] + inputs2[1]*weights3b[1] + inputs2[2]*weights3b[2] + inputs2[3]*weights3b[3] + bias3b]

print("the output of our output layer consisting of 3 neurons is ", output2a)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 2. neuron layer, but with modularized code - cleaner, more dynamic
inputs2 = [1, 2, 3, 2.5]
weights =  [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

layer_outputs = [] #output-array of current layer
for neuron_weights, neuron_bias in zip(weights, biases):#zip function combines two lists into a list of lists, elementwise
                                                        # like: 1st element list 1 and 1st element list2 = 1st list in zip
    neuron_output = 0 #output of given neuron
    for n_input, weight in zip(inputs2, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print("layeroutputs: ", layer_outputs)

#again we can calculate this using numpy's dot-product

output2b = np.dot(weights, inputs2)+biases  #regard the changed order of the arrays, this is because of
                                            # matrix multiplication depending on dimentions!

print("the output of our output layer consisting of 3 neurons is ", output2b, "using dot product with numpy")

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#concept of shape:
    #at each dimention: what is the size of that dimension?
    # ex: 1D array(=vector): list 4 elements: shape=(4)
    # ex: 2D array(=matrix) : list of lists (=[[1,1,1,1],[3,2,1,2]]: shape (2,4)
    #     here: lol has to be homologous, like: both inside lists need same size
    # ex: 3D array(=tensor) : lolol (=[[[3,4,5,6],[3,2,1,2]], [[1,1,1,1],[3,5,6,2]], [[1,7,7,1],[3,2,9,2]]]
    #     shape (3,2,4)
    # tensor is an object, that CAN be represented as an array (ONLY CAN!)

#dot-product (= inneres Produkt, Skalarprodukt zweier Vektoren)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++





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

