# Modelling only one neuron with four inputs
# inputs=[1,2,3,2.5]    #input layer is simply whatever values you have that you are tracking
# weights=[0.2, 0.8, -0.5, 1.0]
# every unique neuron has a unique bias
# bias=2

# first step of neurons is:
# add up all the (inputs * weights)+bias
# output=inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias
# print(output)
#<------------------------------------------------------------>
# modelling three neurons that is one layer
# weights1=[0.2, 0.8, -0.5, 1.0]
# weights2=[0.5, -0.91, 0.26, -0.5]
# weights3=[-0.26, -0.27, 0.17, 0.87]
#
# weights=[[0.2, 0.8, -0.5, 1.0],
#          [0.5, -0.91, 0.26, -0.5],
#          [-0.26, -0.27, 0.17, 0.87]]
# bias1=2
# bias2=3
# bias3=0.5
#
# biases=[2,3,0.5]
#
# output=[inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
#         inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
#         inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]
# print(output)

# weights=[[0.2, 0.8, -0.5, 1.0],
#          [0.5, -0.91, 0.26, -0.5],
#          [-0.26, -0.27, 0.17, 0.87]]
#
# biases=[2,3,0.5]
#
# layer_outputs=[]
# for neuron_weights,neuron_bias in zip(weights,biases)

# some_value=-0.5
# weight=0.7
# bias=0.7
# print(some_value*weight)
# print(some_value+bias)

# Implementing one layer
# import numpy as np
# inputs = [[1,2,3,2.5],
#         [2.0,5.0,-1.0,-0.8],
#         [-1.5,2.7,3.3,-0.8]]
# weights = [[0.2,0.8,-0.5,1.0],
#          [0.5,-0.91,0.26,-0.5],
#          [-0.26,-0.27,0.17,0.87]]
#
# biases = [2,3,0.5]
#
# output=np.dot(inputs,np.array(weights).T) + biases
# print(output)

# implementing 2nd layer with neurons
import numpy as np
inputs = [[1,2,3,2.5],
         [2.0,5.0,-1.0,-0.8],
         [-1.5,2.7,3.3,-0.8]]
weights = [[0.2,0.8,-0.5,1.0],
          [0.5,-0.91,0.26,-0.5],
          [-0.26,-0.27,0.17,0.87]]

biases = [2,3,0.5]
weights2=[[0.1,-0.12,0.5],
          [-0.5,0.12,-0.33],
          [-0.44,0.73,-0.13]]
biases2=[-1,2,-0.5]
layer1_outputs=np.dot(inputs,np.array(weights).T) + biases
layer2_outputs=np.dot(layer1_outputs,np.array(weights2).T) + biases2
print(layer2_outputs)

# import numpy as np
# inputs=np.array([1,2,3,2.5])
# weights=np.array([[0.2, 0.8, -0.5, 1.0],
#           [0.5, -0.91, 0.26, -0.5],
#           [-0.26, -0.27, 0.17, 0.87]])
# biases=[2,3,0.5]
# output=np.dot(weights,inputs)+biases
# print(output)

#creating batches now(because batches helps in generalization)
# {batch of samples --> having many reading of sensors at any specific point in time}
# inputs=np.array([1,2,3,2.5],[2.0,5.0,-1.0,-0.8],[-1.5,2.7,3.3,-0.8])
# weights=np.array([[0.2, 0.8, -0.5, 1.0],
#           [0.5, -0.91, 0.26, -0.5],
#           [-0.26, -0.27, 0.17, 0.87]])
# biases=[2,3,0.5]
# output=np.dot(weights,inputs)+biases
# print(output)

# objects

# import numpy as np
# np.random.seed(0)
# X=[[1,2,3,2.5],
#    [2.0,5.0,-1.0,-0.8],
#    [-1.5,2.7,3.3,-0.8]]
#
# class layer_dense:
#     def __init__(self,n_inputs,n_neurons):
#         self.weights=0.10*np.random.randn(n_inputs,n_neurons)
#         self.biases=np.zeros((1,n_neurons))
#     def forward(self,inputs):
#         self.output=np.dot(inputs,self.weights)+self.biases
# # print(0.10*np.random.randn(4,3))
# layer1=layer_dense(4,5)
# layer2=layer_dense(5,5)
#
# layer1.forward(X)
# print(layer1.output)
# layer2.forward(layer1.output)
# print(layer2.output)

