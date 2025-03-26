# Modelling only one neuron with four inputs
inputs=[1,2,3,2.5]    #input layer is simply whatever values you have that you are tracking
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

weights=[[0.2, 0.8, -0.5, 1.0],
         [0.5, -0.91, 0.26, -0.5],
         [-0.26, -0.27, 0.17, 0.87]]

biases=[2,3,0.5]
#
# layer_outputs=[]
# for neuron_weights,neuron_bias in zip(weights,biases)

some_value=-0.5
weight=0.7
bias=0.7
print(some_value*weight)
print(some_value+bias)

# Activation function => functions that kind of dtermine that final output before it becomes input of another layer
# or it maybe dtermine the final output to your network in gneral