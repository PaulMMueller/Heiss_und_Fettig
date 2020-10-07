#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 17:55:25 2020

@author: paul

Lerning Numbers
"""

import numpy as np


class Layer():
    
    def __init__(self,N_in,N_out):
        """
        

        Parameters
        ----------
        N_in : int
            Number of inputs.
        N_out : int
            Number of outputs.

        Returns
        -------
        None.

        """
        self.N_in = N_in
        self.N_out =N_out
        self.weight_matrix = np.random.rand(self.N_out,self.N_in)
        self.bias_vector = np.random.rand(self.N_out)
        self.a_out = None
        
    def sigmoid(self,val):
        return 1/(1+np.exp(-val))
    def der_sigmoid(self,val):
        return  (1/(1+np.exp(-val))**2) *(-np.exp(-val)) 
        
    def propagate(self,a_in):

        self.a_out =  self.sigmoid(np.matmul(self.weight_matrix, a_in) +self.bias_vector)
        
        
        return self.a_out
    def get_derivatives(self,a_in):
        bias_dev = self.der_sigmoid(np.ones(self.N_out))
        weight_dev = self.der_sigmoid(np.tile(a_in,(self.N_out,1)))
        a_out_dev = self.der_sigmoid(np.sum(self.weight_matrix,axis=1))
        return (a_out_dev,weight_dev,bias_dev)
    
    
    
    
# bsp = np.array([0,0.5,1])

# start_l = Layer(3,2)
# next_l = Layer(start_l.N_out,4)
# stuff = start_l.propagate(bsp)
# print(stuff)
# print(start_l.weight_matrix,start_l.bias_vector)
# print(start_l.get_derivatives(bsp))

class Learner():    
    def __init__(self,layer_structure,data_set,labels,learn_ratio= 0.8, batch_size=100):
        """
        

        Parameters
        ----------
        layer_structure : list
            List of integers. Every integer gives the size of the corresponding layer.
        data_set : list
            Should be a list of tupels. Tupels should consistof the input and the corresponding label.
        labels : list
            All possible labels
        learn_ratio : float, optional
            Which part of the data is used for training and which for testing. The default is 0.8.
        batch_size : TYPE, optional
            Size of batches before adjusting the weights and biases. The default is 100.



        Returns
        -------
        None.

        """
        self.layer_structure = layer_structure

        self.data_set = data_set
        self.labels = labels
        

            
        self.layer_object_vector = []
        self.layer_weights_vector = []
        
        self.layer_bias_vector = []
        for N_in,N_out in zip(layer_structure[:-1],layer_structure[1:]):
            layer_object = Layer(N_in=N_in,N_out=N_out)
            self.layer_object_vector.append(layer_object)
            self.layer_weights_vector.append(layer_object.weight_matrix)
            self.layer_bias_vector.append(layer_object.bias_vector)
         
        self.layer_object_vector = np.array(self.layer_object_vector)
        self.layer_weights_vector = np.array(self.layer_weights_vector)
        
        self.layer_bias_vector = np.array(self.layer_bias_vector)
        
        self.layer_weight_grad = np.zeros_like(self.layer_weights_vector)
        self.layer_bias_grad = np.zeros_like(self.layer_bias_vector)
        
    def cost_function(self,a_in,real_value):
        return (a_in-real_value)**2
    # def get_gradient(self,)
    
    def get_result(self, test_data, test_label):
        layer_input = test_data
        layer_output = None
        for L in self.layer_object_vector:
            layer_output = L.propagate(layer_input)
            layer_input = layer_output
        return (layer_output,test_label)
    
    def get_batch_results(self,data):
        
        
        
        return
    def print_network_properties(self):
        print('\nGiving the layer structure where every layer has as many outputs as the next has inputs.\n')
        for i, obj in enumerate(self.layer_structure[:-1]):
            print(f'Layer number: {i}')
            print(f'Layers inputs: {obj}')
            print(f'Layers weights: {self.layer_weights_vector[i]}')
            print(f'Layers biases: {self.layer_bias_vector[i]}')

        print(f'\nOverall outputs: {self.layer_structure[-1]}\n')
        return (self.layer_weights_vector,self.layer_bias_vector)
 
    
# Make a test data set
length = 1000
data_set = []
possible_labels = range(0,4)
j = 0
for i in range(length):
    in_data = ((i%4)/4,0)
    label = i%4
    data_set.append((in_data,label))

DL = Learner(layer_structure=(2,4), data_set=data_set,labels=possible_labels)

DL.print_network_properties()  

print( DL.get_result((1,2),1))
