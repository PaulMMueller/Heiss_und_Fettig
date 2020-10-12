#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 17:55:25 2020

@author: paul

Lerning Numbers
"""

import numpy as np
from  mlxtend.data import loadlocal_mnist

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
        self.width = N_in*0.25+0.5
        
    def sigmoid(self,val):
        return 1/(1+np.exp(-val/self.width))
    def der_sigmoid(self,val):
        return  (1/self.width) * self.sigmoid(val/self.width)*self.sigmoid(-val/self.width)
        
    def propagate(self,a_in):

        self.a_out =  self.sigmoid(np.matmul(self.weight_matrix, a_in) +self.bias_vector)
        
        
        return self.a_out
    def get_derivatives(self,a_in,pre_grad):
       
        grad_z = self.der_sigmoid(np.matmul(self.weight_matrix,a_in)+self.bias_vector)
        pre_grad_times_grad_z = pre_grad*grad_z
        bias_dev = pre_grad_times_grad_z*np.ones(self.N_out)

        weight_dev = np.outer(pre_grad_times_grad_z,a_in)

        a_in_dev = np.matmul((pre_grad_times_grad_z),(self.weight_matrix))
        return (a_in_dev,weight_dev,bias_dev)
    
    
    
    

# start_l = Layer(3,2)
# next_l = Layer(start_l.N_out,4)
# stuff = start_l.propagate(bsp)
# print(stuff)
# print(start_l.weight_matrix,start_l.bias_vector)
# print(start_l.get_derivatives(bsp))

class Learner():    
    def __init__(self,layer_structure,data_set,labels,learn_ratio= 1, batch_size=None):
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
            How much of gradient should flow into the change of the weights and biases. The default is 1.
        batch_size : TYPE, optional
            Size of batches before adjusting the weights and biases. The default is len(data_set).



        Returns
        -------
        None.

        """
        self.layer_structure = layer_structure

        self.data_set = data_set
        self.labels = labels
        self.out_list = np.zeros(len(self.labels))
        
        self.learn_ratio = learn_ratio
        if batch_size == None:
            self.batch_size =  len(data_set)
        else:
            self.batch_size =  batch_size
            
        self.layer_object_vector = []
        self.layer_weights_vector = []
        
        self.layer_bias_vector = []
        for N_in,N_out in zip(layer_structure[:-1],layer_structure[1:]):
            layer_object = Layer(N_in=N_in,N_out=N_out)
            self.layer_object_vector.append(layer_object)
            self.layer_weights_vector.append(layer_object.weight_matrix)
            self.layer_bias_vector.append(layer_object.bias_vector)
         
        self.layer_object_vector = np.array(self.layer_object_vector,dtype=object)
        self.layer_weights_vector = np.array(self.layer_weights_vector,dtype=object)
        
        self.layer_bias_vector = np.array(self.layer_bias_vector,dtype=object)
        
        self.layer_weight_grad = np.zeros_like(self.layer_weights_vector)
        self.layer_bias_grad = np.zeros_like(self.layer_bias_vector)
        
    def cost_function(self,a_out,real_value):
        return np.matmul((a_out-real_value),(a_out-real_value))
    def cost_function_grad(self,a_out,real_value):
        return 2*(a_out-real_value)
    # def get_gradient(self,)
    
    def get_result(self, test_data, test_label):
        layer_input = test_data
        layer_output = None
        for L in self.layer_object_vector:
            layer_output = L.propagate(layer_input)
            layer_input = layer_output
        return (layer_output,test_label)
    def forward_propagate_full_data(self, in_data):
        """
        Calculates the prediction of the network however, also returns the all hidden steps. The result is the last entry of vector of outpus

        Parameters
        ----------
        in_data : list
            Input data.

        Returns
        -------
        Array of all steps of the propagation: First entry is the input and last the result.

        """
        layer_inputs = [in_data]
        for L in self.layer_object_vector:
            layer_inputs.append(L.propagate(layer_inputs[-1]))
        return layer_inputs
    
    def label_to_out_list(self,label):
        outcome = np.copy(self.out_list)
        outcome[self.labels.index(label)] = 1
        return outcome

    
    def av_cost_of_batch(self,data):
        av_cost = 0
        for (D,L) in data:
            (out,_) = self.get_result(test_data=D, test_label=L)
            label_vector = self.label_to_out_list(L)
            av_cost += self.cost_function(out, label_vector)
        av_cost /= len(data)
        return av_cost
        
        
        
    def calculate_gradient(self,in_data,label):

        layers_weight_grad_example = np.zeros_like(self.layer_weights_vector)
        layers_bias_grad_example = np.zeros_like(self.layer_bias_vector)

        label_vec = self.label_to_out_list(label)
        layer_outputs = self.forward_propagate_full_data(in_data)
        
        first_cost_grad = self.cost_function_grad(layer_outputs[-1], label_vec)

        temp_a_grad = first_cost_grad 

        for i,(layer,layer_input) in enumerate(zip(self.layer_object_vector[::-1],(layer_outputs[:-1])[::-1])):

            a_out_dev,weight_dev,bias_dev = layer.get_derivatives(layer_input,temp_a_grad)
            # temp_a_grad *=a_out_dev

            temp_a_grad = a_out_dev
            layers_weight_grad_example[len(layers_weight_grad_example)-i-1] = weight_dev
            layers_bias_grad_example[len(layers_bias_grad_example)-i-1] = bias_dev
            
        

        return (layers_weight_grad_example, layers_bias_grad_example)
    
    def learn_from_batch(self,batch):
        av_weight_adjust, av_bias_adjust =  self.calculate_gradient(batch[0][0],batch[0][1])
        loss = self.av_cost_of_batch(batch)
        for dat in batch[1:]:
            in_data = dat[0]
            label = dat[1]
            
            weight_adjust, bias_adjust = self.calculate_gradient(in_data, label)
            av_bias_adjust += bias_adjust
            av_weight_adjust += weight_adjust
        av_bias_adjust = self.learn_ratio*av_bias_adjust/len(batch)    
        av_weight_adjust = self.learn_ratio*av_weight_adjust/len(batch)    
        
        for  (w,b,layer) in (zip(av_weight_adjust,av_bias_adjust,self.layer_object_vector)):
            
            layer.weight_matrix -=w
            
            layer.bias_vector -=b
        
        return loss
       
    def learn(self,epochs):
        for i  in range(1,epochs+1):
            print(f'Epoch: {i}/{epochs}')
            for j  in range(int(len(data_set)/(self.batch_size))): 
                try:
                    batch =  self.data_set[j*self.batch_size:self.batch_size*(j+1)]
                    print(batch)
                except IndexError:
                    print('Some data not used as batch size sucks.')
                loss = self.learn_from_batch(batch)
                print(f'Batch : {j+1}/{int(len(data_set)/(self.batch_size))},  Loss: {loss}') 
                
                
      
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
# length = 1000
# data_set = []
# possible_labels = range(0,4)
# j = 0
# for i in range(length):
#     in_data = ((i%4)/4,0)
#     label = i%4
#     data_set.append((in_data,label))

dat , label  =  loadlocal_mnist(images_path="/home/paul/Learning/train-images-idx3-ubyte", labels_path= "/home/paul/Learning/train-labels-idx1-ubyte")
dat = dat/256
data_set = (dat,label)

possible_labels = np.arange(0,10)

DL = Learner(layer_structure=(len(data_set[0][0]),16,16,len(possible_labels)), data_set=data_set,labels=possible_labels,learn_ratio=10,batch_size=1000)
#len(DL.layer_object_vector)
# DL.print_network_properties()  

# grad_w, grad_b = DL.calculate_gradient((1,0), 3)

# # print(DL.calculate_gradient((1,0), 3))
# print(DL.out_list)
DL.print_network_properties()
DL.learn(5)
print(DL.data_set)
# print(DL.get_result((1,0),3))
# print(DL.label_to_out_list(3))
# # print( DL.get_result((1,2),1))
