# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:08:18 2020

@author: andreas
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import pandas as pd





# Custom PCA
class custom_PCA:
    
    
    """
    #Initialize 
    pca = custom_PCA(dataframe, k)
    #fit on the dataframe
    pca.fit()

    """
    def __init__(self, A, k,  is_normalized = True):
        self.A = A.iloc[:, :-1].to_numpy()
        self.k = k
        # No of rows
        self.N_rows = A.shape[0]
        # No of columns
        self.N_columns = A.shape[1]
        # Target class
        self.target_class = A.iloc[:, -1].to_numpy().reshape(self.N_rows,1).astype(int)
        self.is_normalized = is_normalized

    def norm(self, M= ''):
        if M=='' and self.is_normalized == False:
            return (self.A.T - self.A.T.mean(axis = 0))/(self.A.T.max(axis=0)- self.A.T.min(axis=0))
        elif M!='':
            return (M - M.mean(axis = 0))/(M.max(axis=0)- M.min(axis=0))
        else:
            return self.A.T
 
    
    # Calculate the covariance matrix
    def cov_(self):
        A = self.norm()
        return ((A).dot(A.T))/A.shape[0]

    def generate_reduced_eigenvectors(self):
        ''' Provide with the k-threshold and the eigenvalue/eigenvector tuple.
            Returns the new W matrix with the reduced eigenvectors.
        '''
        # Initialize
        partial_sum = 0
        idx = 0
        
        e_tuple = self.list_eigenvalues()
        
        # Sort e_tuple by the highest eigenvalue
        sort_eigen = sorted(e_tuple, key=lambda x: x[0], reverse = True)

        ## Define how many eigenvectors to keep
        # define the Sum of the eigenvalues
        sum_eig = sum([pair[0] for pair in sort_eigen])
        # Add eigenvectors as the k is smaller than the fraction
        for ii in range(len(sort_eigen)):
            if (partial_sum/sum_eig) <= self.k:
                partial_sum += sort_eigen[ii][0]
                # Index of Principal Components in the sort_eigen list
                idx+=1
            else:
                break

        print('Final selection is the first {} PCs'.format(idx))

        # Select eigenvalues, eigenvectors
        selected_eig = [sort_eigen[x][1] for x in range(idx) ]
        # reshape eigenvectors
        stack_eig = list(selected_eig[x].reshape(selected_eig[x].shape[0],1) for x in range(idx))
        # return W
        return np.hstack(stack_eig)
    
    def list_eigenvalues(self):
        e_values, e_vector  = LA.eig(self.cov_())
        # List of (eigenvalues, eigenvectors)
        e_tuple = [ (np.abs(e_values[i].real), e_vector[:,i].real) for i in range(len(e_values))]
        return e_tuple
    
    def fit(self):
        # Projected Data on the N Principal Components and Normalize
        # Should follow: (n,m) = (n,p) x (p, m)
        data_PC_projected = self.A.dot(self.generate_reduced_eigenvectors())
        #N Normalize
        data_PC_projected = (data_PC_projected - data_PC_projected.mean(axis = 0))/(data_PC_projected.max(axis=0)- data_PC_projected.min(axis=0))
        
        # Label the new dataset
        labeled_PC_data = np.concatenate((data_PC_projected, self.target_class), axis = 1)
        
        
        print('Dimensionaly reduced matrix shape', labeled_PC_data.shape )
        
        # Name new columns
        cols = ['PC_'+str(x+1) for x in range(data_PC_projected.shape[1])]
        cols.insert(len(cols), 'Target_Class')

        return pd.DataFrame(data = labeled_PC_data, columns = cols)






# Visualize Confusion Matrix
def confusion_matrix_visualization(confusion_matrix):
    '''Takes input the (2,2) array of the confusion matrix and visualizes.'''
    
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           xticklabels=['Predict_Call', 'Predict_not_call'],  
           yticklabels=['GT_Call', 'GT_not_call'],
           title = 'Confusion Matrix' )
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, confusion_matrix[i, j], size = 18, horizontalalignment='center', verticalalignment='center')
            
            
            
def normalize(M):
    '''M is the input matrix. Normalizes the array by column'''
    return (M - M.mean(axis = 0))/(M.max(axis=0)- M.min(axis=0))



def pprint_coefs(coefs=None, names = None, sort = False, tr=None):
    ''' Helper for Ridge method. Returns the best attribute according to the threshold `tr`'''
    
    if names == None:
        names = ["X{}".format(x for x in range(len(coefs)))]
    lst = zip(coefs, names)
    #print(lst)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return [name for coef, name in lst if abs(coef)>tr]