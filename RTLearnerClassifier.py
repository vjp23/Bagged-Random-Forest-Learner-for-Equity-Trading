# Vince Petaccio

import numpy as np  # Import Numpy
import random as rd  # Import random for use in random feature selection

class RTLearnerClassifier:

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def addEvidence(self,X,Y):
        """
        Adds evidence to the learner's data set
        :param X: Observation data to be added to independent variable, ndarray
        :param Y: Observation data to be added to dependent variable, ndarray
        """
        if self.verbose:
            print "Plant the tree's seeds"
        # Tack the Y values onto the right side of the Xs
        tree_seeds = np.hstack((X,Y[:,None]))
        # Build the decision tree
        self.tree = self.plantTree(tree_seeds)
        if self.verbose:
            print "Tree fully grown"

    def query(self,Xin):
        """
        Uses stored evidence to build a random tree learner, then performs a regression
            on Xin, returning a single y value
        :param Xin: Observation data to be regressed, 1-dimensional ndarray
        :return: Regression result
        """
        guesses = np.ones((Xin.shape[0])) # Build the output vector
        for sample in range(0,Xin.shape[0]): # Query the tree for each sample row
            trimmed = self.tree # Copy the tree afresh
            if self.verbose:
                print "Query row = ", sample
            if len(trimmed.shape) == 1:
                guesses[sample] = trimmed[1]
            else:
                while trimmed[0,0] != -1: # Not a leaf
                    split_val = trimmed[0,1]
                    if self.verbose:
                        print "Test value = ", Xin[sample, :][int(trimmed[0, 0])]
                        print "Split value is ", split_val
                    if ((Xin[sample,:][int(trimmed[0,0])]) <= split_val): # Go left
                        if self.verbose:
                            print "Left tree, starting at row ", int(trimmed[0,2])
                        trimmed = trimmed[int(trimmed[0,2]):,:] # Rebuild the tree starting at the L branch
                        if self.verbose:
                            print "Shape of left tree: ", trimmed.shape
                    else: # Go right
                        if self.verbose:
                            print "Right tree, start at row ", int(trimmed[0,3])
                        trimmed = trimmed[int(trimmed[0,3]):,:] # Rebuild the tree starting at the R branch
                        if self.verbose:
                            print "Shape of right tree: ", trimmed.shape
                # Now at a leaf- accept the output value
                guesses[sample] = trimmed[0,1]
        return(guesses) # Return the array


    def plantTree(self,seeds):
        """
        Recursive function which builds the random tree
        :param seeds: data to parse
        :return: tree data structure
        """
        if self.verbose:
            print "seeds variable size is ", seeds.shape
        # If this is the last piece of data, return the value as a leaf
        if seeds.shape[0] <= self.leaf_size:  # Only leaf_size samples
            if self.verbose:
                print "Lonely leaf"
            meanseed = np.round(np.average(seeds,axis=0)) # Use the mean of all y-values
            return(np.array([-1,meanseed[-1],np.nan,np.nan]))  # Use associated y value as leaf value
        # If all remaining y values are identical, return that y value as a leaf
        if (list(seeds[:,-1]).count(list(seeds[:,-1])[0]) == len(list(seeds[:,-1]))):
            if self.verbose:
                print "Grouped leaf"
            return(np.array([-1,np.round(seeds[0,-1]),np.nan,np.nan])) # Just pick one
        # The data is not a leaf. Build a tree.
        split_val, i, num_features = self.getSplits(seeds)
        # Ensure that the trees aren't 0-length
        attempt = 0
        while ((seeds[(seeds[:, i] > split_val), :]).shape[0] == 0):
            attempt += 1
            split_val, i, num_features = self.getSplits(seeds)
            if attempt == 10 * num_features:
                # Potential situation where each column is homogeneous
                if self.verbose: print "Caution: Lots of split_val attempts!"
                homogeneous_features = 0
                for feature in range(0,num_features):
                    if (seeds[:, feature].sum() / (seeds[0, feature] * len(seeds[:,feature])) == 1.0):
                        homogeneous_features += 1 # Everything in this column is identical
                    if (homogeneous_features * (1.0 / num_features) == 1): # ALL columns are homogeneous
                        meanseed = np.average(seeds, axis=0)  # Use the mean of all y-values
                        return (np.array([-1, meanseed[-1], np.nan, np.nan]))  # Use associated y value as leaf value
        # Build the data to send left or right
        left_tree = self.plantTree(seeds[(seeds[:, i] <= split_val), :])  # Recursively build L tree
        right_tree = self.plantTree(seeds[(seeds[:, i] > split_val), :])  # Recursively build R tree
        # Reshape the left tree to enable accurate entry counting
        if len(left_tree.shape) == 1:
            left_reshaped = left_tree.reshape(left_tree.shape[0], -1)
            left_rows = left_reshaped.shape[1]
        else:
            left_rows = left_tree.shape[0]
        # Build the root
        root = [i, split_val, 1, left_rows+1]
        # Build the tree
        return np.vstack((root, left_tree, right_tree))

    def getSplits(self,seeds):
        """
        Builds the split values- abstracted due to repeated use
        :param seeds: The seeds variable used by plantTree
        :return: split_val and i for use in splitting from nodes
        """
        num_features = seeds.shape[1] - 2  # Count the number of features in the data, removing y
        i = rd.randint(0, num_features)  # PICK A RANDOM SPLITTING FEATURE as per Cutler
        splits = np.random.choice(seeds.shape[0], 2, replace=False)   # Choose random rows to split on
        split_val = (seeds[splits[0], i] + seeds[splits[1], i]) / 2   # Define the split_val
        return split_val, i, num_features
