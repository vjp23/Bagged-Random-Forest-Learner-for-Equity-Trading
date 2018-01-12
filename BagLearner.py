# Vince Petaccio

# Import Numpy
import numpy as np

class BagLearner:

    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = []

    def author(self):
        """
        Announces the author behind this code
        :return: author of this code
        """
        return 'Vince Petaccio'

    def addEvidence(self, x, y):
        """
        Adds evidence to the learner's data set
        :param x: Observation data to be added to independent variable, ndarray
        :param y: Observation data to be added to dependent variable, ndarray
        """
        # Instantiate all of the learners
        if self.verbose:
            print "Open the bags"
        learners = []
        for index in range(0, self.bags):
            learners.append(self.learner(**self.kwargs))
        self.learners = learners

        # Train each learner with a random sampling of the data USING REPLACEMENT
        if self.verbose:
            print "Train the bagged learners"
        for player in self.learners:
            if self.boost:
                # TODO: Implement boosting
                pass
            else:
                # Create a sample mask to select the data randomly with replacement
                sample_mask = np.random.choice(x.shape[0], x.shape[0], replace=True)
                # Train the learner with this data. Keep Y data paired
                player.addEvidence(x[sample_mask, :], y[sample_mask])

        if self.verbose:
            print "Bagged learners are ready"

    def query(self, xin):
        """
        Uses stored evidence to build a random tree learner, then performs a regression
            on xin, returning a single y value
        :param xin: Observation data to be regressed, 1-dimensional ndarray
        :return: Regression result
        """
        # Setup the variables
        if self.verbose:
            print "Start the query"
        guesses = []

        # Query each learner
        index = 0
        for player in self.learners:
            index += 1
            if self.verbose:
                print "Querying learner number ", index, "..."
            guesses.append(player.query(xin))

        # Return the mean of the outputs of the bagged learners
        if self.verbose:
            print "Finished querying"
        return np.round(np.average(guesses, axis=0))