import numpy as np
from typing import Optional, List, Union, Tuple
from collections import Counter


class DistributionModel:
    """
    The class implements a histogram-like distribution model where the values a grouped into bins.
    From this distribution model, it is then possible to sample new values according to the distribution.
    The size of these bins can be configured.

    Properties:
        bin_size: (read only) size of the histogram bins
        distribution_prob: (read only) all filled bins with their according probabilities
        min_value: (read only) the lower bound of a filled bin
        max_value: (read only) the upper bound of a filled bin
        expected_value: (read only) expected value of the samples
        variance: (read only) variance of the samples
        standard_deviation: (read only) standard deviation
        allow_negative_samples: Allow negative samples, useful for debugging if no negative samples are expected
    """
    def __init__(self, samples: Optional[List[Union[int, float]]] = None, bin_size: Union[int, float] = 100,
                 allow_negative_samples: bool = False):
        """
        :param bin_size: size of the histogram bins
        :param samples: (optional) list of samples that should be added
        :param allow_negative_samples: (optional) Allowing negative values to be added to the model.
                                        Disabled by default.
        """
        self.n = 0
        self._distribution_prob = None
        self._bin_size = bin_size
        self.allow_negative_samples = allow_negative_samples
        self._min_value = None
        self._max_value = None
        self._expected_value = None
        self._variance = None
        self._standard_deviation = None

        if samples is not None:
            self.fit(samples)

    @property
    def distribution_prob(self) -> List[Tuple[float, float]]:
        return self._distribution_prob

    @property
    def bin_size(self) -> Union[int, float]:
        return self._bin_size

    @property
    def min_value(self) -> float:
        return self._min_value

    @property
    def max_value(self) -> float:
        return self._max_value

    @property
    def expected_value(self) -> float:
        return self._expected_value

    @property
    def variance(self) -> float:
        return self._variance

    @property
    def standard_deviation(self) -> float:
        return self._standard_deviation

    def clear(self) -> None:
        """ Removes all samples from the model and resets the related statistical values
        :return: None
        """
        self._distribution_prob = None
        self._min_value = None
        self._max_value = None
        self._expected_value = None
        self._variance = None
        self._standard_deviation = None

    def fit(self, samples: Union[List[Union[int, float]]]) -> None:
        """ Fits the distribution model to a number of samples. Previously estimated values will be overwritten.
        :param samples: Samples to which the model is fitted. The samples can be given as list or as set.
        :return: None
        """

        if len(samples) == 0:
            raise AssertionError('Cannot fit distribution with no values provided,'
                                 ' to remove all samples from the model call clear()')

        def bin_map(sample: Union[int, float]) -> float:
            """ Maps a sample to the representative of the corresponding bin, which is the center of the bin.
            :param sample: Sample that should be sorted into a bin
            :return: Representative of the bin
            """
            if sample >= 0:
                offset = self.bin_size/2
            elif not self.allow_negative_samples:
                raise AssertionError(
                    "No negative samples allowed. For enabling add allow_negative_samples = True as parameter")
            else:
                offset = -self.bin_size/2

            return int(sample / self.bin_size)*self.bin_size+offset

        # Use the Counter class to count the absolute occurrences
        counter_abs = Counter(list(map(bin_map, samples)))

        # Compute the probabilities for each bin, and then sort by the representative of the bin (required for sampling)
        self._distribution_prob = list(map(lambda x: (x[0], x[1]/len(samples)), list(counter_abs.items())))
        self._distribution_prob.sort(key=lambda x: x[0])

        self.n = len(samples)

        # Update statistical values
        if self.n > 0:
            self._min_value = self._distribution_prob[0][0] - self._bin_size/2
            self._max_value = self._distribution_prob[-1][0] + self._bin_size/2
            self._expected_value = sum([prob*val for val, prob in self._distribution_prob])
            self._variance = sum([prob*(val-self._expected_value)**2 for val, prob in self._distribution_prob])
            self._standard_deviation = self._variance ** 0.5

    def sample_value(self, rng: Optional[np.random.random] = None, random_state: Optional[int] = None,
                     sample_integer: bool = True, minimum_value: Optional[int] = None,
                     maximum_value: Optional[int] = None) -> Union[int, float]:
        """
        Sample a value according to the currently estimated distribution saved in the distribution model.
        It is also possible to restrict the area to an interval from which a sample is drawn.
        In this case, the distribution inside the interval is normalized to the probability 1 and then used for sampling.

        :param rng: (optional) The numpy rng that should be used, the rng should generate a number in the interval [0,1)
                    If not set a new uniform rng is used.
        :param random_state: (optional) Seed for the default random number generator.
                             If not set, no seed is used for the rng, so the samples are no reproducible.
        :param sample_integer: (optional) When set to true, the sampled value is an integer, otherwise it is a float.
                               Default: True.
        :param minimum_value: (optional) minimal value that should be sampled (including minimum_value)
        :param maximum_value: (optional) maximum value that should be sampled (excluding maximum_value)
        :return: sample according to the distribution Integer, when sample_integer is True.
        """

        if rng is None:
            rng = np.random.default_rng(random_state)

        if self.n == 0:
            raise AssertionError("No samples has been added to the model. Sampling not possible.")

        if minimum_value is None:
            p_min = 0
        else:
            p_min = self.get_cdf_value(minimum_value)

        if maximum_value is None:
            p_s = 1-p_min
        else:
            p_s = self.get_cdf_value(maximum_value)-p_min

        temp = p_min + rng.random()*p_s

        for (val, prob) in self.distribution_prob:
            if temp <= prob:
                value = val-self.bin_size/2 + temp/prob*self.bin_size
                if sample_integer:
                    return int(value)
                else:
                    return value
            else:
                temp -= prob

        return self.max_value

    def get_cdf_value(self, value: Union[int, float]) -> float:
        """
        Returns the value of the cumulative distribution function (cdf) for the given value.
        In other words returns the probability that a random sample is smaller than value.

        :param value: Value for which the cdf should be evaluated
        :return: Output of the cdf function at the given value.
        """

        if value < self.min_value:
            return 0
        elif value > self.max_value:
            return 1
        else:
            probability = 0
            for (val, prob) in self.distribution_prob:
                if value < (val+self.bin_size/2):
                    return probability + prob*(value % self.bin_size) / 100
                else:
                    probability += prob

        return 1

    def __repr__(self):
        ret = "Number of Samples: " + str(self.n)
        ret += " Minimum value:" + str(self.min_value)
        ret += " Maximum value:" + str(self.max_value)
        ret += " Expected value:" + str(self.expected_value)
        ret += " Standard derivation:" + str(self.standard_deviation)
        ret += " Variance:" + str(self.variance)
        return ret

    def plot(self, show = False, fig = None, ax = None):
        """
        Creates a plot of the distribution model using matplotlib and
         returns a figure and axes with the corresponding plot.
        @:param show: (optional) When set to True the figure is directly shown
        @:param fig: (optional) Figure on which a new axes with the plot is created.
                     Will be overwritten when ax is given.
                     When not given and also ax is not provided the function creates a new figure
                     with one axes and uses this for the plot.
        @:param ax: (optional) axes on which the plot is created, when not provided
                    the function creates a new axes on the figure, when also the figure is not provided
                    then the function creates a new figure with one axes and uses this for the plot.
        :return: Figure and axes with the plot of the distribution.
                 When an axis but no figure is given as input then the tuple (None,ax) is returned.
        """
        import matplotlib.pyplot as plt
                
        if self.n == 0:
            raise AssertionError("No samples has been added to the model. Plot is empty.")

        if fig is None and ax is None:
            fig, ax = plt.subplots()
        elif fig is not None:
            ax = fig.add_axes()

        ax.hist(list(map(lambda x: x[0], self.distribution_prob)),
                bins=int((self.max_value - self.min_value)/self.bin_size),
                weights=list(map(lambda x: x[1], self.distribution_prob)),
                range=(self.min_value, self.max_value),
                alpha=0.75)
        ax.set_xlabel("Values")
        ax.set_ylabel("Probability")
        ax.axvline(x=0, linestyle='--', linewidth=1, color='grey')
        if show:
            plt.show()
        return fig, ax


def statistical_distance(d1: DistributionModel, d2: DistributionModel) -> float:
    """
    Calculates the statistical distance (total variation distance,
    https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures)
    of two distribution models (d1 and d2).
    :param d1: DistributionModel for comparison
    :param d2: DistributionModel for comparison
    :return: statistical distance
    """
    if d1.n == 0:
        raise AssertionError("No samples has been added to the first model. No comparison possible.")
    elif d2.n == 0:
        raise AssertionError("No samples has been added to the second model. No comparison possible.")

    ret = 0

    values = set(map(lambda x: x[0], d1.distribution_prob)).union(set(map(lambda x: x[0], d2.distribution_prob)))
    c1 = Counter(dict(d1.distribution_prob))
    c2 = Counter(dict(d2.distribution_prob))

    for val in values:
        ret += abs(c1[val]-c2[val])

    return 0.5*ret
