import logging
import sys
import numpy as np

from operator import add
from typing import Dict, Optional, Union
from lazy_dataset import Dataset, from_dict

from mms_msg.sampling.utils.distribution_model import DistributionModel
from mms_msg.sampling.pattern.meeting.state_based.transition_model import MarkovModel, MultiSpeakerTransitionModel


logger = logging.getLogger('dataset_statistics_estimation')
logger.setLevel(logging.INFO)
if sys.stdout not in [handler.stream for handler in logger.handlers if type(handler) is logging.StreamHandler]:
    logger.addHandler(logging.StreamHandler(sys.stdout))


class MeetingStatisticsEstimatorMarkov:
    """
    Class that estimates characteristics of existing dataset. For the speaker transitions a MarkovModel is created.
    The model distinguishes between 4 transition (TH: Turn hold, TS: turn switch, OV: Overlap, BC: Backchannel).
    These transitions are based on the following paper:
        Improving the Naturalness of Simulated Conversations for End-to-End Neural Diarization,
        https://arxiv.org/abs/2204.11232

    Also, distributions for silence and overlap are computed, using a histogram like distribution model.
    This class also supports processing of datasets which utilize Voice activity detection (VAD) data.

    Each sample in the processed dataset must have the following keys:
        - speaker_id: List with the speaker_ids, which belong to the sources in the example
        - offset: Dictionary with a key that depends on the use of vad data:
                  'aligned_source' when it is used, 'original_source' otherwise.
                  The associated item must be a list of the offsets of the sources.
        - speaker_end: Dictionary with a key that depends on the use of vad data:
                  'aligned_source' when it is used, 'original_source' otherwise.
                  The associated item must be a list of the speaker endings of the sources.

    Properties:
        dataset: (read only) last processed dataset
        model: (read only) markov model of the transitions the meetings of the dataset
        silence_distribution: (read only) distribution model of the length of the silence
        overlap_distribution: (read only) distribution model of the length of the overlap
    """

    def __init__(self, dataset: Optional[Union[Dataset, Dict]] = None, use_vad: bool = False):
        """
        Initialization of the Markov Statistics estimator. Optionally a dataset can be given as input.
        The Estimator is then used with the given dataset.

        Args:
            dataset: dataset: dataset that should be processed
            use_vad: (optional) Set to True, when VAD data is
                present in the dataset and that data should be recognized for sampling
        """
        self._dataset = None
        self._model = None
        self._silence_distribution = None
        self._overlap_distribution = None

        if dataset is not None:
            self.fit(dataset, use_vad)

    def fit(self, dataset: [Dataset, Dict], use_vad: bool = False) -> None:
        """
        Iterates over the given dataset and computes the MarkovModel and the distributions for silence and overlap.
        The dataset, model and according distributions are then stored in the class.
        Overrides the previously fitted dataset.

        Args:
            dataset: dataset that should be processed
            use_vad: (optional) Set to True, when VAD data is
                present in the dataset and that data should be recognized for sampling
        """

        logger.info("Begin with processing the dataset")
        if len(dataset) == 0:
            raise AssertionError('Cannot compute statistics for an empty dataset.')

        # Make sure dataset has the type Dataset
        if type(dataset) is dict:
            dataset = from_dict(dataset)
        # Remove FilterExceptions
        self._dataset = dataset.catch()

        states_absolute_values = np.array([0] * 4)
        matrix_absolute_values = np.zeros((4, 4))

        silence_durations = []
        overlap_durations = []

        n = 0

        for sample in self._dataset:
            if n % 100 == 0:
                logger.info(f'Processed samples: {n}')
            n += 1

            # Depending on the usage of VAD data different keys are used
            if use_vad:
                offsets = sample['offset']['aligned_source']
                speaker_ends = list(map(add, offsets, sample['num_samples']['aligned_source']))
            else:
                offsets = sample['offset']['original_source']
                speaker_ends = list(map(add, offsets, sample['num_samples']['original_source']))
            speaker_ids = sample['speaker_id']

            current_state = 0
            last_foreground_end = speaker_ends[0]
            last_foreground_speaker = speaker_ids[0]

            for i in range(len(offsets) - 1):
                states_absolute_values[current_state] += 1
                # Turn-hold
                if last_foreground_speaker == speaker_ids[i + 1]:
                    if offsets[i + 1] - last_foreground_end < 0:
                        input()
                    new_state = 0
                    silence_durations.append(offsets[i + 1] - last_foreground_end)

                # Turn-switch
                elif last_foreground_end < offsets[i + 1]:
                    new_state = 1
                    silence_durations.append(offsets[i + 1] - last_foreground_end)

                # Overlap
                elif last_foreground_end < speaker_ends[i + 1]:
                    new_state = 2
                    overlap_durations.append(last_foreground_end - offsets[i + 1])
                # Backchannel
                else:
                    new_state = 3

                # Adjust foreground information, in all states except backchannel
                if new_state in {0, 1, 2}:
                    last_foreground_end = speaker_ends[i + 1]
                    last_foreground_speaker = speaker_ids[i + 1]

                matrix_absolute_values[current_state][new_state] += 1

                current_state = new_state

        # Add at least one sample to the durations, otherwise the DistributionModel can not be fitted
        if len(silence_durations) == 0:
            silence_durations = [0]
        if len(overlap_durations) == 0:
            overlap_durations = [0]

        # Fit silence and overlap distributions

        self._silence_distribution = DistributionModel(silence_durations)
        self._overlap_distribution = DistributionModel(overlap_durations)

        # Fixes matrix when some states are never reached, otherwise the resulting matrix is not a stochastic matrix,
        # but this required for the markov model. (Fixed problem: division by 0, leads to infinite values)
        for i in range(len(states_absolute_values)):
            if states_absolute_values[i] == 0:
                matrix_absolute_values[i] = np.zeros((1, len(matrix_absolute_values[i])))
                matrix_absolute_values[i][i] = 1
                states_absolute_values[i] = 1

        # Fit MarkovModel and create fitting SpeakerTransitionModel
        self._model = MultiSpeakerTransitionModel(MarkovModel(matrix_absolute_values / states_absolute_values[:, None],
                                                            state_names=["TH", "TS", "OV", "BC"]))

        logger.info("Finished processing the dataset")

    @property
    def model(self) -> MultiSpeakerTransitionModel:
        return self._model

    @property
    def silence_distribution(self) -> DistributionModel:
        return self._silence_distribution

    @property
    def overlap_distribution(self) -> DistributionModel:
        return self._overlap_distribution

    @property
    def dataset(self) -> Dataset:
        return self._dataset
