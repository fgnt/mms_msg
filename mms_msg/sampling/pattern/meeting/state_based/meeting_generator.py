from mms_msg.sampling.pattern.meeting.state_based.dataset_statistics_estimation import MeetingStatisticsEstimatorMarkov
from mms_msg.sampling.pattern.meeting.state_based.weighted_meeting_sampler import WeightedMeetingSampler
from mms_msg.sampling.pattern.meeting.state_based.action_handler import DistributionActionHandler
from mms_msg.sampling.pattern.meeting.state_based.sampler import DistributionSilenceSampler, DistributionOverlapSampler
from lazy_dataset import Dataset
from typing import Dict, Type
import mms_msg


class MeetingGenerator:
    """
    Class that can be used to generate artificial meetings that have a silence, overlap and speaker transition
    distribution that is statistical close to a given source dataset.
    The samples that are used to generate the artificial data is from an input dataset
    that can be independent of the source dataset.

    This class used a Markov based model for the different transitions of the speakers and tries to balance
    the activity of all speakers in each meeting.

    Properties:
        model:   Transition Model
        silence: Silence Distribution
        overlap: Overlap Distribution
    """
    def __init__(self, estimator_class: Type = MeetingStatisticsEstimatorMarkov):
        """
        Initialize the Meeting Generator

        Args:
            estimator_class: Class that should be used to determine the statistics on the input dataset.
                The constructor must accept at least two parameters: dataset, use_vad
                Also must have the following properties: model, silence_distribution, overlap_distribution
        """
        self.model = None
        self.silence = None
        self.overlap = None

        self.estimator_class = estimator_class

    def fit(self, source_dataset: [Dict, Dataset], use_vad: [bool] = False) -> None:
        """ Estimates the speaker transitions and the overlap/silence distributions of a given dataset.
         This data can then be used in the generate function. Must be called at least once before calling generate.
         It is possible that given VAD is considered when estimating the distributions.

        Args:
            source_dataset: Dataset for which the statistics are estimated.
                This data can then be used to generate new meetings.
            use_vad: Should VAD data be used. When set to True the fit function uses VAD data,
                when estimating the distributions of the source dataset.

        Returns: None
        """
        db_sampler = self.estimator_class(dataset=source_dataset, use_vad=use_vad)

        self.model = db_sampler.model

        self.silence = DistributionSilenceSampler(distribution=db_sampler.silence_distribution)
        self.overlap = DistributionOverlapSampler(max_concurrent_spk=2, distribution=db_sampler.overlap_distribution)

    def generate(self, input_dataset: [Dict, Dataset], num_speakers: int = 2, duration: int = 960000,
                 use_vad: [bool] = False) -> Dataset:
        """Generate a dataset of artificial meeting, with sources from the input_dataset.
         The distribution of the generated dataset follows the last fitted distribution,
         so the fit method must be called at least once before calling this method.
         Also, can utilize VAD data.

        Args:
            input_dataset: Dataset from which the sources are drawn, that are used for generation new meetings
            num_speakers: Number of speakers the meetings in the generated datasets should have.
            duration: Duration that the newly generated examples should roughly have, can be slightly exceeded.
            use_vad: Should VAD data be used. When set to true VAD data is used,
                during generation of the new dataset and the output dataset hat also VAD information.

        Returns Output dataset, with as many entries as the input dataset has samples.
        """

        if self.model is None or self.silence is None or self.overlap is None:
            raise Exception('No dataset is fitted, you have to use the fit method first.')

        if self.model.num_speakers != num_speakers:
            try:
                self.model.change_num_speakers(num_speakers)
            except SystemError:
                print('Cannot change the number of speakers of the transition model.'
                      'It is possible that the generation fails for the desired number of speakers.')

        ds = mms_msg.sampling.source_composition.get_composition_dataset(input_dataset, num_speakers=num_speakers)
        ds = ds.map(mms_msg.sampling.environment.scaling.UniformScalingSampler())
        ds = ds.map(mms_msg.sampling.environment.noise.UniformSNRSampler(20.0, 30.0))

        return ds.map(WeightedMeetingSampler(transition_model=self.model, duration=duration,
                                             action_handler=DistributionActionHandler(overlap_sampler=self.overlap,
                                                                                      silence_sampler=self.silence),
                                             use_vad=use_vad)({'*': input_dataset}))
