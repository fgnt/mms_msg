from collections import defaultdict

import paderbox as pb


def plot_meeting(ex):
    with pb.visualization.axes_context(columns=1, figure_size=(10, 3)) as ac:
        speech_activity = defaultdict(pb.array.interval.zeros)
        num_samples = pb.utils.nested.get_by_path(ex, 'num_samples.original_source', allow_early_stopping=True)
        for o, l, s, in zip(ex['offset']['original_source'], num_samples, ex['speaker_id']):
            speech_activity[s][o:o + l] = True

        pb.visualization.plot.activity(speech_activity, ax=ac.new)
