import contextlib
from collections import defaultdict

import paderbox as pb


def plot_mixture(ex, sample_rate=None, ax=None):
    if ax is None:
        from matplotlib import pyplot as plt
        _, ax = plt.subplots(1, 1)
    speech_activity = defaultdict(pb.array.interval.zeros)
    num_samples = pb.utils.nested.get_by_path(ex, 'num_samples.original_source', allow_early_stopping=True)
    for o, l, s, in zip(ex['offset']['original_source'], num_samples, ex['speaker_id']):
        speech_activity[s][o:o + l] = True

    pb.visualization.plot.activity(speech_activity, ax=ax)
    ax.axvline(ex['num_samples']['observation'])
    if sample_rate is not None:
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticks() / sample_rate)
