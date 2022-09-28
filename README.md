# Multi-purpose Multi-Speaker Mixture Signal Generator (MMS-MSG)

![GitHub Actions](https://github.com/fgnt/mms_msg/actions/workflows/pytest.yml/badge.svg)


MMS-MSG is a highly modular and flexible framework for the generation of speech mixtures.
It extends the code-base of the [SMS-WSJ](https://github.com/fgnt/sms_wsj) database 
for mixture signal generation to be able to generate both meeting-style speech mixtures and 
mixture signals corresponding to classical speech mixture databases.



## What is the purpose of MMS-MSG?
Meeting data describes a highly dynamic setting. Both the environment
With MMS-MSG, we don't aim to provide a single, new database.

Instead,  we want to provide an adaptable framework that allows the prototyping and evaluation of meeting 
procesing and transcription system in as many environments as possible.


## Features

### Generation of Meeting data
The core aspect of MMS-MSG is the generation of meeting-style data. The meetings are generated in a modular fashion.
Adjustable parameters are:
 * **Source database** (e.g. WSJ, LibriSpeech): Any audio database consisting of clean, single-speaker utterances 
  that provides access to speaker identities can be used to simulate meeting data. Any additional information like
  transcriptions are kept and can still be used.
 * **Number of participants**:  The number of speakers per meeting can be freely chosen. Furthermore, it is possible
 to set a range, so that meetings with varying numbers of active speakers are generated.
 * **Activity distribution per speaker**: Aside from fully random sampling algorithms to sample the next active speaker of
   a meeting, we also provide an activity-based speaker sampling. Here, the activity distribution per speaker (i.e. the speech ratio of each speaker)
 can be freely specified. Over the course of the meeting, the activity distribution will converge to the desired ratio,
 so that the generation of highly asymmetric meetings (e.g. lecture situations) are possible to be generated
 * **Amount & distribution of silence/overlap**:
   The probability and length of silence and/or overlap between consecutive utterances of a meeting 
   can be freely chosen. Furthermore, the distribution from which to sample the silence
   also can be specified by the user.
 * **Background noise**: We offer an easy framework to add external influences like background noise to your mixtures.
 Currently, a sampling for static background noise is implemented. The addition of more realistic environmental noises 
 (e.g. from WHAM!) is supported in theory. Sampling functions for this use-case will be implemented in the future. 
 * **Reverberation/Scaling**:
 MMS-MSG natively supports the simulation of reverberated meetings. Here, any additional database that provides room 
 impulse responses can be used to reverberate the utterances of each speaker.
 While the currently implemented modules only support static speaker positions,
 speakers can theoretically change their position for each utterance.

### Modular Design
The sampling process is modularized, so that many scenarios can be created by slightly changing the sampling 
pipeline. We provide example classes to show how the single modules are used. If a scenario is not supported, 
new sampling modules can be easily implemented to adapt MMS-MSG to your requirements.

### On-demand data generation
The data simulation process of MMS-MSG is split into the parameter sampling and the actual data generation.
Through this, we support on-demand data generation. In this way, only the source data and the meeting parameters need to 
be saved, allowing the simulation of various meeting scenarios while minimizing the required disk space.
However, we also support the offline generation of meeting data if saving them to the hard disk is required for your 
workflow.

### Generation of Classical Speech Mixture Scenarios
We provide code to  generate speech mixtures according to the specifications of currently used source separation databases, where
single utterances of multiple speakers either partially or fully overlap with each other.
By using MMS-MSG to generate training data for these databases, we offer a native support of dynamic mixing.

Supported speech mixture databases:
 * [WSJ0-2mix/WSJ0-3mix](https://arxiv.org/abs/1508.04306)
 * [LibriMix](https://arxiv.org/abs/2005.11262)
 * [SMS-WSJ](https://arxiv.org/abs/1910.13934)
 * [Partially Overlapped WSJ](https://www.microsoft.com/en-us/research/uploads/prod/2018/04/ICASSP2018-5ad5f1b79bd16.pdf)

Planned:
 * [WHAM! & WHAMR!](https://wham.whisper.ai/)

## Using Generated Mixtures

The mixture generator uses [lazy_dataset](https://github.com/fgnt/lazy_dataset).
While the core functionality of mms_msg can be used without lazy_dataset, some features (like dynamic mixing and the database abstraction) are not available then.

```python
from mms_msg.databases.classical.full_overlap import WSJ2Mix
from mms_msg.sampling.utils import collate_fn
db = WSJ2Mix()

# Get a train dataset with dynamic mixing
# This dataset only emits the metadata of the mixtures, it doesn't load
# the data yet
ds = db.get_dataset('train_si284_rng')

# The data can be loaded by mapping a database's load_example function
ds = ds.map(db.load_example)  

# Other dataset modifications (see lazy_dataset doc)
ds = ds.shuffle(reshuffle=True)
ds = ds.batch(batch_size=8).map(collate_fn)
# ...

# Parallelize data loading with lazy_dataset
ds = ds.prefetch(num_workers=8, buffer_size=16)

# The dataset can now be used in any training loop
for example in ds:
    # ... do fancy stuff with the example.
    # The loaded audio data is in example['audio_data']
    print(example)
```

Any other data modification routines can be mapped to `ds` directly after loading the example.

### Using the torch DataLoader

A `lazy_dataset.Dataset` can be plugged into a `torch.utils.data.DataLoader`:

```python
from mms_msg.databases.classical.full_overlap import WSJ2Mix
db = WSJ2Mix()
ds = db.get_dataset('train_si284_rng').map(db.load_example)  

# Parallelize data loading with torch.utils.data.DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=8)

for example in loader:
    print(example)
```

## Planned Features:
  * WHAM! background noise sampling
  * ~~Sampling Rate Offset (SRO) utilities (see [paderwasn](https://github.com/fgnt/paderwasn))~~
  * Markov Model-based dialogue sampling (refer to [this paper](https://arxiv.org/abs/2204.11232))
---
**NOTE:**
Example recipes to reproduce our baseline results are still under construction and will be provided at a later date.
---
## Extending MMS-MSG

### Example structure

The input examples should have this structure:

```python
example = {
    'audio_path': {
        'observation': 'single_speaker_recording.wav'
    },
    'speaker_id': 'A',
    'num_samples': 1234,  # Number of samples of the observation file
    # 'num_samples': {'observation': 1234} # Alernative, if other audios are present
    'dataset': 'test',  # The input dataset name
    'example_id': 'asdf1234', # Unique ID of this example. Optional if the input data is passes as a dict
    'scenario': 'cafe-asdf1234',  # (Optional) If provided, mms_msg makes sure that all examples of the same speaker in a mixture share the same scenario
    # ... (any additional keys)
}
```

After selecting utterances for a mixture, these utterance examples are normalized and "collated", which results in a structure similar to this:

```python
example = {
    'audio_path': {
        'original_source': [
            'source1.wav',
            'source2.wav',
        ],
    },
    'speaker_id': [
        'A', 'B'  
    ],
    'num_samples': {  # The structure under some keys mirrors the structure in 'audio_path'
        'original_source': [
          1234, 4321
        ]
    },
    'source_id': [  # Reference to the source examples this mixture was created from
        'asdf1234', 'asdf1235'
    ],
    ...
}
```

Starting from such a structure, sampling modules can be applied to fill the example with more information, e.g.,
offsets or scaling of the utterances.

### Creating a custom database from existing sampling modules

Database classes or definitions are provided for a few common scenarios in `mms_msg.databases`.
Each database class has to define two methods:

 - `get_mixture_dataset`, which encapsulates the "sampling" stage and builds a pipeline of sampling modules, and
 - `load_example`, which provides the "simulation" stage, i.e., loading and mixing the audio data.

A basic (parameter-free) database would look like this:

```python
from mms_msg.databases.database import MMSMSGDatabase
from lazy_dataset.database import JsonDatabase
import mms_msg

class MyDatabase(JsonDatabase, MMSMSGDatabase):
    def get_mixture_dataset(self, name, rng):
        ds = mms_msg.sampling.source_composition.get_composition_dataset(
            input_dataset=super().get_dataset(name),
            num_speakers=2,
            rng=rng,
        )
        ds = ds.map(mms_msg.sampling.pattern.classical.ConstantOffsetSampler(8000))
        ds = ds.map(mms_msg.sampling.environment.scaling.ConstantScalingSampler(0))
        return ds

    def load_example(self, example):
        return mms_msg.simulation.anechoic.anechoic_scenario_map_fn(example)
```
and can be instantiated with
```python
db = MyDatabase('path/to/source/database.json')
```

The structure of the dataset sampling pipeline is described in the next section.

### Pipeline structure

This is an example of a simple sampling pipeline for a single dataset:

```python
import mms_msg

input_ds = ...  # Get source utterance examples from somewhere

# Compute a composition of base examples. This makes sure that the speaker distribution
# in the mixtures is equal to the speaker distribution in the original database.
ds = mms_msg.sampling.source_composition.get_composition_dataset(input_dataset=input_ds, num_speakers=2)

# If required: Offset the utterances
ds = ds.map(mms_msg.sampling.pattern.classical.ConstantOffsetSampler(0))

# If required: Add log_weights to simulate volume differences
ds = ds.map(mms_msg.sampling.environment.scaling.UniformScalingSampler(max_weight=5))

```

The sampling process always starts with the creation of a "source composition", i.e., sampling (base) utterances for each mixture.
This is done in `get_composition_dataset`, which implements a sampling algorithm similar to SMS-WSJ that uses each utterance from 
the source database equally often.

After this, sampling modules can be applied to simulate different speaking patterns or environments.
The example above sets all offsets to zero (i.e., all utterances start at the beginning of the mixture) with the `ConstantOffsetSampler` 
and samples a random scale with a maximum of 5dB with the `UniformScalingSampler`.

Many other sampling modules are available, including one that simulates meeting style speaking patterns.
Examples for this can be found [in this notebook](notebooks/mixture_generator.ipynb).


### Writing a custom sampling module

Mixtures in `mms_msg` are created by applying individual sampling modules to an example one
after the other. Each sampling module is fully deterministic, i.e., its output only depends
on its hyperparameters and the input example, but is not allowed to maintain a mutable state.
This is to ensure reproducibility: The sampling does not depend on the order in which the mixtures are generated, 
the number or order in which the modules are applied.

A sampling module is a callable that receives an (intermediate) mixture as a dictionary, modifies it, and returns it.
A basic sampling module, implemented as a function without hyperparameters, could look like this:
```python
import mms_msg
def my_sampling_module(example: dict) -> dict:
    # Get a deterministic random number generator based on the input example
    # and an additional seed string. The seed string ensures that the RNGs
    # differ between different sampling modules
    rng = mms_msg.sampling.utils.rng.get_rng_example(example, 'my_sampler')

    # Sample whatever based on RNG and possibly the contents of example
    example['my_random_number'] = rng.random()
    return example
```

An important part is the `mms_msg.sampling.utils.rng.get_rng_example` function.
It returns a `np.random.Generator` object that is initialized with a seed computed from basic information from the
example dictionary (example-ID and dataset) and an additional seed string.
This means that the random numbers generated in a module are equal every time the module is applied to the same input example.

If your sampling module has hyperparameters, we recommend a frozen dataclass to ensure immutability:
```python
import mms_msg
from dataclasses import dataclass

@dataclass(frozen=True)
class MySamplingModule:
    size: int = 42
    
    def __call__(self, example):
        rng = mms_msg.sampling.utils.rng.get_rng_example(example, 'my_sampler')

        # Sample whatever based on RNG and possibly the contents of example
        example['my_random_number'] = rng.random(self.size)
        return example
```

A more practical example is given [in this notebook](notebooks/extending_mms_msg.ipynb).

## Cite
MMS-MSG was proposed in the following publication:
```bibtex
@inproceedings{cordlandwehr2022mms_msg,
title={MMS-MSG: A Multi-purpose Multi-Speaker Mixture Signal Generator},
author={Tobias Cord-Landwehr and Thilo von Neumann and Christoph Boeddeker and Reinhold Haeb-Umbach},
year={2022},
booktitle={International Workshop on Acoustic Signal Enhancement (IWAENC)},
publisher = {{IEEE}},
}```