# Multi-purpose Multi-Speaker Mixture Signal Generator (MMS-MSG)

![GitHub Actions](https://github.com/fgnt/mms_msg/actions/workflows/pytest.yml/badge.svg)

---
**NOTE:**

This repository currently still is **under construction**.
We will update, extend and modify parts of the code within the next weeks.
Also, example notebooks for using MMS-MSG and example code to reproduce our 
Baseline model will be provided on a later date. 

---
MMS-MSG is a highly modular and flexible framework for the generation of speech mixtures.
It extends the code-base of the [SMS-WSJ]() database 
for mixture signal generation to be able to generate both meeting-sytle speech mixtures and 
mixture signals corresponding to classical spech mixture databases.



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
  that provides access to speaker identitites can be used to simulate meeting data. Any additional information like
  transcriptions are kept and can still be used
 * **Number of participants**:  The number of speakers per meeting can be freely chosen. Furthermore, it is possible to
 to set a range, so that meetigns with varying numbers of active speakers are generated.
 * **Activity distribution per speaker**: Aside from fully random sampling algorithms to sample the next active speaker of
   a meeting, we also provide an activity-based speaker sampling. Here, the activity distribution per speaker (i.e. the speech ratio of each speaker)
 can be freely specified. Over the course of the meeting, the activity distribution will converge to the desired ratio,
 so that the generation of highly asymmetric meetings (e.g. lecture situations) are possible to be generated
 * **Amount/distribution of silence and overlap**:
   The probability and length of silence and/or overlap can be freely chosen. Furthermore, the distribution from which to sample the silence
   also can be specified by the user
 * **Background noise**: We offer an easy framework to add external influences like background noise to your mixtures.
 Currently, a sampling for static background noise is implemented. The addition of more realistic environmental noises 
 (e.g. from WHAM!) is supported in theory. Sampling functions for this use-case will be implemented in the future. 
 * **Reverberation/Scaling**:

### Modular Design
The sampling process is modularized, so that many different scenarios can be created by slightly changing the sampling 
pipeline. We provide example classes to show how the single modules are used. If a scenario is not supported, 
new sampling modules can be easily implemented to adapt MMS-MSG to your requirements.

### On-demand data generation
The data simulation process of MMS-MSG is split into the parameter sampling and the actual data generation.
Therefore, we support on-demand data generation. In this way, only the source data and the meeting parameters need to 
be saved, allowing the simulation of various meeting scenarios while minimizing the required disk space.
However, we also support the offline generation of meeting data if saving them to the hard disk is required for your 
workflow.

### Generation of Classical Speech Mixture Scenarios
We provide code to  generate speech mixtures according to the specifications of currently used source separation databases, where
single utterances of multiple speakers either partially or fully overlap with each other.
By using MMS-MSG to generate training data for these databases, we offer a native support of dynamic mixing.

Supported speech mixture databases:
 * [WSJ0-2mix/WSJ0-3mix]()
 * [LibriMix]()
 * [SMS-WSJ]()
 * [Partially Overlapped WSJ]()

Planned:
 * [WHAM!]()
 * [WHAMR!]()




## Planned Features:
  * WHAM! background noise sampling
  * Sampling Rate Offset (SRO) utilities (see [paderwasn]())
  * Markov Model-based dialogue sampling (refer to [hitachi_conversation]()) 