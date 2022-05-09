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