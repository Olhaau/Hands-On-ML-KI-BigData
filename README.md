


# Hands-On  Machine Learning, Artificial Intelligence and Big Data
Overview of hands-on methods of machine learning (ML), artificial intelligence (AI), Big Data and its applications in official statistics using Scikit-Learn, Keras and TensorFlow.

## Table of Contents
<!--
TODO: Soll sich einklappen
-->

- [I. Fundamentals of Machine Learning](#i-fundamentals-of-machine-learning)
  * [6. Decision Trees](#6-decision-trees)
- [II. Neural Networks and Deep Learning](#ii-neural-networks-and-deep-learning)
  * [15. Processing Sequences Using RNNs and CNNs](#15-processing-sequences-using-rnns-and-cnns)
    + [Forecasting a Time Series](#forecasting-a-time-series)
      - [Forecasting a single Time Series](#forecasting-a-single-time-series)
      - [Baseline Metrics](#baseline-metrics)
      - [Simple RNN](#simple-rnn)
      - [Deep RNNs](#deep-rnns)
      - [Forecasting Several Time Steps Ahead](#forecasting-several-time-steps-ahead)
      - [Handling Long Sequences](#handling-long-sequences)
      - [Fighting the Unstable Gradients Problem](#fighting-the-unstable-gradients-problem)
      - [LSTM](#lstm)
- [III. Big Data Processing](#iii-big-data-processing)
  * [Optimize Data Processing On One Device](#optimize-data-processing-on-one-device)
  * [Configurate A Virtual Machine Cluster On A Single Host](#configurate-a-virtual-machine-cluster-on-a-single-host)
  * [Test The Application Of Cloud Computing Services](#test-the-application-of-cloud-computing-services)
  * [Save And Process Data Remotely On A Server](#save-and-process-data-remotely-on-a-server)
  * [Build A Big Data Laboratory Using Multiple Hosts](#build-a-big-data-laboratory-using-multiple-hosts)
- [IV. Applications In Official Statistics](#iv-applications-in-official-statistics)
- [Resources](#resources)

## I. Fundamentals of Machine Learning
...

### 6. Decision Trees
Decision Tree Classifier from Scratch (Google Developers): 

<a align ="center" href="https://www.youtube.com/watch?v=LDRbO9a6XPU" title="Link Title"><img src="https://img.youtube.com/vi/LDRbO9a6XPU/0.jpg" alt="Alternate Text" /></a>

## II. Neural Networks and Deep Learning
Introduction:
* A simple neural network in 9 lines of Python code can be found [here](https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1).
* With Neural Networks can be tinkered in the [TensorFlow playground](http://playground.tensorflow.org/#activation=sigmoid&batchSize=10&dataset=xor&regDataset=reg-plane&learningRate=1&regularizationRate=0&noise=20&networkShape=2,2,2,2&seed=0.88379&showTestData=false&discretize=false&percTrainData=70&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false).

...

### 15. Processing Sequences Using RNNs and CNNs
See https://github.com/ageron/handson-ml2/blob/master/15_processing_sequences_using_rnns_and_cnns.ipynb


#### Forecasting a Time Series

##### Forecasting a single Time Series
A single (time-series) can be forecasted using a stacked LSTM model in https://github.com/Olhaau/Hands-On-ML-KI-BigData/blob/main/forecast_one_timeseries_tool.ipynb, e.g.:

<p align="center">
  <img width="600"  src="https://github.com/Olhaau/Hands-On-ML-KI-BigData/blob/main/forecast_timeseries_4.png">
</p>


##### Baseline Metrics
We use the results of an example in Géron, 2019 p. 503ff: 10000 timeseries, where each series is the sum of two sine waves of fixed amplitudes but random frequencies and phases, plus a bit of noise. Before using RNNs, it is often a good idea to calculate the error (e.g. MSE) of some baseline estimations:
* naive forecasting (predict the last value of the series), MSE in example: 0.020
* fully connected network (e.g. linear regression), MSE in exampe: 0.004
...

##### Simple RNN
MSE in example: 0.014

##### Deep RNNs
MSE in example: 0.003

##### Forecasting Several Time Steps Ahead
...

##### Handling Long Sequences
...

##### Fighting the Unstable Gradients Problem
...

##### LSTM
...



## III. Big Data Processing
A overview of tasks to gain hands-on experience with Big Data infrastructures.

### Optimize Data Processing On One Device
- [x] use the GPU-support of TensorFlow, see https://www.tensorflow.org/install/gpu
- [x] optimize performance with TensorFlow's tf.data API, see https://www.tensorflow.org/guide/data_performance  
- [x] manage massive matrices with shared memory and memory-mapped files (e.g. bigmemory package for R), see https://cran.r-project.org/web/packages/bigmemory/bigmemory.pdf

### Configurate A Virtual Machine Cluster On A Single Host
- [ ] configurate multiple connected vitual machines e.g. Ubuntu over VirtualBox

<b> Remarks and Resources </b>
- remark: only for personal experience and just limited advantage in processing
- see https://docs.vmware.com/en/VMware-vSphere/6.7/com.vmware.vsphere.mscs.doc/GUID-01B4B067-9AAC-41C0-BF9B-1D085F36DF51.html 

### Test The Application Of Cloud Computing Services 
- [ ] process data on e.g. MS Azure, AWS (Amazon Web Services), Google Cloud, IBM Cloud, Oracle Cloud Infrastructure, CloudLinux

<b> Remarks and Resources </b>
- a comparison of services can be found here: https://www.techradar.com/best/best-cloud-computing-services

### Save And Process Data Remotely On A Server
- [ ] configurate e.g. a single Raspberry Pi server

<b> Remarks and Resources </b>
- see https://www.elektronik-kompendium.de/sites/raspberry-pi/2002251.htm

### Build A Big Data Laboratory Using Multiple Hosts 
- [ ] configurate a host cluster (e.g. a 3+ Raspberry Pi Cluster)
- [ ] manage cluster computing using a Hadoop and Spark framework

<b> Remarks and Resources </b>
* see https://towardsdatascience.com/assembling-a-personal-data-science-big-data-laboratory-in-a-raspberry-pi-4-or-vms-cluster-e4c5a0473025


## IV. Applications In Official Statistics
![Table of applications in official statistics](https://github.com/Olhaau/Hands-On-ML-KI-BigData/blob/main/ML_Applications_in_OS.csv) (csv), which can be found in the meta analysis of <a href=https://arxiv.org/abs/1812.10422> Beck, Dumpert, Feuerhake,  (2018)</a>. Overview of the applications and used machine learning methods in official statistics:

<p align="center">
  <img width="600"  src="https://github.com/Olhaau/Hands-On-ML-KI-BigData/blob/main/9_heatmap_ml_methods_and_applications.png">
</p>

## Resources
1. Beck, M., Dumpert, F., & Feuerhake, J. (2018). Machine Learning in Official Statistics. <i>
<a href=https://arxiv.org/abs/1812.10422> arXiv preprint  1812.10422</a>
</i>.
2. Géron, A. (2019). <i> Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow: Concepts, tools, and techniques to build intelligent systems</i>. O'Reilly Media

<!--
TODO:
-> basic formatting
https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax
-> table of content generator: 
https://ecotrust-canada.github.io/markdown-toc/

-> markdown for video preview
[![decision tree from scratch](https://img.youtube.com/vi/LDRbO9a6XPU/0.jpg)](https://www.youtube.com/watch?v=LDRbO9a6XPU)

-->
