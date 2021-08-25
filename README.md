


Hands-On  Machine Learning, Artificial Intelligence and Big Data
================================================================
Overview of hands-on methods of machine learning (ML), artificial intelligence (AI), Big Data and its applications in official statistics using Scikit-Learn, Keras and TensorFlow. Great resources:

* https://github.com/ageron/handson-ml2/blob/master/15_processing_sequences_using_rnns_and_cnns.ipynb
* ...


# Table of Contents
- [I. Fundamentals of Machine Learning](#i-fundamentals-of-machine-learning)
  * [6. Decision Trees](#6-decision-trees)
- [II. Neural Networks and Deep Learning](#ii-neural-networks-and-deep-learning)
  * [15. Processing Sequences Using RNNs and CNNs](#15-processing-sequences-using-rnns-and-cnns)
    + [Description of RNNs](#description-of-rnns)
    + [Forecasting a Time Series](#forecasting-a-time-series)
      - [Forecasting a single Time Series](#forecasting-a-single-time-series)
      - [Model Comparison](#model-comparison)
    + [Handling Long Sequences](#handling-long-sequences)
      - [Fighting the Unstable Gradients Problem](#fighting-the-unstable-gradients-problem)
      - [Tackling the Short-Term Memory Problem](#tackling-the-short-term-memory-problem)
- [III. Big Data Processing](#iii-big-data-processing)
  * [Optimize Data Processing On One Device](#optimize-data-processing-on-one-device)
  * [Configurate A Virtual Machine Cluster On A Single Host](#configurate-a-virtual-machine-cluster-on-a-single-host)
  * [Test The Application Of Cloud Computing Services](#test-the-application-of-cloud-computing-services)
  * [Save And Process Data Remotely On A Server](#save-and-process-data-remotely-on-a-server)
  * [Build A Big Data Laboratory Using Multiple Hosts](#build-a-big-data-laboratory-using-multiple-hosts)
- [IV. Applications In Official Statistics](#iv-applications-in-official-statistics)
- [Resources](#resources)


# I. Fundamentals of Machine Learning
...

## 6. Decision Trees

Introduction:
<details><summary>Decision Tree Classifier from Scratch (Google Developers)</summary>
<p>

<a align ="center" href="https://www.youtube.com/watch?v=LDRbO9a6XPU" title="Link Title"><img src="https://img.youtube.com/vi/LDRbO9a6XPU/0.jpg" alt="Alternate Text" /></a>

</p>
</details>




# II. Neural Networks and Deep Learning
Introduction:
* A simple neural network in 9 lines of Python code can be found [here](https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1).
* With Neural Networks can be tinkered in the [TensorFlow playground](http://playground.tensorflow.org/#activation=sigmoid&batchSize=10&dataset=xor&regDataset=reg-plane&learningRate=1&regularizationRate=0&noise=20&networkShape=2,2,2,2&seed=0.88379&showTestData=false&discretize=false&percTrainData=70&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false).

...

## 15. Processing Sequences Using RNNs and CNNs

* **RNN** - recurrent neural network
* **CNN** - convolutional neural network

Introduction:
* A simple implementation of a RNN only using Numpy can be found [here](https://ngrayluna.github.io/post/rnn_wnumpy/)

### Description of RNNs

<p align="center">
  <img width="400"  src="https://www.researchgate.net/profile/Weijiang-Feng/publication/318332317/figure/fig1/AS:614309562437664@1523474221928/The-standard-RNN-and-unfolded-RNN.png">
</p>
<p align="center"> 
 Resource: 
 <a href="https://www.researchgate.net/publication/318332317_Audio_visual_speech_recognition_with_multimodal_recurrent_neural_networks">Feng et al., 2017, Fig. 1</a>
</p>


* **RNNs** are
  * similar to feedforward neural networks, except it has connections pointed backward
  * trained by **backpropagation through time (BPTT)**, which unrolls it through time and uses regular backpropagation 
* **Applications**: inputs can be sequences of arbitrary lengths (e.g. sentences, documents, audio samples; rather than fixed-sized inputs for most other nets), which lets RNNs very versatile handle e.g. 
  * analysis and prediction of time series, 
  * anticipating car trajectories in autonomous driving systems, 
  * automatic translation, 
  * converting speech-to-text
* **Difficulties**:
  * **unstable gradients**, which can be alleviated using e.g. recurrent dropout or recurrent layer normalization
  * **limited short-term memory** (~10 steps backward), which can be extendet using LSTM and GRU cells (~100 steps)
* **Alternatives**: also other neural networks can handle sequential data, e.g.:
  * for small sequences, a **regular dense networks** can work
  * for long sequences (e.g. audio samples or text), **CNNs** work quite well (e.g. WaveNet)  
<details><summary>show more theoretical details </summary>
<p>

 **Recurrent Neurons and Layers**
 
* **feedforward neural networks** - activations flow only in one direction, from the input to the output layer
* RNN are similar, except it has connections pointing backward
* simplest example: a network composed of only one neuron receiving inputs, produce an output and sending that output to itself
* **unrolling the network through time** - represent the network once per time step

**Memory Cell (or simply Cell)** - part of a neural network that preserves some state across time steps
* a single recurrent neuron or a layer of recurrent neurons, is a very basic cell capable of learning only short patterns (typically ~10 steps, depending on the task) since its output at time step *t* is a function of all the inputs from previous time steps
* more complex cells are capable of learning longer pattern (~100 steps, depending on the task)
* a cell's state at time step **h** (for **hidden**) is a function of some inputs at that time and its  state at a previous time step 

**Possible Inputs and Outputs**
 
* **sequence-to-sequence** - e.g. inputs are the values of a time series over the last *N* days until today, outputs are the values from *N*-1 days ago to tomorrow
* **sequence-to-vector** 
* **vector-to-sequence**
* **encoder** (sequence-to-vector) followed by a **decoder** (vector-to-sequence) - this could be used for translating a sentence from one language to another: 
  * the encoder converts sentence into a single vector representation 
  * the decoder would convert this vector into a sentence in another language

**Training RNNs**
 
A RNN is trained by **backpropagation through time (BPTT)**, which unrolls it through time and uses regular backpropagation. In detail:
1. first a forward pass through the unrolled network
2. the output sequence is evaluated using a cost function (may ignore some outputs)
3. the gradients of that cost function are then propagated backward through the unrolled network
4. model parameters are updated using the computed gradients (the gradients flow backward through alle the outputs used by the cost function, not just the final output)
 </p>
</details>
 
### Forecasting a Time Series
**Time series** - sequence of one (called univariate) or more (called multivariate) values per time step

Typical tasks are:  
* **forecasting** - predict future values 
* **imputation**  - predict (or rather "postdict") missing values from the past

#### Forecasting a single Time Series
A single (time-series) can be forecasted using a stacked LSTM model in https://github.com/Olhaau/Hands-On-ML-KI-BigData/blob/main/forecast_one_timeseries_tool.ipynb, e.g.:

<p align="center">
  <img width="600"  src="https://github.com/Olhaau/Hands-On-ML-KI-BigData/blob/main/forecast_timeseries_4.png">
</p>


####  Model Comparison
We use the results of an example in Géron, 2019, p. 503 ff: 10000 timeseries, where each series is the sum of two sine waves of fixed amplitudes but random frequencies and phases, plus a bit of noise, see https://github.com/ageron/handson-ml2/blob/master/15_processing_sequences_using_rnns_and_cnns.ipynb. 

<details><summary>Code to create a training, validation and test set</summary>
 <p>
  
```python
n_Steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]
```
  
</p>
</details>

Note that the input features are generally represented as 3D arrays of shape:   
> \[batch size = number of time series, time steps (of the time series), dimensonality (of the input)\]

1. **Baseline Metrics** - before using RNNs, it is a good idea to calculate the error (e.g. MSE) of some baseline estimations, e.g.
   1. **naive forecasting** (predict the last value in each series), MSE = 0.020<br/>
      <details><summary>show code</summary>
      <p>
 
      ```python
      y_pred = X_valid[:, -1]
      np.mean(keras.losses.mean_squared_error(y_valid, y_pred))
      ```
 
      </p>
      </details>  
   2. **fully connected network** (e.g. linear regression), MSE = 0.004<br/>
      <details><summary>show code</summary>
      <p>
     
      ```python
      model = keras.models.Sequential([
              keras.layers.Flatten(input_shape=[50, 1]),
              keras.layers.Dense(1)
      ])

      model.compile(loss="mse", optimizer="adam")
      history = model.fit(X_train, y_train, epochs=20,
                          validation_data=(X_valid, y_valid))
                    
      model.evaluate(X_valid, y_valid)               
      ```
    
     </p>
     </details>

2. **Simple RNN**, MSE = 0.014  
   <details><summary>show code</summary>
   <p>
 
   ```python
   model = keras.models.Sequential([
           keras.layers.SimpleRNN(1, input_shape=[None, 1])
           ])

   optimizer = keras.optimizers.Adam(lr=0.005)
   model.compile(loss="mse", optimizer=optimizer)
   history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))

   model.evaluate(X_valid, y_valid)
   ```
 
   </p>
   </details>
   
3. **Deep RNNs**, MSE = 0.003
   <details><summary>show code</summary>
   <p>
 
   ```python
   model = keras.models.Sequential([
           keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
           keras.layers.SimpleRNN(20, return_sequences=True),
           keras.layers.SimpleRNN(1)
    ])

    model.compile(loss="mse", optimizer="adam")
    history = model.fit(X_train, y_train, epochs=20,
                        validation_data=(X_valid, y_valid))

    model.evaluate(X_valid, y_valid)
    ```
    </p>
    </details>
 
**Forecasting Several Time Steps Ahead**

In the example, we will predict *n* = 10 steps ahead.
<details><summary>regenerate the sequences with 9 more time steps</summary>
<p> 
 
```python
n_steps = 50
series = generate_time_series(10000, n_steps + 10)
X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
X_test, Y_test = series[9000:, :n_steps], series[9000:, -10:, 0]
```
 </p>
</details>

**Baseline:**
* **naive** (constant in the next *n* steps): MSE = 0.223
* **linear model**: MSE = 0.0188


**Possible implementations:**
* predict iteratively single next values (erros might accumulate), MSE = 0.029
  <details><summary>show code</summary>
  <p>
   
  ```python
  X = X_valid
  for step_ahead in range(10):
      y_pred_one = model.predict(X)[:, np.newaxis, :]
      X = np.concatenate([X, y_pred_one], axis=1)

  Y_pred = X[:, n_steps:, 0]
  
  ```
     
  </p>
  </details>
  
* train a new RNN to predict all next *n* values at once, MSE = 0.008
  <details><summary>show code</summary>
  <p>
   
  ```python
  model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(10)
  ])

  model.compile(loss="mse", optimizer="adam")
  history = model.fit(X_train, Y_train, epochs=20,
                    validation_data=(X_valid, Y_valid))
  
  series = generate_time_series(1, 50 + 10)
  X_new, Y_new = series[:, :50, :], series[:, -10:, :]
  Y_pred = model.predict(X_new)[..., np.newaxis] 
  ```
   
  </p>
  </details>
...

### Handling Long Sequences

#### Fighting the Unstable Gradients Problem
Problem: ...  
Solutions:
* ...
* ...

#### Tackling the Short-Term Memory Problem
Problem: ...  
Solution: LSTM, GRU



# III. Big Data Processing
A overview of tasks to gain hands-on experience with Big Data infrastructures.

## Optimize Data Processing On One Device
- [x] use the GPU-support of TensorFlow, see https://www.tensorflow.org/install/gpu
- [x] optimize performance with TensorFlow's tf.data API, see https://www.tensorflow.org/guide/data_performance  
- [x] manage massive matrices with shared memory and memory-mapped files (e.g. bigmemory package for R), see https://cran.r-project.org/web/packages/bigmemory/bigmemory.pdf

## Configurate A Virtual Machine Cluster On A Single Host
- [ ] configurate multiple connected vitual machines e.g. Ubuntu over VirtualBox

<b> Remarks and Resources </b>
- remark: only for personal experience and just limited advantage in processing
- see https://docs.vmware.com/en/VMware-vSphere/6.7/com.vmware.vsphere.mscs.doc/GUID-01B4B067-9AAC-41C0-BF9B-1D085F36DF51.html 

## Test The Application Of Cloud Computing Services 
- [ ] process data on e.g. MS Azure, AWS (Amazon Web Services), Google Cloud, IBM Cloud, Oracle Cloud Infrastructure, CloudLinux

<b> Remarks and Resources </b>
- a comparison of services can be found here: https://www.techradar.com/best/best-cloud-computing-services

## Save And Process Data Remotely On A Server
- [ ] configurate e.g. a single Raspberry Pi server

<b> Remarks and Resources </b>
- see https://www.elektronik-kompendium.de/sites/raspberry-pi/2002251.htm

## Build A Big Data Laboratory Using Multiple Hosts 
- [ ] configurate a host cluster (e.g. a 3+ Raspberry Pi Cluster)
- [ ] manage cluster computing using a Hadoop and Spark framework

<b> Remarks and Resources </b>
* see https://towardsdatascience.com/assembling-a-personal-data-science-big-data-laboratory-in-a-raspberry-pi-4-or-vms-cluster-e4c5a0473025


# IV. Applications In Official Statistics
![Table of applications in official statistics](https://github.com/Olhaau/Hands-On-ML-KI-BigData/blob/main/ML_Applications_in_OS.csv) (csv), which can be found in the meta analysis of <a href=https://arxiv.org/abs/1812.10422> Beck, Dumpert, Feuerhake,  (2018)</a>. Overview of the applications and used machine learning methods in official statistics:

<p align="center">
  <img width="600"  src="https://github.com/Olhaau/Hands-On-ML-KI-BigData/blob/main/9_heatmap_ml_methods_and_applications.png">
</p>

# Resources
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

https://guides.github.com/features/mastering-markdown/

-> markdown for video preview
[![decision tree from scratch](https://img.youtube.com/vi/LDRbO9a6XPU/0.jpg)](https://www.youtube.com/watch?v=LDRbO9a6XPU)

-->
