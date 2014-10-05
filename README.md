# Warning #

This library is quite old. And its not supported anymore due to lack of time. 
Perhaps it's the first CUDA implementation of the ConvNets which was first hosted on MatlabCentral, then on Bitbucket. I've cloned it here just for the reference.

Overview
========

CudaCnn is a Convolutional neural networks library writen on C++/CUDA
with Matlab frontend. It is the new version of my old Mablab + CUDA [CNN
library][].

Current features are:

-   Training methods: Stochastic gradient, **Stochastic
    Levenberg-Marquardt**
-   Layers: Convolutional, Pooling (max, average), Fully-connected
-   Transfer functions: Linear, Tansig, Tansig\_mod (variance normalized
    version of tansig)
-   CUDA is optional, you can compile CPU version of library, however
    it's not optimized at all
-   All dependencies are optional, you can compile library with or
    without them:
    -   HDF5 for saving and loading network in this format
    -   Matlab libs if you want Matlab interface
    -   Boost needed for shared\_ptr in case you don't want to compile
        lib with C++0x support
    -   gtest for building tests

There's Matlab demo sctipts with GUI showing the training of ConvNet on
MNIST dataset.

Quick start for Matlab users
============================

If you just want to run the training or simulation, follow these simple
steps:

1. Clone the sources
--------------------

Very simple:

*git clone <https://github.com/sirotenko/cudacnn.git>*

This will clone the repository in current folder.

2. Generate build files
-----------------------

In order to generate build files you'll need [CMake cross-compile
tool][]. Ubuntu users just type *sudo apt-get install cmake*. For
those who's not familar with CMake installation of CMake GUI tool is
recomeded (for Windows it's installed by default)

Also you'll need some environment to process makefiles (e.g. Visual C++
on Windows or gcc on Ubuntu) installed. If you going to compile CUDA
version of library you should have [CUDA 4.2 Toolkit][] installed.

In the *cudacnn-public* folder, create folder *build*, go there and
type

*cmake -G <generator> -DCUDA\_COMPUTE\_CAPABILITIES=x.x ../ *

where <generator> is the environment which you want to use to build and
x.x is compute capability of your CUDA graphics card (2.0 for Fermi
cards (default), 3.0 for Kepler). If you build it without CUDA then you
don't need this option For example, if you run it on Windows with 64-bit
version of Visual Studio 10 and you have graphics card with Kepler GPU,
then the command will look like this:

*cmake -G “Visual Studio 10 Win64” -DCUDA\_COMPUTE\_CAPABILITIES=3.0
../ *

Please note that Visual Studio version should match to Matlab version.
That is, if you have 64-bit Matlab installed, you should use *-G
“Visual Studio 10 Win64”*, otherwise you should use *-G “Visual Studio
10”*.

Also note that previously I didn't mention about
DCUDA\_COMPUTE\_CAPABILITIES option, so by default CC was set to 2.0 and
you might got an error on Kepler devices: Cuda exception. Failed to
Propagate in CLayerCudainvalid device function File

Default settings are set to compile Matlab wrapper, use CUDA and not use
HDF5. If you want to compile without CUDA you can add *-D
WITH\_CUDA=FALSE* when running cmake. If everything is ok, you'll see
the messages that configuring and generating is done.

3. Compile the project
----------------------

In case of Windows/Visual Studio just open cudacnn.sln, make sure that
the configuration type is Release and build all. In case of Ubuntu type
*make* in build folder to build the project. You can also setup
Eclipse or NetBeans makefile-based project and build it from there.

After build is finished two files are created cudacnn.lib(so) and
cudacnnMex.<mexext>. Latter is an wrapper for Matlab and automatically
copied to m\_files/@cnn.

4. Run the training demo
------------------------

In *demos* folder there's *train\_mnist.m* script which runs
Convolutional neural network training on the MNIST handwritten digits
recognition dataset. Before run it you should download MNIST dataset
from here: <http://yann.lecun.com/exdb/mnist/> and put it to
cudacnn-public/data/MNIST folder.

If everything was ok, after running the train\_mnist.m in Matlab, you'll
see the GUI with training information:

Full training could take pretty much time (hours), so you can either
abort training by pressing the corresponding button (in this case all
training progress is saved) or you can set less number of training
samples (line 35) or less epochs (line 143). Usually MCR
(missclassification rate) drops to \~ 15% after 30 s on modern PC even
without CUDA, which could be enough to see the progress. The resulting
trained network is saved in *cnnet* variable, which can be saved on
disk for later use.

If you want to see how the network recognizes specific simbols from
dataset, you can use *mnist\_gui.m*. But before that you should save
your trained network into *demos* folder and name it
*mnist\_cnet.mat*. If you run the script you'll see the window like
this:

Specify the path to the MNIST images dataset
(*data/MNIST/t10k-images.idx3-ubyte*) and check *Autorecognition* to
see the preview of the handwritten digit and recognition result:

Use "\<" and "\>" buttons to change the images.

Quick start without Matlab
==========================

To be done...

How to train network with your own data
=======================================
Datareader paradigm
-------------------

Cudacnn library uses *datareader* concept to abstract training from
actual data. You may want to train network with different types of data
(images, audio, some other features). These datasets can be really huge,
so it would be impossible (or impractical) to load them alltogether in
memory and you need to load and preprocess them by parts. Datareader
helps intended to help with this.

Datareader is a Matlab struct, which should have the following fields:

-   *num\_samples* - total number of samples in dataset;
-   *read* - reference to a function which takes two inputs:
    datareader struct and sample number and outputs 3 variables: input
    sample, target value for sample, and datareader struct.

Example
-------

Suppose you simply have all your data already preprocessed in a Matlab
workspace and want to feed them in to a training function. For example,
there're cell arrays in your workspace, which define your training and
test datasets: *training\_inputs*, *training\_labels*,
*test\_inputs* and *test\_labels*. Then you create a simple
function: And add the following lines to your train script:

More advanced version of the datareader, which loads the data from file
to a buffer can be seen in [demos/mnist\_datareader.m][]

Limitations
===========

 - CUDA version of library only works with graphics cards of Compute capability >= 2.0
 - Lib was only tested on Windows 7 64-bit, Ubuntu 12.04, Ubuntu 11.04`

  [demos/mnist\_datareader.m]: https://bitbucket.org/intelligenceagent/cudacnn-public/src/b41b9d154ea651f2d7d12ad5af5493f6090894c2/demos/mnist_datareader.m?at=default
    "wikilink"

  [CNN library]: http://www.mathworks.com/matlabcentral/fileexchange/24291
    "wikilink"
  [here]: http://mercurial.selenic.com/downloads/ "wikilink"
  [1]: http://tortoisehg.bitbucket.org/download/index.html "wikilink"
  [CMake cross-compile tool]: http://www.cmake.org/cmake/resources/software.html
    "wikilink"
  [CUDA 4.2 Toolkit]: http://developer.nvidia.com/cuda/cuda-downloads
    "wikilink"
