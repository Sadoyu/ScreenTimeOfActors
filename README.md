# ScreenTimeOfActors
An application to predict the screen time of actors in a movie


  Screen time is extremely important for an actor. It is directly related to the money
he/she gets. So determining the total screen time of the actors allows us to know the total time
spent by the actor on screen, thereby determining how much the film success is dependent on
the screen time of the actor.

  For understanding how this software works, consider a flip book. We have a different
image on each page of the book, and as we flip these pages, we get an animation of an object.
This corresponds to a visualization of video. The visualization gets better the faster we flip
the pages. In other words, this visual is a collection of different images arranged in a
particular order.

  Similarly, videos are nothing but a collection of a set of images. These images are
called frames and can be combined to get the original video. So, a problem related to video
data is not that different from an image classification or an object detection problem. There is
just one extra step of extracting frames from the video.

  The following are the steps we can follow to achieve our goal: Import and read the
video, extract frames from it, and save them as images. Label a few images for training the
model. Build our model on training data. Make predictions for the remaining images.

Calculate the screen time of each and every actor

We can determine the screen time of the actor by the following formula:

Screen Time = Number of Frames the actor is present in / Frame rate of video

Area: Soft Computing
Languages Used: Python






Download the project and run the test.py file.

The project is built using the following packages, we recommend using the same for the first run. Also make sure you are installing the following packages in the same order as installation of tensorflow caused problems with the installation of the other python libraries.

Python 3.6

Anaconda

CUDA Toolkit 8.0 Ensure you append the relevant Cuda pathnames to the %PATH% environment variable as defined in the NVIDIA documentation.

Nvidia Drivers for CUDA Toolkit 8.0

cuDNN v5.1. Note that cuDNN is typically installed in a different location from the other CUDA DLLs. Ensure that you add the directory where you installed the cuDNN DLL to your %PATH% environment variables

Tensorflow-GPU - 1.2.0

For support in installation of CUDA, cuDNN and Tensorflow - https://www.youtube.com/watch?v=Ebo8BklTtmc

Pip - 19

Python Libraries - 

Numpy

Pandas

Matplotlib

Keras

Skimage

OpenCV
