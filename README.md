# Digits Detection
Implementation of a Convolutional Neural Network (CNN) for detecting digits (0-9)
in an image (Detection Problem).

## Preparing the data

Here we are going to use a collection of mnist digits. Each image is 28x28, but we 
will resize them to 32x32.

````python
python image_resize.py
````

So, it is important to have the _data/digits_ folder with one folder per digit in it. 
Inside each digit folder, there should be at least 10 different images of the 
corresponding digit.

##### Making the datasets

Each training and testing image should be a concatenation of 8 images 
(currently working), and for that, the code is included in the cnn.py file.

You could change N_DIGITS value, if so, you will need to adapt the code.

## Classifying

Just run 

````python
python cnn.py
````
