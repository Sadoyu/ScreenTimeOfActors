try:

	#PART 1 START

	print("Importing the packages.....")
	import cv2     # for capturing videos
	import math   # for mathematical operations
	import matplotlib.pyplot as plt    # for plotting the images
	#matplotlib inline
	import pandas as pd
	from keras.preprocessing import image   # for preprocessing the images
	import numpy as np    # for mathematical operations
	from keras.utils import np_utils
	from skimage.transform import resize   # for resizing images
	


	print("Collecting Frames.....")
	count = 0
	videoFile = "sample.mp4"
	cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
	frameRate = cap.get(5) #frame rate
	x=1
	while(cap.isOpened()):
	    frameId = cap.get(1) #current frame number
	    ret, frame = cap.read()
	    if (ret != True):
	        break
	    if (frameId % math.floor(frameRate) == 0):
	        filename ="frame%d.jpg" % count;count+=1
	        cv2.imwrite(filename, frame)
	cap.release()
	


	
	print("plotting frames")
	img = plt.imread('frame0.jpg')   # reading image using its name
	plt.imshow(img)
	

	#PART 1 END

	#PART 2 START

	print("using our model")
	data = pd.read_csv('mapping.csv')     # reading the csv file
	data.head()      # printing first five rows of the file
	print(data.head())


	print("stealing your ram")
	X = [ ]     # creating an empty array
	for img_name in data.Image_ID:
	    img = plt.imread('' + img_name)
	    X.append(img)  # storing each image in array X
	X = np.array(X)    # converting list to array




	print("encoding class")
	y = data.Class
	dummy_y = np_utils.to_categorical(y)    # one hot encoding Classes



	print("resizing of frames")
	image = []
	for i in range(0,X.shape[0]):
	    a = resize(X[i], preserve_range=True, output_shape=(224,224), mode = 'constant').astype(int)      # reshaping to 224*224*3
	    image.append(a)
	X = np.array(image)
	


	print("preprocessing of frames")
	from keras.applications.vgg16 import preprocess_input
	X = preprocess_input(X, mode='tf')      # preprocessing the input data
	

	print("dividing into training and testing pairs and preparing for validation")
	from sklearn.model_selection import train_test_split
	X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)    # preparing the validation set
	


	#PART 2 END

	#PART 3 START
	
	print("Impoting additional packages.....")
	from keras.models import Sequential
	from keras.applications.vgg16 import VGG16
	from keras.layers import Dense, InputLayer, Dropout



	print("loading pretrained model")
	base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))    # include_top=False to remove the top layer



	print("training the model")
	print("making predictions")
	X_train = base_model.predict(X_train)
	X_valid = base_model.predict(X_valid)
	X_train.shape, X_valid.shape


	print("Reshaping models")
	X_train = X_train.reshape(208, 7*7*512)      # converting to 1-D
	X_valid = X_valid.reshape(90, 7*7*512)
	

	print("zero centering the images")
	train = X_train/X_train.max()      # centering the data
	X_valid = X_valid/X_train.max()


	print("building the model")
	# i. Building the model
	model = Sequential()
	model.add(InputLayer((7*7*512,)))    # input layer
	model.add(Dense(units=1024, activation='sigmoid')) # hidden layer
	model.add(Dense(3, activation='softmax'))    # output layer

	model.summary()

	print("Compiling the model")
	# ii. Compiling the model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	print("Training the model")
	# iii. Training the model
	model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid))


except:
	print("Excemption!")