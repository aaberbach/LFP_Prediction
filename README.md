# LFP_Prediction
Goal: Analyze how a neural network can be useful in Gamma prediction and using XAI to determine what the neural network is looking at. 

Past Work

Forecasting raw LFP - We showed that a neural network using LSTM and Linear layers did better at forecasting LFP than baseline persistence and autoregression. We also showed that using the interneurons and pns as well as the raw LFP got better results. Results were not good enough to be practically used for raw LFP forecasting. We could try doing this on the newer dataset with more variables and we could try using a CNN for this, which we have not done. 
Classifying LFP - The next thing we tried was taking a window of LFP and trying to predict if the next 10 ms included a Gamma burst. The results were not very good, we were able to get about 70% burst accuracy and 66% non-burst accuracy. We have not tried this with the new data. 
Hilbert Classification - Next we tried taking a window of LFP (with the old data) and trying to classify the next 10 ms as having either low, medium, or high average hilbert value (of the filtered LFP). I did not get good results on my model but Ben seemed to do much better, maybe worth looking at again with new data?
4 Path Classification - The most recent task we tried was using a few different classification tasks. The tasks were: classifying if the next peak/trough will be above threshold, classifying how long until the next peak/trough, and classifying how many consecutive peaks/troughs above threshold there will be. The classification techniques about being above threshold did not work very well but the classification on time until peak worked pretty well. Then, we improved the classification of whether the next peak will be above threshold to about 65% above accuracy and 86% bellow accuracy using only samples where the peak is less than 3 ms away. We then implemented GradCam and Guided Backpropagation on that model. 

Current Work

	Our current idea is to create a more simple dataset. The dataset will just be made up of some x frequency noisy sine waves and some y frequency noisy sine waves that the model will have to classify between. The point of this is that we can more easily build a pipeline to use XAI techniques. Techniques we are looking to use include: Guided GradCam, Occlusion, Named Neurons, Gradient descent on input, and failure analysis. 

