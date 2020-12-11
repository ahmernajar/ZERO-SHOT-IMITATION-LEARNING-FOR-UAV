# ZERO-SHOT-IMITATION-LEARNING-FOR-UAV

Ahmer Jalil Najar Ganesh Rapeti Mohd Kashif Ansari 
Abstract
Zero-shot learning consists in training a machine/agent with little or no training examples and then testing it on similar random test examples. Convolutional networks are at the core of most state-of-the-art computer vision solutions for a wide variety of tasks. There are very few models that can utilise useful feature but do not learn end to end embedding thus the architecture of our model uses Convolutional Neural Networks (CNN) to extract features of the frames from video (recorded by the drone) and thus making these features as input to the LSTM to remember the input over a long period of time. Thus, making it easy to classify between different labels. We have developed four deep learning models to carry out this task. 
1. Introduction
The recent success of the deep learning approaches to object recognition is supported by the collection of large datasets with millions of images and thousands of labels. Although the datasets continue to grow larger and are acquiring a broader set of categories, they are very time consuming and expensive to collect. Furthermore, collecting detailed, fine- grained annotations, such as attribute or object part labels, is even more difficult for datasets of such size. The ability to classify images of an unseen class is transferred from the semantically or visually similar classes that have already been learned by a visual classifier. Machine learning algorithms have been successfully applied to learning classifiers in many domains such as computer vision, fraud detection, and brain image analysis. Typically, classifiers are trained to approximate a target function f: X → Y, given a set of labelled training data that includes all possible values for Y, and
sometimes additional unlabelled training data.
Automatic classification is arguably the first problem considered in machine learning, thus it has been thoroughly studied and analysed, leading to a wide variety of classification approaches which have been proved useful in many areas such as computer vision and document classification. However, these approaches cannot generally tackle challenging scenarios in which new classes may appear after the learning stage. We find this scenario in lots of real-world situations.
Inspired by the deep learning breakthroughs in the image domain [20] where rapid progress has been made in the past few years in feature learning, various pre-trained convolutional network (ConvNet) models [14] are made available for extracting image features. These features are the activations of the network’s last few fully-connected layers which perform well on transfer learning tasks [12, 13].

  Zero-shot learning is closely related to one-shot learning [11, 5, 1, 8], where the goal is to learn object classifiers based on a few labelled training exemplars. The key difference in zero-shot learning is that no training images are provided for a held-out set of test categories. Thus, zero-shot learning is more challenging, and the use of side information about the interactions between the class labels is more essential in this setting. Nevertheless, we expect that advances in zero-shot learning will benefit one-shot learning, and visual recognition in general, by providing better ways to incorporate prior knowledge about the relationships between the object categories
ZSL consists in recognising new categories of instances without training examples, by providing a high-level description of the new categories that relate them to categories previously learned by the machine. Zero-shot learning aims to recognize objects whose instances may not have been seen during training [1], [2], [3], [4], [5], [6]. This paradigm is inspired by the way human beings are able to identify a new object by just reading a description of it, leveraging
similarities between the description of the new object and previously learned concepts.
Among the many existing network structures, Convolutional Neural Networks (CNN) have demonstrated great success on various tasks, including image classification [18, 25, 24], image-based object localization [7], speech recognition [8], etc. For video classification, Ji et al. [11] and Karparthy et al. [17] extended the CNN to work on the temporal dimension by stacking frames over time. Recently, Simonyan et al. [23] proposed a two- stream CNN approach, which uses two CNNs on static frames and optical flows respectively to capture the spatial and the motion information. It focuses only on short-term motion as the optical flows are computed in very short time windows. With this approach, similar or slightly better performance than the hand-crafted features like [26] has been reported.
2. Problem to be solved
The objective is to test the drone on an unseen random environment by training it on a different environment only once. We

 get the video from parrot-bebop drone which is converted into of NumPy array (‘.npy’ format) of each frame with corresponding keystrokes from keyword to train the drone to move in a specified direction. Our aim is to construct a deep learning model to classify new frames and to detect the specified keystroke that has to be used for that frame at real-time. The NumPy array which we got was a 5- dimensional data. These dimensions are number of frames, width of the screen, height of the screen, colour and the keystrokes. We have to use the features of the frame to detect the keystroke to be used. We can reduce the data into 4- dimensions by using radon transform which converts the coloured video frame to a black and white video frame. We also separate the keystrokes from the data and use them as labels.
By reducing the dimension of the data, we can easily fit the data to a model like LSTM (Long Short-Term Memory), the Convolution-2D model, Convolutional-3D model which uses 3D data as input. We assume that the labels are themselves composed of attributes and attempt to
been previously considered in [9, 16], where the attributes are manually annotated. In [9], the training set attributes are predicted along with the image label at test time, [19] explores relative attributes, which capture show images relate to each other along different attributes.
Assume that a labelled training dataset of images D0 ≡ (xi, yi) i=1 is given,
where each frame (in the form of image) is
represented by a ρ -dimensional feature
vector, xi ∈ Rρ . For generality we
denote χ def Rρ . There are n0 distinct =
class labels available for training, i.e., yi ∈ Y0 ≡ {1,..., n0} . In addition, a test dataset denoted
{ }m
D0 ≡ x′i, y′ is provided, {( i)}m′
j=1
where x′jεχ as above, while
y′jεY≡{n0+1,...,n0+n1}.
The test set contains n1 distinct class labels, which are omitted from the training set. Let n = n0 + n1 denote the total number of labels in the training and test
 learn the semantic relationship between the attributes and frames. In this way, new
labels can be constructed by combining different sets of attributes. This setup has
sets. The goal of zero-shot learning is to train a classifier on the training set D0, which performs reasonably well on the unseen test set D1. Clearly, without any

 side information about the relationships between the labels in Y0 and Y1, zero-shot learning is infeasible as Y0 ∩ Y1 = ∅ .
3. Approaches to solve the problem
1) 2D-Convolutional Neural Network (2DConvNets)
We introduce a 2D-convolutional classifier that operates directly on the intermediate feature maps of a CNN. The image-based CNN features have recently been directly adopted for video classification, extracted using off-the-shelf models trained on large-scale image datasets like the ImageNet [10, 22, 27]. For instance, Jain et al. [10] performed action recognition using the SVM classifier with such CNN features and achieved top results in the 2014 THUMOS action recognition challenge [15]. A few works have also tried to extend the CNN to exploit the motion information in videos. Ji et al. [11] and Karparthy et al. [17] extended the CNN by stacking visual frames in fixed size time windows and using spatial-temporal convolutions for video classification. Differently, the two-stream CNN approach by Simonyan et al. [23] applies the CNN separately on visual frames (the spatial stream) and stacked optical flows (the motion stream). This approach has been found to be more effective, which is adopted as the basis of our proposed framework. However, as discussed, all these approaches [11, 17, 23] can only model short-term motion, not the long- term temporal clues.
The convolutional classifier model operate on the feature maps extracted from CNNs. Recent work [21] has shown that using features from convolutional layers is
beneficial over just using the final fully connected layer features of a CNN.
2) 3D-Convolutional Neural Network (3DConvNets)
We propose a simple, yet effective approach for spatiotemporal feature learning using deep 3-dimensional convolutional networks (3DConvNets) trained on a large-scale supervised video dataset. We empirically show that these learned features with a simple linear classifier can yield good performance on various video analysis tasks. We believe that 3DConvNet is well-suited for spatiotemporal feature learning. Compared to 2D ConvNet, 3D ConvNet has the ability to model temporal information better owing to 3D convolution and 3D pooling operations. In 3D ConvNets, convolution and pooling operations are performed spatiotemporally while in 2D ConvNets they are done only spatially.
However, Conv2D methods are built on using only 2D convolution and 2D pooling operations (except for the Slow Fusion model in [17]) whereas our model performs 3D convolutions and 3D pooling propagating temporal information across all the layers in the network. We have separated the keystrokes from the total data and by now we have only 4-D data to be used from convolution, but convolution 3-D has to be given a 5-D input so we reshape the remaining data into 5-D and pass the data through convolution layer and maxpooling layer. the obtained features are flattened and passed through a fully connected layer.
3) Long-Short Term Memory (LSTM)
Long Short-Term Memory (LSTM) networks are an extension for recurrent neural networks, which basically extends their memory.

 Therefore, it is well suited to learn from important experiences that have very longtime lags in between. The units of an LSTM are used as building units for the layers of RNN, which is then often called an LSTM network.
LSTM’s enable RNN’s to remember their inputs over a long period of time. This is because LSTM’s contain their information in a memory, that is much like the memory of a computer because the LSTM can read, write and delete information from its memory.
In an LSTM you have three gates: input, forget and output gate. These gates determine whether or not to let new input in (input gate), delete the information because it isn’t important (forget gate) or to let it impact the output at the current time step (output gate).
You can see an illustration of RNN with its three gates below:
The gates in a LSTM are analog, in the
form of sigmoid, meaning that they range from 0 to 1.
The fact that they are analog, enables them to do backpropagation with it.
The problematic issues of vanishing gradients are solved through LSTM because it keeps the gradients steep enough and therefore the training relatively short and the accuracy high.
4) Long-term Recurrent Convolution Network (LRCN)
In this section we propose Long-term Recurrent Convolutional Networks (LRCNs), a class of architectures for visual recognition and description which combines convolutional layers and long- range temporal recursion and is end-to- end trainable.
Research on CNN models for video processing has considered learning 3D spatio-temporal filters over raw sequence data, and learning of frame-to-frame representations which incorporate instantaneous optic flow or trajectory- based models aggregated over fixed windows or video shot segments.
We show here that convolutional networks with recurrent units are generally applicable to visual time-series modelling.
The CNN Long Short-Term Memory Network or CNN LSTM for short is an
  
   LSTM architecture specifically designed
LSTM networks for long term temporal modelling.
CNN LSTM Architecture
The CNN LSTM architecture involves using Convolutional Neural Network (CNN)
layers for feature extraction on input data
 for sequence prediction problems with spatial inputs like images or videos.
Given an input video, two types of features are extracted using the CNN from spatial frames and short-term stacked motion optical flows respectively. The features are separately fed into two sets of
combined with LSTMs to support sequence prediction.
CNN LSTMs were developed for visual time series prediction problems and the application of generating textual descriptions from sequences of images (e.g. videos). Specifically, the problems of:

 Image Description : Generating a textual description of a single image.
Video Description: Generating a textual description of a sequence of images.
Activity Recognition: Generating a textual description of an activity demonstrated in a sequence of images.
A CNN LSTM can be defined by adding CNN layers on the front end followed by LSTM layers with a Dense layer on the output.
It is helpful to think of this architecture as defining two sub-models: the CNN Model for feature extraction and the LSTM Model for interpreting the features across time steps.
Overall, deep learning affords novel and
powerful techniques for video prediction and analysis. Accordingly, it is important to summarize the current state-of-the-art for video analysis using deep learning techniques and the associated challenges.
4. How Problem Solved
As we discussed above four approaches to solve the problem CNN, Convolution 3D, LSTM, LRCN.
We can collect the dataset simply in ‘.mp4’ format or ‘.npy’ format . But the ‘.npy’ format was good for us because it automatically merges the frames with the key strokes pressed on the keyboard (W, A, S, D, etc.).
So, our problem was to detect the output which is keystrokes taking input as frames recorded by the camera in the drone.
It was visualizing the map and simultaneously it was categorizing the features of the frames.
4.1. How we got the desired output
CNN
In the architecture of our model we have used four-layer network with two fully connected layers which extract the most important features of the frames and relates it to the unseen classes to give the output by reading the frames in the video.
The CNN model accuracy was 94.02% which was pretty good and it was able to give the output of the unseen classes by reading the features of it.
Conv3D
We have built this model with two 3-D convolution layers and two Maxpooling layers and followed by flatten layer to flatten the features obtained and later we passed it through dense layer of 2048 neurons and finally constructed the output dense layer.
This model requires high performance GPU computers to train and test the model. So, this model is not that efficient to be used for testing.
LSTM
It was able to remember the values over arbitrary time intervals but it wasn’t that much good as CNN to extract the features from the frames while testing it.
So, the accuracy was 81.67% which is good but less than the CNN.
LRCN
LRCN is the combination of the CNN and LSTM. As far as CNN part is concerned it will extract the features to relate the unseen classes and however the LSTM part remembers the value over time intervals.
The accuracy was about 86% which is pretty good and it was able to detect the

 unseen classes giving us the desired output.
5. Conclusion
Similar to traditional classification models, our proposed method can also be used for object recognition by training on the entire dataset.
We’ve presented LSTM, Conv3D, Conv2D and LRCN, a class of models that is both spatially and temporally deep, and flexible enough to be applied to a variety of vision tasks involving sequential inputs and outputs.
Although deep learning based approaches have been successful in addressing many problems, effective network architectures are urgently needed for modelling sequential data like the videos. Several researchers have recently explored this direction. However, compared with the progress on image classification, the achieved performance gain on video classification over the traditional hand- crafted features is much less significant. Our work in this paper represents one of the few studies showing very strong results. For future work, further improving the capability of modelling the temporal dimension of videos is of high priority. In addition, audio features which are known to be useful for video classification can
also be easily incorporated into our framework.
As a concluding comment, we acknowledge that many problems require complex solutions, but that does not mean that
    
 simple baselines should be ignored. On the contrary, simple but strong baselines both bring light about which paths to follow in order to build more sophisticated solutions, and also provide a way to measure the quality of these solutions.
Fig.1 Training the drone on the above straight path.  
References
Fig.2 Testing the drone a different straight path.
  Fig.3 Training the drone on a curvy path
  Fig. 4 Testing the drone on a different curvy path.
[1] C. Lampert, H. Nickisch, and S. Harmeling, “Attribute-based classification for zero-shot visual object categorization,” in TPAMI, 2013.
[2] H. Larochelle, D. Erhan, and Y. Bengio, “Zero-data learning of new tasks,” in AAAI, 2008.

 [3] M. Rohrbach, M. Stark, and B.Schiele, “Evaluating knowledge transfer and zero-shot learning in a large-scale setting,” in CVPR, 2011.
[4] X. Yu and Y. Aloimonos, “Attribute-based transfer learning for object categorization with zero or one training example,” in ECCV, 2010.
[5] X. Xu, Y. Yang, D. Zhang, H. T. Shen, and J. Song, “Matrix trifactorization with manifold regularizations for zero-shot learning,” in CVPR, 2017.
[6] A. Farhadi, I. Endres, D. Hoiem, and D. Forsyth. Describing objects by their attributes. In CVPR, pages 1778–1785. IEEE, 2009.
[7] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In CVPR, 2014.
[8] G. E. Dahl, D. Yu, L. Deng, and A. Acero. Context-Dependent Pre-Trained Deep Neural Networks for Large-Vocabulary Speech Recognition. IEEE TASLP, 2012.
[9] J. Donahue, L. A. Hendricks, S. Guadarrama, M. Rohrbach, S. Venugopalan, K. Saenko, and T. Darrell. Long-term recurrent convolutional networks for visual recognition and description. CoRR, 2014.
[10] M. Jain, J. van Gemert, and C. G. M. Snoek. University of Amsterdam at THUMOS challenge 2014. In ECCV THUMOS Challenge Workshop, 2014.
[11] S. Ji, W. Xu, M. Yang, and K. Yu. 3d convolutional neural networks for human action recognition. In ICML, 2010.
[12] N.Zhang, M.Paluri, M.Ranzato, T.Darrell and L.Bourdev. Panda: Pose aligned networks for deep attribute modelling. In CVPR, 2014.
[13] B. Zhou, A. Lapedriza, J. Xiao, A. Torralba, and A. Oliva. Learning deep features for scene recognition using places database. In NIPS, 2014.
[14] Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S.Guadarrama and T.Darrell. Caffe: Convolutional architecture for fast feature embedding. arXiv preprint arXiv: 1408.5093, 2014.
[15] C. H. Lampert, H. Nickisch, and S. Harmeling. Learning to detect unseen object classes by between class attribute transfer. In CVPR, 2009
[16] Y.-G. Jiang, J. Liu, A. Roshan Zamir, G. Toderici, I. Laptev, M. Shah, and R.Sukthankar. THUMOS challenge: Action recognition with a large number of classes. http://crcv.ucf.edu/ THUMOS14/, 2014.
[17] A. Karpathy, G. Toderici, S. Shetty, T. Leung, R. Sukthankar, and L. Fei-Fei. Large-scale video classification with convolutional neural networks. In CVPR, 2014.
[18] A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet classification with deep convolutional neural networks. In NIPS, 2012.
[19] Z. Lan, M. Lin, X. Li, A. G. Hauptmann, and B. Raj. Beyond gaussian pyramid: Multi- skip feature stacking for action recognition. CoRR, 2014.

 [20] A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet classification with deep convolutional neural networks. In NIPS, 2012.
[21] K. Xu, J. Ba, R. Kiros, A. Courville, R. Salakhutdinov, R. Zemel, and Y. Bengio. Show, attend and tell: Neural image caption generation with visual attention. arXiv preprint arXiv: 1502.03044, 2015.
[22] A. S. Razavian, H. Azizpour, J. Sullivan, and S. Carlsson. CNN features off-the-shelf: an astounding baseline for recognition. CoRR, 2014.
[23] K. Simonyan and A. Zisserman. Two-stream convolutional networks for action recognition in videos. In NIPS, 2014.
[24] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. CoRR, 2014.
[25] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going Deeper with Convolutions. CoRR, 2014.
[26] H. Wang and C. Schmid. Action recognition with improved trajectories. In ICCV, 2013.
[27] S. Zha, F. Luisier, W. Andrews, N. Srivastava, and R. Salakhutdinov. Exploiting image- trained CNN architectures for unconstrained video classification. CoRR, 2015.
