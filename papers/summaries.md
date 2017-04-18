# Paper summaries and relevance

### (Li, 2010)
Uses a CNN to distinguish between 10 main genres. It resulted in 84% accuracy
on the training set, but less than 30% accuracy on the test set. Their
explanation: 80 songs can't represent a genre. Their solution: adding affine
transformations and increasing the dataset. I think a CNN will only work on the
lowest level, linking rhythmic features within 2 bars together. All higher
abstractions are useless.  
Li, T.L., Chan, A.B., Chun, A. (2010). *Automatic Musical Pattern Feature Extraction Using Convolutional Neural Network*. In *In Proc. IMECS*.

### (Sturm, 2014)
Overview of previous work, the different approaches, datasets and their
analyses. (haven't finished reading)  
Sturm, B.L. (2014). A Survey of Evaluation in Music Genre Recognition. In: Nürnberger, A., Stober, S., Larsen, B. and Detyniecki, M., ed., *Adaptive Multimedia Retrieval: Semantics, Context, and Adaptation*, 1st ed. AMR 2012. Lecture Notes in Computer Science, vol 8382. Springer, Cham. pp.29-66.

### (Vogler, 2016)
Created a CNN-LSTM hybrid, achieved 83% accuracy. The dataset contains 3 genres (rock, pop, hip-hop) with 100 songs each. Each track is split into
segments of 30s, then the MFCC and OSC are extracted for each segment, then those are used as features in the training process.
The track label is the most frequent genre tag. It has been implemented in Python using Keras on Theano. Randomized search
to tweak the hyper features (Spearman's Rank Correlation Coefficient). It is a really unprofessionally written paper.  
The results suprise me. Although the CNN-LSTM approach is a good choice, and the few target genres eases the challenge,
I am not convinced MFCC and OSC are sufficient to define a genre, nor should 100 tracks be enough to train a network. These two
criticisms are also what they proposed as future work. Although the results appear to dominate the competition, no effort is
made to do this comparison. The paper hasn't been published in any journal.  
Vogler, B.S., Othman, A. (2016). *Music Genre Recognition*. [online] benediktsvogler.com.

### (Lidy, 2016)
Winner of the MIREX 2016 Latin Genre Classification. Uses a sequential CNN and a parallel CNN on the Mel spectrogram. Builds on their winning MIREX 2015 music/speech classifier.
Softmax layer makes it a distribution. A dense layer with 20% dropout reduces overfitting. Implemented in Python-Keras-Theano.
Managed a 69.54% accuracy on latin genres, and 73.14% on the mixed genres (lost to (Wu, 2015)).  
Lidy, T., Schindler, A. (2016). *Parallel Convolutional Neural Networks for Music Genre and Mood Classification*. MIREX2016.

### (Costa, 2017)
Compares the internal representations of CNNs with the most common handpicked features. Concludes that the use of CNNs is
justified in MGR.  
Costa, Y.M., Oliveira, L.S. and Silla, C.N. (2017). *An evaluation of Convolutional Neural Networks for music classification using spectrograms*. Applied Soft Computing, 52, pp.28-38.

### (Siva, 2014)
Applied a PNN to 4 EDM subgenres (Trance, Electro-house, Dubstep, Techno). The database contained 100 tracks of 22s for
each genre, 70-30 train/test split. The features were Statistical Spectrum Descriptors and Rhythm Histograms, 128 features were brought down to 50
using RELIEFF.  
Baruah, T. and Tiwari, S. (2014). 'Intelligent classification of electronic music'. In Signal Processing and Information Technology (ISSPIT), 2014 IEEE International Symposium on (pp. 000031-000035). IEEE.

### (Kirss, 2007)
Classification of 5 electronic subgenres using rhythmic features in a SVM classification resulted in an accuracy of 96.4%.
The dataset consisted of 50 excerpts of 20s for each of 5 genres: deep house, techno, trance, drum & bass and ambient.
The evaluation was done with 10-fold cross validation, which means that 25 tracks of each genre were used to classify only 5
test tracks.  
For this reason, I don't think the results can be compared to that of other MGR experiments. There is a good
chance that the dataset lacks diversity. The exact composition of the dataset has been provided in the appendix for reference.  
Kirss, P. (2007). 'Audio Based Genre Classification of Electronic Music', MMT thesis, Jyväskylän Yliopisto, Jyväskylä.

### (Chen, 2014)
Applied a GMM using 7 features (including MFCC) which resulted in a 80.67% accuracy. The dataset contained 10 30s clips
for each of the genres: deep house, dubstep, progressive house. Three-fold cross validation was used in the evaluation.  
Once agian, the size of the dataset is insufficient, which damages the reliability of the experiments. Even within the 10
tracks that ought to represent an entire subgenre are artist duplicates.  
Chen, A.C. (2014). 'Automatic Classification of Electronic Music and Speech/Music Audio Content', ECE MSc Thesis, University of Illinois, Urbana, Illinois.