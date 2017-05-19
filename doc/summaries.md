# Paper summaries and relevance

### (Kirss, 2007)
Classification of 5 electronic subgenres using rhythmic features in a SVM classification resulted in an accuracy of 96.4%.
The dataset consisted of 50 excerpts of 20s for each of 5 genres: deep house, techno, trance, drum & bass and ambient.
The evaluation was done with 10-fold cross validation, which means that 25 tracks of each genre were used to classify only 5
test tracks.  
For this reason, I don't think the results can be compared to that of other MGR experiments. There is a good
chance that the dataset lacks diversity. The exact composition of the dataset has been provided in the appendix for reference.  
Kirss, P. (2007). 'Audio Based Genre Classification of Electronic Music', MMT thesis, Jyväskylän Yliopisto, Jyväskylä.

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

### (Siva, 2014)
Applied a PNN to 4 EDM subgenres (Trance, Electro-house, Dubstep, Techno). The database contained 100 tracks of 22s for
each genre, 70-30 train/test split. The features were Statistical Spectrum Descriptors and Rhythm Histograms, 128 features were brought down to 50
using RELIEFF.  
Baruah, T. and Tiwari, S. (2014). 'Intelligent classification of electronic music'. In Signal Processing and Information Technology (ISSPIT), 2014 IEEE International Symposium on (pp. 000031-000035). IEEE.

### (Chen, 2014)
Applied a GMM using 7 features (including MFCC) which resulted in a 80.67% accuracy. The dataset contained 10 30s clips
for each of the genres: deep house, dubstep, progressive house. Three-fold cross validation was used in the evaluation.  
Once agian, the size of the dataset is insufficient, which damages the reliability of the experiments. Even within the 10
tracks that ought to represent an entire subgenre are artist duplicates.  
Chen, A.C. (2014). 'Automatic Classification of Electronic Music and Speech/Music Audio Content', ECE MSc Thesis, University of Illinois, Urbana, Illinois.

### (Alexandridis, 2014)
Alexandridis, A., Chondrodima, E., Paivana, G., Stogiannos, M., Zois, E. and Sarimveis, H., 2014, September. Music genre classification using radial basis function networks and particle swarm optimization. In Computer Science and Electronic Engineering Conference (CEEC), 2014 6th (pp. 35-40). IEEE.

### (Kong, 2014)
Used a CNN and got 72.4% on the GTZAN dataset.

### (Sturm, 2015)
Sturm B.L., Kereliuk C., Larsen J. (2015) ¿El Caballo Viejo? Latin Genre Recognition with Deep Learning and Spectral Periodicity. In: Collins T., Meredith D., Volk A. (eds) Mathematics and Computation in Music. MCM 2015. Lecture Notes in Computer Science, vol 9110. Springer, Cham

### (Wu, 2015)
In response to the bag-of-frames approaches, Wu argued that time-frequency analysis is important as well. Wu won the MIREX
mixed genre classification contests 2011-2016. It extracts multilevel visual features from the spectrogram and their temporal
variations. It also includes beat tracking, and a confidence-based classification. The results of the 2016 MIREX contest
were an accuracy of 76.27%, and in 2014 even 83.55%. The evaluation and training used the 4 most used datasets for MGR,
one of which comprised more than 25 million songs.  
The way song-level patterns (like hooks, drops and verses) are combined with beat-level patterns (which are so descriptive in
electronic music) give insight into what defines a genre. This could be a very interesting aspect to keep in mind while
engineering a neural network alternative.  
Wu, M.J. and Jang, J.S.R. (2015). *Combining acoustic and multilevel visual features for music genre classification*. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 12(1), p.10.

### (Geng, 2016)
Geng, S., Ren, G. and Ogihara, M., 2016, September. A Hierarchical Sonification Framework Based on Convolutional Neural Network Modeling of Musical Genre. In Audio Engineering Society Convention 141. Audio Engineering Society.

### (Dai, 2016)
Dai, J., Liang, S., Xue, W., Ni, C. and Liu, W., 2016, October. Long short-term memory recurrent neural network based segment features for music genre classification. In Chinese Spoken Language Processing (ISCSLP), 2016 10th International Symposium on (pp. 1-5). IEEE.

### (Panteli, 2016)
Models the similarity between timbre and rhythm in electronic music. Rhythmic elements are best to describe electronic music.
Panteli, M., Rocha, B., Bogaards, N. and Honingh, A., 2016. A model for rhythm and timbre similarity in electronic dance music. Musicae Scientiae, p.1029864916655596.

### (Choi, 2016 #1)
Deconvolutes a CNN trained to classify music using the process of auralisation.  
Choi, K., Fazekas, G. and Sandler, M. (2016). *Explaining deep convolutional neural networks on music classification*. arXiv preprint arXiv:1607.02444.

### (Choi, 2016 #2)
Combined a CNN and a GRU into a CRNN to catch both local features as temporal features. A CRNN is essentially a CNN with the
last layers being RNNs instead of convolutions. The CRNN outperformed k2c2 CNNs with a slight margin in accuracy, but
sacraficed speed and memory to do so.

### (Iloga, 2016)
Scored an accuracy of 91.6% on the GTZAN dataset through hierarchical pattern matching. It aims at unsupervised taxonomy generation using K-nearest neighbor.

### (Shi, 2016)
Developed 'ShuttleNet', a GRU RNN with both feedforward as feedback connections, and embedded it in a CNN-RNN network.
It beat all 14 previous state of the art networks tested. (not finished reading)

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