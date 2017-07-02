# Electronic Music Genre Recognition
## Electronic Subgenre Recognition Using Convolutional Neural Networks
### Abstract
The presented work proposes a new method of classifying music genres. The field of music genre recognition (MGR) aims to develop a system that can predict the genre of any song. However, the past 20 years of research on this topic have taken certain variables for granted, such as the size of the datasets and the genres used for classification. Calling these variables into question could discover valuable directions for the entire field. Since music genres are structured like a taxonomy, one could train a model on the subgenres of the genres used for testing by abstracting the predictions back to their main genre. This allows the model to learn about the specific ways a genre is manifested. Electronic music is a suitable testing ground for such an inquiry, since it is saturated with taxonomical subdivisions, the resulting classifier is relevant in contemporary society, and yet insufficient research has been conducted on this genre. No prior research was found using a neural network to classify electronic subgenres. Two convolutional neural network (CNN) models and two different datasets have been applied to test the effect of subgenre targeting. It achieved moderately positive results, only being beneficial for larger models and larger datasets. The best accuracy was 92.4% on a binary classification task using no subgenre targeting. Future work could evaluate the effects of subgenre targeting on larger models and datasets while controlling for the amount of parameters.
