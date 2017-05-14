# Logbook

## Week 1 - 📚 literary research
### *(total: 10.5 hrs)*  
__**Th 13-04-2017**__  
**11:00-11:45:** Introductory meeting. The current plan is to directly apply an RNN to the signal, using ID3 tags as target variables. The dataset will include Electronic subgenres only, but this may change depending on early results.  
My next coarse of action:
 * ✔ Research different types of RNNs
 * ✔ Search for existing MGR algorithms using RNNs/NNs (ISMIR, MGR challenges)
 * ✔ Learn how RNNs work
 * ✔ Set up a Git repos including a logbook and article dump folder

__**Fr 14-04-2017**__  
**10:00-12:30:** Setting up Git architecture. Understanding intuition behind RNNs.  
**15:30-16:30:** First literary search

__**Su 16-04-2017**__  
**10:30-12:30:** Read (Vogler, 2016) and (Lidy, 2016), and collected more papers.

__**Mo 17-04-2017**__  
**11:00-12:00:** Read (Costa, 2017), collected more papers.  
**14:00-:14:30** Read (Siva, 2014)

__**Tu 18-04-2017**__  
**10:30-12:15:** Read (Kirss, 2007) and (Chen, 2014).

__**We 19-04-2017**__  
**15:00-16:00:** Read (Choi, 2016 #1), (Choi, 2016 #2) and (Wu, 2015).

## Week 2 - 📝 Proposal
### *(total: 16.25 hrs)*  
__**Th 20-04-2017**__  
**11:00-11:30:** Second meeting. I am well on schedule. The proposal will focus on electronic music,
it will use a proven algorithm (probably Wu or Choi), but it will innovate on the scale of classes/genres.  
My plan for the coming week:  
* ✔ Write & hand in literature review (✖ may run it by supervisors first)
* ✔ Prepare the proposal presentation (✔ run by supervisors first)
* ✔ Write & hand in research proposal (✖ run by supervisors first)
* ✖ Have a concrete plan for an architecture by Tuesday April 2nd.

__**Fr 21-04-2017**__  
**12:00-18:45:** Completed the 'What is AI Research?' assignment and literature draft.  

__**Mo 24-04-2017**__  
**9:30-14:00:** Project proposal presentation

__**Tu 25-04-2017**__  
**10:15-12:00:** Research proposal

__**We 26-04-2017**__  
**10:30-11:30:** Research proposal: finished the literature review  
**19:45-21:30:** Finished research proposal

## Week 3 - 🖼️ Design
### *(total: 5 hrs)*  
__**Fr 28-04-2017**__  
**10:30-13:00:** Searched for CNN-RNN hybrids, found a very interesting research by (Shi, 2016). Explored the basics of NN design and -implementation with Keras and it appears
to be very feasible. Dove deeper into CNN mechanics. More work was required to get an architecture down than expected. I'm
planning to continue researching the exact implementations of CNN-RNN hybrids on Monday, and to have a concrete plan by the
end of the day (or at the very least an outline without a detailed NN implementation) so there's something to discuss on Tuesday.

__**Mo 01-05-2017**__  
**10:30-12:00:** Pipeline design, NN architecture research. I believe I can realize confidence scores using a softmax layer.
The input will be the Mel spectrogram, so as to not deviate from previous research too much. I am unsure about multiple target
variables and predictions to accomodate genre hierarchies.  
**15:30-17:30:** CNN layers research.

__**Tu 02-05-2017**__  
**13:30-14:30:** Third meeting. My design so far is insufficient. It lacks detailed descriptions of window- and filter sizes. The plan for the coming two weeks is to implement a simple CNN pipeline similar to (Choi, 2016). My deadline for Tuesday the 9th is to
* ✔ Determine the size and ΔT for the FFT
* ✔ Determine the dimensions of the convolutional filters
* ✔ Get access to the UvA DAS
* ✖ Start implementing the preprocessing

## Week 4 - 🔬 Details
### *(total: 10.5 hrs)*  
__**Th 03-05-2017**__  
**09:30-11:00:** I tried to make spectrograms of some sample beats, but I stumbled upon multiple librosa import errors.  
**13:00-15:30:** I made the spectrograms with an online tool instead, analysed some drum loops and individual samples, and found the properties presented below. In order to recognize the smallest element, the FFT window will be 20ms with a stride of 10ms. The convolutional filter will be a vertical slice, 20kHz x 60ms, with a stride of 20ms.

| sample  | len (ms) | freq (kHz) |
|---------|----------|------------|
| kicks   | 50       | 0-0.25     |
| snares  | 100-250  | 0-15       |
| hi-hats | 100      | 4-20       |

__**Tu 09-05-2017**__  
**10:00-10:30:** Fourth meeting. I've had a busy week and little results. My hypothesis was interesting, but it's still better to start with a preexisting architecture.
The four things to do before Friday the 19th is to
* Put in a minimum of 30 hours a week (or 5 hours a day)
* Get comfortable with the DAS (✖ attend the introduction on Thursday)
* Start to write the paper, especially the skeleton, intro, related work and preprocessing
* Get as far as possible on implementing a simple CNN in Tensorflow
	* ✔ Change to Python 2.7 and Anaconda
	* ✔ Install Librosa
	* ✔ Preprocessing
	* CNN network

**10:30-12:30:** Planning and paper set-up  

__**We 10-05-2017**__  
**10:00-11:30:** Switching to Anaconda and Python 2.7. The librosa import "no backend error" was solved by 'channeling' the conda install.  
**12:45-14:00:** Started writing preprocessing.py  
**18:45-20:00:** Tested and debugged the ID3 extraction and Mel spectrogramification. Both work. I only need to save the results to storage.

## Week 5 - ⚗️ Prototype
### *(total: 11 hrs)*
__**Th 11-05-2017**__  
**09:15-11:00:** Wrote a save_to_file and read_from_file function. Rein has fallen ill so the DAS introduction has been suspended.  
**13:30-15:30:** Tested the read- and write functions. Wrote and tested the k-fold cross validation function.  

__**Fr 12-06-2017**__  
**10:00-11:00:** Started writing [the report](https://www.sharelatex.com/project/58e0fb33c37936547fda88b0). The paper skeleton and introduction are done (for now).  
**13:15-13:30:** In my opinion, the Related Work section requires a lot more field orientation before it can be written. Anticipating on the academic English assignment that's due next week, I reckon it's best to continue the literary search now (in absense of the Tensorflow/DAS explanation) in order to facilitate the assignment. Tomorrow I will explore the first three pages of search results on the UvA library and Google Scholar for the keywords "music genre recognition" in combination with any one of: "electronic", "dance", "EDM", "neural network", "cnn", "convnet", "rnn" or "LSTM".  

__**Sa 12-05-2017**__  
**09:00-10:15:** Mapped all MIREX MGR submissions since 2014 (found in doc/further_literary_search.md). What suprised me is that only (Lidy, 2016) uses a network, and everyone else has used a SVM (which contradicts my earlier findings).  
**10:30-11:30:** Searched UvA library for "electronic", "dance" and "EDM". Another task that can be done while waiting for the DAS access is deciding on what subgenres to use, and to justify that choice.  
**13:00-14:00:** Searched UvA library for "cnn", "convnet", "rnn" and "LSTM".  

__**Su 13-05-2017**__  
**10:00-11:00:** Searched Google Scholar for "electronic", "dance" and "EDM".  
**11:00-12:00:** Searched Google Scholar for "neural network".  
**12:30-13:15:** Searched Google Scholar for "cnn".  
**18:00-19:00:** Searched Google Scholar for "convnet", "rnn" and "LSTM". All relevant papers I have found are listed below. I have only taken into account MGR research specifically (no related MIR tasks). I found more neural network approaches, although they remain countable (in contrast to SVM approaches). The 'Features' column consists of papers that discuss or define what features ought to be used/aimed at.  

| Electronic   | MIREX Veterans                            | Neural Networks                 | CNN        | RNN               | Hybrids                | Features                          | Other                                  |
|--------------|-------------------------------------------|---------------------------------|------------|-------------------|------------------------|-----------------------------------|----------------------------------------|
| Siva_2014    | Seyerlehner_2011                          | Klec_2015                       | Li_2010    | Dai_2016 (LSTM)   | Vogler_2016 (CNN-LSTM) | Ullrich_2014                      | Sturm_Evaluation (Evaluation problems) |
| Panteli_2016 | Wu_2013                                   | Goel_2014 (MLP Network)         | Lidy_2016  | Sigtia_2014       | Choi_2016 (CRNN)       | Geng_2016                         | Shi_2016 (ShuttleNet)                  |
| Kirss_2007   | Foleiss_2016                              | Siva_2014 (PNN)                 | Kong_2014  | Irvin_2016 (LSTM) |                        | Su_2014 (BoF eval)                | Iloga_2016 (Taxonomies)                |
| Chen_2014    | Lidy_2016 (CNN)                           | Prabhu_2014 ('Improved' NN)     | Costa_2017 |                   |                        | Reynolds_2007 (Rhythm importance) | Nasridinov_2014                        |
|              | Pikrakis_2013 (*Recreated by Sturm_2015*) | Alexandridis_2014 (RBF Network) |            |                   |                        | Gwardys_2014 (Image similarity)   | Sturm_2012 (1998-1012 Summary)         |
|              |                                           | Dai_2015 (Multi-DNN)            |            |                   |                        | Sturm_Features                    | Sturm_GTZAN (Dataset problems)         |
|              |                                           | Pikrakis_2013 (Rhythm Modeling) |            |                   |                        | Nakashika_2012                    | Widmer_2017 (Challenges)               |
|              |                                           | Reid_2014                       |            |                   |                        |                                   |                                        |
