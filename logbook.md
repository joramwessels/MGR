# Logbook

## Week 1 - 📚 literary research
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
* Start implementing the preprocessing

## Week 4 - 🔍 Details
__**Th 03-05-2017**__  
**09:30-11:00:** I tried to make spectrograms of some sample beats, but I stumbled upon multiple librosa import errors.  
**13:00-15:30:** I made the spectrograms with an online tool instead, analysed some drum loops and individual samples, and found the properties presented below.In order to recognize a the smallest element, the FFT window will be 20ms with a stride of 10ms. The convolutional filter will be a vertical slice, 20kHz x 60ms, with a stride of 20ms.

| sample  | len (ms) | freq (kHz) |
|---------|----------|------------|
| kicks   | 50       | 0-0.25     |
| snares  | 100-250  | 0-15       |
| hi-hats | 100      | 4-20       |

__**Fr 04-05-2017**__  
