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
* ✖ Put in a minimum of 30 hours a week (or 5 hours a day)
* ✖ Get comfortable with the DAS (✖ attend the introduction on Thursday)
* ✔ Start to write the paper, especially the ✔ skeleton, ✔ intro, ✔ related work and ✖ preprocessing
* ✔ Get as far as possible on implementing a simple CNN in Tensorflow
	* ✔ Change to Python 2.7 and Anaconda
	* ✔ Install Librosa
	* ✔ Preprocessing
	* ✖ CNN network

**10:30-12:30:** Planning and paper set-up  

__**We 10-05-2017**__  
**10:00-11:30:** Switching to Anaconda and Python 2.7. The librosa import "no backend error" was solved by 'channeling' the conda install.  
**12:45-14:00:** Started writing preprocessing.py  
**18:45-20:00:** Tested and debugged the ID3 extraction and Mel spectrogramification. Both work. I only need to save the results to storage.

## Week 5 - 📝 Writing
### *(total: 15.25 hrs)*
__**Th 11-05-2017**__  
**09:15-11:00:** Wrote a save_to_file and read_from_file function. Rein has fallen ill so the DAS introduction has been suspended.  
**13:30-15:30:** Tested the read- and write functions. Wrote and tested the k-fold cross validation function.  

__**Fr 12-06-2017**__  
**10:00-11:00:** Started writing [the report](https://www.sharelatex.com/project/58e0fb33c37936547fda88b0). The paper skeleton and introduction are done (for now).  
**13:15-13:30:** In my opinion, the Related Work section requires a lot more field orientation before it can be written. Anticipating on the academic English assignment that's due next week, I reckon it's best to continue the literary search now (in absense of the Tensorflow/DAS explanation) in order to facilitate the assignment. Tomorrow I will explore the first three pages of search results on the UvA library and Google Scholar for the keywords "music genre recognition" in combination with any one of: "electronic", "dance", "EDM", "neural network", "cnn", "convnet", "rnn" or "LSTM".  

__**Sa 13-05-2017**__  
**09:00-10:15:** Mapped all MIREX MGR submissions since 2014 (found in `doc/further_literary_search.md`). What suprised me is that only (Lidy, 2016) uses a network, and everyone else has used a SVM (which contradicts my earlier findings).  
**10:30-11:30:** Searched UvA library for "electronic", "dance" and "EDM". Another task that can be done while waiting for the DAS access is deciding on what subgenres to use, and to justify that choice.  
**13:00-14:00:** Searched UvA library for "CNN", "ConvNet", "RNN" and "LSTM".  

__**Su 14-05-2017**__  
**10:00-11:00:** Searched Google Scholar for "electronic", "dance" and "EDM".  
**11:00-12:00:** Searched Google Scholar for "neural network".  
**12:30-13:15:** Searched Google Scholar for "CNN".  
**18:00-19:15:** Searched Google Scholar for "ConvNet", "RNN" and "LSTM". All relevant papers I have found are listed below. I have only taken into account MGR research specifically (no related MIR tasks). I found some more neural network approaches, although they remain countable (in contrast to SVM approaches). The 'Features' column consists of papers that discuss or define what features ought to be used/targeted.  

| Electronic   | MIREX Veterans                            | Neural Networks                 | CNN        | RNN               | Hybrids                | Features                          | Other                                  |
|--------------|-------------------------------------------|---------------------------------|------------|-------------------|------------------------|-----------------------------------|----------------------------------------|
| Siva_2014    | Seyerlehner_2011                          | Klec_2015                       | Li_2010    | Dai_2016 (LSTM)   | Vogler_2016 (CNN-LSTM) | Ullrich_2014 (CNN boundary det.)  | Sturm_Evaluation (Evaluation problems) |
| Panteli_2016 | Wu_2013                                   | Goel_2014 (MLP Network)         | Lidy_2016  | Sigtia_2014 (no MGR) | Choi_2016 (CRNN)       | Geng_2016 (CNN hierarchy)         | Shi_2016 (ShuttleNet)                  |
| Kirss_2007   | Foleiss_2016                              | Siva_2014 (PNN)                 | Kong_2014  | Irvin_2016 (LSTM) |                        | Su_2014 (BoF eval)                | Iloga_2016 (Taxonomies)                |
| Chen_2014    | Lidy_2016 (CNN)                           | Prabhu_2014 ('Improved' NN)     | Costa_2017 |                   |                        | Reynolds_2007 (Rhythm importance) | Nasridinov_2014 (automatic feature ext)|
|              | Pikrakis_2013 (*Recreated by Sturm_2015*) | Alexandridis_2014 (RBF Network) |            |                   |                        | Gwardys_2014 (Image similarity)   | Sturm_2012 (1998-1012 Summary)         |
|              |                                           | Dai_2015 (Multi-DNN)            |            |                   |                        | Sturm_Features (deconvolution)    | Sturm_GTZAN (Dataset problems)         |
|              |                                           | Pikrakis_2013 (Rhythm Modeling) |            |                   |                        | Nakashika_2012 (how2use CNN)      | Widmer_2017 (Challenges)               |
|              |                                           | Reid_2014                       |            |                   |                        |                                   |                                        |

__**Tu 16-05-2017**__  
**12:45-13:45:** Adjusted the introduction to the new information found in my literary search. Wrote the structure of the Related Work section.  
**14:00-15:00:** Started writing the Related Work section

## Week 6 - 🚚 Logistics
### *(total: 25 hrs)*
__**Th 17-05-2017**__  
**09:00-10:00:** Continued writing the Related Work

__**Fr 19-05-2017**__  
**10:00-10:30** Fifth meeting. The aforementioned introduction of the DAS is not required to be able to start programming the network.
Coming week will therefore likely see the first CNN prototype. If any complications do arise, I can call upon *Bas Visser*'s experience
or address them during the next meeting. My progress on the report was below expectations. I have therefore decided to write for one hour
every business day. The coming 6 weeks (week 6-11) are reallocated as follows. Week 6-7 are concerned with implementing the algorithm,
while keeping track of the steps I take in the report. Week 8-10 are reserved for the evaluation and for writing the corresponding sections of the report.
The start of week 11 is my self-imposed deadline for the end report. The report is due Friday June 30 of that week. Our next meeting
is scheduled on May 24, and before then I need to:
* ✖ Implement a CNN prototype
* ✔ Include a part about NNs in the report
* ✔ Finish the Related Work section (elaborating on each citation)
* ✖ Write the subsection on preprocessing
* ✖ Write a subsection on Tensorflow
* ✔ Start constructing a database (and justify choices)

|          | Daytime      | Evening          |
|----------|--------------|------------------|
| Today    | Related Work | Related Work     |
| Saturday | Start CNN    | NN intro         |
| Sunday   | CNN          | Preprocessing    |
| Monday   | CNN          | Tensorflow       |
| Tuesday  | Finish CNN   | Assemble Dataset |

**13:00-15:30:** Continued writing the Related Work and planned the coming week.  
**18:00-20:00:** Continued writing. Explored the various datasets used.  
**20:30-22:00:** Finished the Related Work section, but I'll need to finish the bibliography before I can hand it in.  

__**Sa 20-05-2017**__  
**10:00-11:30** Explored how to install Tensorflow, figured I'd best use the installation on the DAS, tried to log in to the DAS and failed, installed the UvAVPN and tried agian, failed again.  
**14:00-16:00:** Created half the bib file, which took a lot more time than expected.  
**16:30-18:15:** Finished the bib file and handed in Academic English assignment 1. I'm already one day behind on the schedule from yesterday.  

__**Su 21-05-2017**__  
**13:00-14:30:** Got access to the DAS (using the UvA-MN head node) and installed Anaconda. I followed the instructions in `/home/koelma/tensorflow_pkg/readme.txt` but got stuck trying to install wheel, upgrade pip and install Tensorflow.  
**15:30-17:00:** Explored dataset options. I found sources for  the popularity of subgenres, I inventarized my personal collection as of now, and I listed sources with which to enrich the collection.  
**19:00-20:00:** Restructured the skeleton (by looking at old dissertations supervised by Taco or Rein) and wrote the neural network introduction.  
**20:30-22:00** Wrote the CNN background, started the RNN background and constructed a visual of an RNN. I've asked Bas and Rein for help with the DAS. I still haven't completed what I should've completed last night; the DAS is a complex environment.

__**Mo 22-05-2017**__  
**08:00-11:00:** Awaiting a response about the DAS, I've continued writing the report. I finished the paragraphs on the LSTM and GRU, but I've added a header for a paragraph on CNN-RNN hybrids that I might write some other time. I also created another clearer visual of RNNs.  
**12:30-14:00:** Used Bas' tips to install Tensorflow. I cloned my repos and tried to test preprocessing.py on the new machine but ran into the old "no backend error" problem. I've done a crash course on command line Emacs and learned how to use `screen`. I (think I) have requested a GPU node using `qrsh`, but it apperas that will take a while. I am unsure if I am supposed to reserve it only once, or do so everytime I want to run my code. I created cheat sheets for everything I did on the way. I'll need to fix the `ffmpeg` import using conda channels, but I can't seem to remember the command that lists all channels.  
**16:00-17:00:** Wrote something about the DAS, and struggled a little with LaTeX headers and layout.  

__**Tu 23-05-2017**__  
**09:00-10:30:** Looked into the `Librosa`/`audioread` error. Managed to install `ffmpeg`. Apparently there is no Git installation on the GPU node?  
**12:00-13:00:** For some reason the DAS Python console has no `readline` (shouldn't I have noticed that earlier if it was already like this, or did I break something?). I have tried to fix it by installing `readline` and creating a `.pythonstartup` file, but it only started throwing more errors.

__**We 24-05-2017**__  
**10:00-10:30:** Sixth meeting. I shouldn't waste any more time on logistical DAS problems. I will proceed to write the CNN locally using a CPU based Tensorflow installation, and port it to the DAS when it's finishen. By next week Tuesday I intend to have:
* ✖ A working local prototype
* ✔ Written the Preprocessing paragraph
* ✔ Written the Tensorflow paragraph
* ✖ Added more detail to the NN paragraphs

**17:45-19:00:** I fixed the readline problem by channeling the install through `conda-forge`. Sent supervisors a ShareLaTeX invite. Increased the line spacing. I installed Tensorflow, which is apparently only available for Python 3.5 on Windows, so I hope it won't result in too many problems when porting to the DAS.

## Week 7 - ⚗️ Prototype
### *(total: 23.5 hrs)*
__**Th 25-05-2017**__  
**9:45-11:30:** My local Anaconda installation has stopped responding after installing Tensorflow. Since I fixed `readline`, I continued on the DAS. I was unable to git push, supposedly because I cloned using https, so I switched it to SSH. I was then also unable to pull because I haven't set up my SSH keys. I'll stick to HTTPS, write my code locally, and manually copy changes I make while debugging.  
**12:30-13:30:** I need to fix this bug before I can access my dataset:  
```
>>> isfile('/var/scratch/jwessels/MGR/doc/experiments/Techno.mp3')
True
>>> load('/var/scratch/jwessels/MGR/doc/experiments/Techno.mp3')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/var/scratch/jwessels/anaconda/lib/python2.7/site-packages/librosa/core/audio.py", line 107, in load
    with audioread.audio_open(os.path.realpath(path)) as input_file:
  File "/var/scratch/jwessels/anaconda/lib/python2.7/site-packages/audioread/__init__.py", line 109, in audio_open
    return ffdec.FFmpegAudioFile(path)
  File "/var/scratch/jwessels/anaconda/lib/python2.7/site-packages/audioread/ffdec.py", line 150, in __init__
    self._get_info()
  File "/var/scratch/jwessels/anaconda/lib/python2.7/site-packages/audioread/ffdec.py", line 206, in _get_info
    raise IOError('file not found')
IOError: file not found
```  
**14:00-15:00** Posted [a question](https://stackoverflow.com/questions/44179007/librosa-load-file-not-found-error-using-anaconda) about it on StackOverflow, and in the meantime introduced myself to Tensorflow.

__**Fr 26-05-2017**__  
**10:00-13:00** I now completely understand the k2c2 algorithm in (Choi, 2016) I am trying to recreate. Tensorflow appears quite intuitive and easy, although I haven't executed any code so far. Yesterday's StackOverflow question hasn't yielded any answers so far. I just asked [another one](https://stackoverflow.com/questions/44199665/tensorflow-cnn-mnist-example-weight-dimensions) regarding the weight dimensions in a CNN Tensorflow example.  
**14:30-15:30** These two problems prevent me from finishing the first code, and running the code. I've written a little on Tensorflow in the report.  
**21:00-22:00:** Finished the paragraph about Tensorflow (wouldn't know what else to write about it). Searched for CNN formulas to add to the NN paragraphs. I found some, but I reckon I'll need a quick refresh on backprop in MLPs first before looking into CNN backprop. I am well on schedule, but I do need to fix my logistical issues by tomorrow in order not to stall the actual programming for too long.

__**Sa 27-05-2017**__  
**09:30-10:30** Rein helped me tackle the first bug. The `ffmpeg` package was already installed through `conda-forge`, but installing it again using `conda install -c menpo ffmpeg=3.1.3` did the trick.  
**12:45-16:00:** StackOverflow answered my second question. I've finished the CNN (conceptually, debugging awaits).  
**16:45-17:45:** Looked into data feeding. Wrote a batch dispenser. Tomorow I'll do error handling, finish the data input and test it on the DAS.  
**20:30-21:30:** Got my backprop straight and looked into CNN backprop.

__**Su 28-05-2017**__  
**14:00-15:30** Started exception handling.  
**16:00-18:00** Sank way too much time into exception handling. I used the python distribution on the DAS, but the WiFi here shut down periodically, after which I had to establish the VPN and SSH connections again before redeclaring my variables in the pyton console.  
**21:00-23:30** Started writing Electronic Music theoretical background intro. Mailed my supervisors for feedback.

__**Mo 29-05-2017**__  
**10:00-12:30** Wrote a `Dataset` class to handle data feeding.  
**13:00-14:30** Finished data feeding. Tested everything on the DAS. I managed to debug `mgr_utils.py` by running `preprocessing.py`, but due to a Tensorflow installation problem, `training.py` and `Choi2016.py` can't be debugged yet. I emailed Bas for help.  
**15:45-17:00** Bas once again saved the day. I've debugged a couple of errors but got stuck on a tensor dimension related problem.

__**Tu 30-05-2017**__  
**10:00-10:30** Seventh meeting. Things have turned around, and if I keep this tempo up everything will work out fine. My writing thus far was in accordance with Taco's standards. Tomorrow and the day after are reserved for writing the presentation and studying for my other exam. On Friday I will give the presentation and do the exam, which leaves little time left to do some actual work until this weekend. Nonethelss, before the end of next week I intend to
* Finalize the assembly of the dataset (deadline: Saturday)
* Get the CNN up and running (deadline: Tuesday)
* Test the CNN on the dataset (deadline: Wednesday)
* Start on the RNN implementation (deadline: Thursday)

**16:00-18:00** Finished the Electronic Music paragraph.

__**We 31-06-2017**__  
**09:30-10:30:** The tensor dimension problem probably has to do with the input dimensions. The preprocessing parameters of (Choi, 2016) don't add up, but my test dataset so far consists of songs of insufficient length anyway. I will therefore first compose the final dataset and extract a test segment from that.  
**10:30-12:30:** Labeled Monstercat albums.  
**13:00-15:00:** Dennis urged me not to leave the GPU idle when reserved. Seeing as I am preoccupied with other tasks until Saturday I've terminated my reservation. This means that I won't be able to debug Tensorflow code until then. I've started preparing the presentation.  

## Week 8 - 🌡️ Results
### *(total: 2 hrs)*
__**Fr 02-06-2017**__  
**9:30-11:00:** Preparing- and performing presentation.  
**16:00-16:30:** Wrote- and handed in Academic English assignment 2

__**Sa 03-06-2017**__  
**10:00-11:30:** I tried to uncover Choi's exact preprocessing parameters, which can't be the exact numbers he reported. I fetched a small testing set from the actual dataset. I won't be able to develop and debug the network on the DAS-4 anymore, so I'll have to fix my local Anaconda after all.  
**12:30-14:30:** Reinstalling Anaconda and enriching the dataset in the mean time. After manually adding the python executable to the windows path I can run the interpreter in powershell (not in mingw bash). Still, `conda`, `anaconda`, and `pip` are not recognized as commands. The installer usually takes care of these things itself.  
**14:30-16:00:** Wrote a paragraph on deep house, and half a paragraph on hip hop. Found enough sources to finish the dataset and started assembling them. I check each track by hand to see if it is labeled correctly and if it is a typical example of the genre.

__**Su 04-06-2017**__  
**09:30-:** 
