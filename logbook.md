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

## Week 7 - 🐛 Debugging
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
* ✖ Finalize the assembly of the dataset (✖ deadline: Saturday)
* ✖ Get the CNN up and running (✖ deadline: Tuesday)
* ✖ Test the CNN on the dataset (✖ deadline: Wednesday)
* ✖ Start on the RNN implementation (✖ deadline: Thursday)

**16:00-18:00** Finished the Electronic Music paragraph.

__**We 31-06-2017**__  
**09:30-10:30:** The tensor dimension problem probably has to do with the input dimensions. The preprocessing parameters of (Choi, 2016) don't add up, but my test dataset so far consists of songs of insufficient length anyway. I will therefore first compose the final dataset and extract a test segment from that.  
**10:30-12:30:** Labeled Monstercat albums.  
**13:00-15:00:** Dennis urged me not to leave the GPU idle when reserved. Seeing as I am preoccupied with other tasks until Saturday I've terminated my reservation. This means that I won't be able to debug Tensorflow code until then. I've started preparing the presentation.  

## Week 8 - ⚗️ Prototype
### *(total: 21.5 hrs)*
__**Fr 02-06-2017**__  
**9:30-11:00:** Preparing- and performing presentation.  
**16:00-16:30:** Wrote- and handed in Academic English assignment 2

__**Sa 03-06-2017**__  
**10:00-11:30:** I tried to uncover Choi's exact preprocessing parameters, which can't be the exact numbers he reported. I fetched a small testing set from the actual dataset. I won't be able to develop and debug the network on the DAS-4 anymore, so I'll have to fix my local Anaconda after all.  
**12:30-14:30:** Reinstalling Anaconda and enriching the dataset in the mean time. After manually adding the python executable to the windows path I can run the interpreter in powershell (not in mingw bash). Still, `conda`, `anaconda`, and `pip` are not recognized as commands. The installer usually takes care of these things itself.  
**14:30-16:00:** Wrote a paragraph on deep house, and half a paragraph on hip hop. Found enough sources to finish the dataset and started assembling them. I check each track by hand to see if it is labeled correctly and if it is a typical (i.e. non-ambiguous) example of the genre.

__**Su 04-06-2017**__  
**12:30-14:30:** Finished the paragraph on hip hop. Prepared paragraphs on house and dubstep.  

__**Tu 06-06-2017**__  
**10:00-10:45:** Anaconda works in the special Anaconda prompt. I installed `librosa`, `ffmpeg`, and `mutagen` using `conda install -c conda-forge`, then created a python 3.5 environment for Tensorflow, installed Tensorflow, and reinstalled the aforementioned packages in the environment. Everything finally works, but Tensorflow urges me to [activate SSE instructions](https://stackoverflow.com/questions/43134753/tensorflow-wasnt-compiled-to-use-sse-etc-instructions-but-these-are-availab), which I might do later.  
**10:45-12:00:** Solved small bugs in `mgr_utils.py` and `preprocessing.py`, and fixed the dimension problem in the network.  
**12:45-16:45:** Debugged the system. Ended with a json error saying that a list "is not JSON serializable".  

__**We 07-06-2017**__  
**09:45-11:45** Debugged the system. Since a bug forced me to rewrite the memory allocation code, I also started rewriting the way the allocation works to make it more resource efficient.  
**12:30-14:30** Finished rewriting the memory allocation.  
**19:00-20:00** Debugged the CNN up untill the last 2 lines of the network. The last bug was `Cannot feed value of shape (6,) for Tensor 'Placeholder_1:0', which has shape '(?, 10)'`. Tomorrow I will also test the `Dataset` object extensively, as it is presumably still very buggy.  
**20:30-22:00** Checked the trap entries of the dataset while writing a paragraph on future bass.


## Week 9 - 🌡️ Results
### *(total: 24.75 hrs)*
__**Th 08-06-2017**__  
**9:30-11:30** Experimented with placeholders to try and solve the dimension bug.  
**13:00-14:15** Eighth meeting. There appears to be something wrong with our interpretation of Choi's k2c2 network. If the layers indeed do have (20, 41, 41, 62, 83) filters each, the output would be more than 173 million variables. Rein will review the paper tomorrow, and in the meantime I will implement a simpler network of 3 conv layers and a softmax layer. The deadlines for next week:  
* ✔ Finish dataset (Monday)
* ✔ CNN evaluation (Tuesday)

__**Fr 09-06-2017**__  
**12:30-15:00:** Implemented the 3-layer CNN. It runs, but reports only 0.0 accuarcies. The model saving procedure still errors, so I will need to check the Tensorflow manual to learn how it works.  

__**Sa 10-06-2017**__  
**09:30-12:00:** Read some Tensorflow `Saver` tutorials and changed the code. Now that I understand Tensorflow better, I might need to restructure my network code for clarity. I got stuck on an error about restoring saved variables, and three consecutive errors about variables of the same name already being in place. I'll need to debug the saving procedure next.  
**12:30-14:00:** Finished *Dataset-1* (4GB), which includes about 100 tracks for each Electro House, Progressive House, Chillhop and Trap. Only Electro House includes subgenres (Complextro & Big Room). All Electro House tracks and Trap tracks have been screened, the other 2 collections were tagged by reliable sources.  
**14:00-15:00:** Looked into the saving problem. Mailed my supervisors for help.  
**15:00-15:30:** Prepared 50 Future Bass tracks for the next dataset update.  
**19:30-21:00:** Wrote the paragraph on dubstep.  

__**Su 11-06-2017**__  
**10:00-12:00:** Fixed the saver by calling `restore` using the returned filename. Tried to fix the lost variables by naming the accuracy, x and y tensors individually using the optional `name` argument, and referencing them by their name rather than calling `get_tensor_by_name` (which probably creates a new tensor). I removed the weights and biases from the saver constructor to solve the uninitialized tensor errors. **The network now runs without errors!!!🎉🎊🎈** (but returns extremely poor results, of course). I'll run it on *Dataset-1* during lunch.  
**12:30-14:15:** Preprocessing ~400 files only took 11 minutes, and training using 5-fold cross validation less than 2 minutes. Accuracy results are nonzero, but terrible nonetheless. A few files errored due to encoding issues, but these were handled appropriately. There is an issue with the average accuracy calculation I'll need to solve. I've added the `monitor` log option and started writing an argparse main function for easy access during evaluation.  

__**Mo 12-06-2017**__  
**15:30-17:30** Continued working on the argparse functions.  
**18:30-20:00** Restructured the code, and continued on the argparse functions.  

__**Tu 13-06-2017**__  
**13:30-18:30:** Looked into the accuracy results. Fixed some bugs with the Dataset class. Fixed the JSONDecodeError. Improved the logging format. Finished the argparse functions (but they don't completely work according to plan yet).  

## Week 10 - 🔗 Loose Ends
### *(total: 38.5 hrs)*
__**Th 15-06-2017**__  
**09:30-12:00** Copied an RNN implementation from a tutorial and adapted it into my system. It doesn't run yet due to a dimensionality problem.  
**13:00-14:30** Ninth meeting. My misunderstanding of the Tensorflow dimensions has been alleviated. CNNs aren't trees, but networks (as the name suggests). We looked into the code and compared it to Rein's experience with Tensorflow. He argued the biases should be initialized as small positive floats when used in combination with ReLu activation. The final implementation will be a CNN. Next week thursday I will need to have written most of the paper, and have a network that's able to learn genres. When the CNN works, I can tweak the parameters to improve results, and when there is time I can try to replace the fully connected layer with an RNN. My backlog for coming week:  
* ✔ Find out why the accuracy values are invalid
	* ✔ Loss function (Softmax)?
	* ✔ `correct_pred` adapted to multi-label targets
	* ~~Implement `prediction.py` to aid debugging~~
* ✔ Minor problems that can't be the cause
	* ✔ Replacing dropout
	* ~~Bias initalization (0.1)~~
* ✖ Writing
	* ✖ Explain tanh in LSTM paragraph
	* ✖ Write more on Tensorflow
	* ✖ Add RNN formuulas
  
**16:00-18:30** Rein informed me that `tf.nn.softmax_cross_entropy_with_logits` only works with one-hot (uni-label) targets. Sigmoid does work with multi-label-, but not with multi-class targets. I've changed the `correct_pred` computation to  
```
correct_conf = tf.where(tf.cast(y, tf.bool), pred, tf.zeros(tf.shape(y)))
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(correct_conf, 1))
```  
**19:00-20:00** Elaborated and updated the paragraph on preprocessing, but I got confused about the Mel spectrogram computation.  

__**Fr 16-06-2017**__  
**09:30-12:00** Looked into solutions to the multi-label problem. Apparently sigmoid can handle multi-class as well, so I adopted `tf.nn.sigmoid_cross_entropy_with_logits` as the new loss function. I fixed some simple bugs that jeopardized the entire execution. The model now classiifies with an accuracy of 35.19%, whereas the random prediction is 37.50% accuracy. The execution details can be found in [logs/FirstCNNResults.log](https://github.com/joramwessels/MGR/blob/master/logs/FirstCNNResults.log). I reinstated the dropout and improved the accuracy with 7.24% with dropout=0.75 (acc=42.43%). This trumps the random baseline by a slim margin, but still lacks expertise. The executions can be found in [logs/DropoutCnn.log](https://github.com/joramwessels/MGR/blob/master/logs/DropoutCnn.log).  

__**Sa 17-06-2017**__  
**09:30-10:30** Futher dropout benchmarking varies in results. The testing algorithm doesn't work when calling it explicitly. This might mean that during training the testing procedure uses the last model instead of the assigned .meta file. Yesterday's baseline evaluation was wrong; 88% of the tracks only have 2 labels. That makes the baseline 12.5 * 2.22 = 27.75%. However, the proportions of the dataset matter in computing the random baseline. 211/394 tracks are labelled as house, which would therefore always be his best guess. The zero rule baseline accuracy of dataset-1 is 53.55%, above any of our results so far.  
**13:30-17:30** Started writing the paragraph on the CNN implementation. Created a visual for the CNN explanation. Wrote the paragraph on the dataset assembly.  

__**Su 18-06-2017**__  
**09:00-11:30** I restructured the code in `cnn.py` to fit my current understanding of Tensorflow and to solve the training issue. It caused a new error, `InvalidArgumentError: You must feed a value for placeholder tensor 'x_1' with dtype float`, while training the _second_ CV fold. I added a call to `tf.reset.default.graph()` which fixed the error, but accuracy results for train- and test set are now identical for every fold. The test function already included a graph reset.  
**13:00-14:00** Looked into Tensorflow RNN implementations.  
**14:00-16:00** Wrote most of the paragraph on drum & bass.  
**19:45-21:15** Finished the drum & bass part and wrote the intro to house  

__**Mo 19-06-2017**__  
**09:30-12:30** Separated the functions for accuracy, optimization and saving so that the testing procedure can now build a new model instead of loading one from storage. I have confirmed that the weights are successfully being restored. Yet, the new accuracy function freezes when called individually, and returns `None` when used in `training.py`. When training using the old testing function, it returns surprisingly acceptable accuracy scores (~50%).  
**14:30-17:00** The saver problems are solved. It was a combination of a lot of things relating to the dataset fold, test/train sets, and Tensor reference. The accuracy is always between 40 and 50 percent.  

__**Tu 20-06-2017**__  
**10:45-12:30** I wrote- and unit tested a function that determines the abstraction of labels used. On just the highest abstraction (distinguishing house from hip hop), it scored an accuracy of 81.37%! The random baseline for this task is 50% and the zero rule (0R) is equal to last tests: 53.55%. On one fold it even scored a perfect 100%. The execution can be found in [logs/AbsCNN.log](https://github.com/joramwessels/MGR/blob/master/logs/AbsCNN.log). I am now certain that this exact network will be used for the project, which means I will start explaining it thoroughly in the report, testing it with different parameters, and write about the results as well.  
**13:00-15:15** Tested with other abstraction levels. Also implemented a function that enables you to train on one abstraction and evaluate using another. This way, the influence of training on subgenres can be determined by abstracting to higher taxonomical levels during evaluation.  
**18:30-20:00** Wrote about preprocessing and models. I haven't implemented all three, but I intend to and so they are relevant to the project.  

__**We 21-06-2017**__  
**10:30-12:00** Debugged the evaluation abstraction function.  
**12:30-14:30** Finished `Dataset-2` (1351 tracks, 12.4 GB) contiaining 100 Chillhop, 70 D&B, 100 Dubstep, 109 Electro House, 122 Future Bass, 129 Future House, 226 Happy Hardcore, 120 Hardstyle, 100 Liquid Funk, 100 Neurofunk, 100 Progressive House, and 75 Trap tracks.  
**17:30-19:30** Added dataset-1 to the appendix, and started implementing the k2c2 network.  

## Week 11 - 📥 Handing In
### *(total: 18.5 hrs)*
__**Th 22-06-2017**__  
**10:30-12:00** k2c2 works (so far).  
**13:00-14:00** Tenth meeting. There is a very acceptable and very feasible to this project. We will drop the CNN-RNN hybrids part and focus on training a CNN on electronic subgenres. That means that all programming is finished and that I will now start evaluating my two networks and two datasets for hierarchical targeting, learning rate, and dropout. The report draft is due on Monday. Today and tomorrow are reserved for evaluation, and the weekend for writing. On Monday we will go over the draft with the three of us. Until then I will report daily on my progress.  
**15:00-17:00** Wrote an evaluation module that loops over all parameters (model, dataset, training abstraction, learning rate & dropout) so I can evaluate overnight.  
**18:30-20:00** Ran some tests, debugged the evaluation module. `Dataset-2` is clearly a lot bigger than the first. My laptop doesn't have the juice to run something like this overnight. I will try to run it on the DAS, and otherwise on the CPU of my game pc.  
**21:00-22:30** Applied for a GPU on the DAS-4. scp'ed my datasets to the server. Built a new environment on the DAS (CPU) node in a `screen` and installed Tensorflow and the other dependencies in an effort to run the evaluation regardless of the GPU reservation, but it can't find the Tensorflow runtime.  

__**Fr 23-06-2017**__  
**08:30-12:00** Got a GPU, the Tensorflow installation works, compatibility issues were surprisingly few (just `__next__()` and a path separator). The evaluation runs and MY GOD it's fast. I've added a results logger for convenience. The abstractions all work, the networks work, but `Dataset-2` ran into a `Resource exhausted: OOM when allocating tensor` error. I've applied batching for the test set to resolve it. The amount of results created by 2 models * 2 datasets * 5 abstractions * 3 learning rates * 3 dropouts is a little overwhelming (180). I wanted to write a function that stores them in a matrix so I can load it and evaluate per dimension, but that might be too much of a tangent (the json module already posed a problem, so I'd better not waste my time on that right now).  
**12:30-15:30** The evaluation still ran smoothly. I started writing the evaluation paragraphs in the report, and when I came back the same resource exhausted error blocked the training. The resource in question was storage. I cleared the models folder and added a part to the script that deletes models after testing.  
**15:30-18:30** Tested k2c2 on dataset-1 while writing down the results of the small CNN and the k2c2. When starting the tests on dataset-2 another resource exhaustion occured, this time specifying it was memory. The second dataset is 3.1GB (theres 4GB RAM on DAS-4), and both the train- and test data are being fed in batches of 40 samples a time. Unless python/anaconda/Tensorflow needs more than half a GB I wouldn't know how to solve this. On my own pc it runs fine, but slow (RAM usage indeed peaks at 13GB). I will therefore do a different type of evaluation on the second dataset. The first dataset fascilitated subgenre targeting, whereas the second dataset can be used to optimize- and compare the two models.  
**20:30-22:00** Completely finished the theoretical background by adding the paragraph on hard dance, finishing the one on house, and elaborating a little bit more about Tensorflow.  

__**Sa 24-06-2017**__
**09:30-** I found out Choi used a dropout of 0.5 after every maxpool layer, so I adjusted k2c2 to match that. However, there is still a bug in the code (even before this change) that might prevent proper results from showing. The accuracies produced are very disappointing indeed.
