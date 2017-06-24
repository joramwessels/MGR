## writing
	* abstract
	* theoretical background
		* add RNN functions
	* approach
		* Mel spectrogram computation
	* algorithm
		* code walkthrough (Tensorflow high level implementation)
	* Results
		* write (raw) results of (small discussion about them)
		* subgenre targeting
			* train on abs=1 vs train on abs=2 while eval on abs=1
		* networks
			* differences between networks on both datasets
		* dataset
			* dataset-1 vs dataset-2
				* number of classes/targets
					* abs=<1 vs abs=2 and abs=3 and abs=leafs (keeping evaluation the same)
				* dataset size
	* conclusion
		* Is CNN justified for this task?
		* best model
			* k2c2 better than small cnn?
			* crnn better than k2c2?
		* best targeting
		* best n/o classes and targets
			* multi-label fails
			* more classes is harder duh, but by keeping the evaluation the same?
		* justified to use small datasets in MGR?
		* answer research question and hypothesis
			* To what extent does subgenre targeting work?
			* Does the state of the art NN improve results? (compared to previous EDM MGR research)
	* discussion
		* memory
		* bugs
		* future work
			* learning separately on music aspects (rhythm, chord progression, melody)
			* drop recognition to get the best sample
			* CRNN
	* go through Taco's notations
