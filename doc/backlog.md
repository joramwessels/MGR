## programming
	* k2c2
	* CRNN
	* bias initialization
	* prediction

## evaluation
	* optimize a
	* optimize dropout
	* influence of training on subgenres (train on abs=2 or abs=3, test on abs=1), requires more genres

## writing
	* rationale for chillhop
	* house
	* elaborate RNN background
		* tanh
		* formulas
	* more on tensorflow
	* go through Taco's notations
	* CNN-RNN hybrids background (abort?)
	* finish house (prog and electro (and complextro?))
	* write hard dance (hardstyle and happy hardcore)
	* finish CRNN approach
	* write evaluation approach
	* code walkthrough in algorithm
	* write (raw) results of
		* networks
		* subgenre targeting
			* train on abs=1 vs train on abs=2 and eval on abs=1
		* number of classes/targets
			* abs=<1 vs abs=2 and abs=3 and abs=leafs
			* keep evaluation the same
		* dataset size
			* dataset-1 vs dataset-2
	* write conclusion
		* best model
			* k2c2 better than small cnn?
			* crnn better than k2c2?
		* best targeting
		* best n/o classes and targets
			* multi-label fails
			* more classes is harder duh, but by keeping the evaluation the same?
		* justified to use small datasets?
		* answer research question and hypothesis
	* write future work
		* learning separately on music aspects (rhythm, chord progression, melody)

## dataset
	* 100 tropical house tracks
	* split drum & bass tracks into liquid and neuro
	* 48+ future bass tracks
	* 48+ complextro
	
	dataset-2 (600-800 files in addition to 400 of dataset-1, 1000+ total (goal))  
		+ 3 - future bass (42, 48 more needed)  
		+ 2 - hardstyle (120) - not checked  
		+ 2 - happy hardcore (240) - not checked  
		+ 1 - dubstep (94) <- just dubstep  
		+ 1 - drum & bass (195, 174 to split into liquid or neuro)  
		+ 3 - future house (29) (100) - not checked  
		+ 3 - tropical house (0, 100 more needed)  
