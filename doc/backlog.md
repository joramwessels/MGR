## evaluation
	* optimize a
	* optimize dropout
	* influence of training on subgenres (train on abs=2 or abs=3, test on abs=1)

## writing
	* add RNN functions
	* go through Taco's notations
	* -> write evaluation approach
	* code walkthrough in algorithm (Tensorflow high level implementation)
	* write (raw) results of (small discussion about them)
		* networks
		* subgenre targeting
			* train on abs=1 vs train on abs=2 and eval on abs=1
		* number of classes/targets
			* abs=<1 vs abs=2 and abs=3 and abs=leafs
			* keep evaluation the same
		* dataset size
			* dataset-1 vs dataset-2
	* write conclusion
		* Is CNN justified for this task?
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
		* Hybrid
