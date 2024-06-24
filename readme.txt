Preprocessing:
In order to execute preprocessing steps (tokenization, stemming, stop-words removal) run the file named "_1tf_idf_save.py"
This file will require:
	- Dataset ("four_category_dataset.csv")
	- Stop-words file ("stopwords\stopwords.txt")
	- Special characters file ("special_chars\special-chars.txt")
This file will generate:
	- tf_idf_vector_4_classes.npy 
	- idf_vector_4_classes.npy
	- term_frequency_dataset.json
	- unique_terms_4_classes.txt
	- X_4car.npy
	- y_4car.npy
For efficiency, these files are provided in the zip folder.

Feature Selection:
In order to generate the graph of k features for mutual information, chi-squared and ANOVA f value run the file named "_2feature_selection.py". Execution may take hours based on the processing power of machine. 
For efficiency, generated graph is provided in the zip folder. 

Model Training:
To train the neural network model, execute the file named "_3model_training_testing.py". 
It will require:
	- X_4car.npy
	- y_4car.npy
It will generate:
	- trained model in the directory "models\nn_model_four.keras"
	- training and validation accuracy graph (accuracy.png)
	- training and validation F1 score graph (F1.png)
	- training and validation loss graph (loss.png)
	- confusion matrix (confusion_matrix.png)
For efficiency, trained model and generated graphs are provided in the zip folder.


Testing on new document:
To test the model on new data, execute the file named "_4test_on_new_document.py". 
It will require:
	- new_research_paper.txt (put title and/or abstract in this text file)
	- Stop-words file ("stopwords\stopwords.txt")
	- Special characters file ("special_chars\special-chars.txt")
	- X_4car.npy
	- y_4car.npy
	- unique_terms_4_classes.txt
	- idf_vector_4_classes.npy
It will output:
	-predicted class

	
