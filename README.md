getConfig.py is responsible for reading the seq2seq.ini file, including the data set file path, line beginning and end symbols, model parameters and other configurations.

data_util.py converts the ask corpus and response corpus divided by the main program into a seq2seq file for further data processing by the data processor. For the entire system, generally the data preprocessor only needs to be run once.

execute.py contains functions: preprocess_sentence, create_dataset, max_length, read_data, tokenize, train, predict. The functions of these functions are to preprocess sentences, create data sets, obtain maximum length, read data, mark analysis, train models, and test patterns. .

seq2seqmodel.py defines a TF-based model: encoder Encoder, attention mechanism BahdanauAttention, decoder Decoder, loss function loss_function, iterative training function train_step.

app.py is the top-level visual display module, which contains functions: heartbeat, reply, and index. The functions of these functions are heartbeat, online dialogue, and homepage entry.

calculate_bleu.py is the performance evaluation module, which evaluates the algorithm results on the specified data set by calling the sentence_bleu function of the nltk.translate.bleu_score package.

