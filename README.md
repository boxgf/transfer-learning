# transfer-learning
Transfer Learning (policy) from multiple Linear PSRs

To pull out linearly independent column vectors (for the ADL algorithm while modelling the PSR), we use matlab's QR decomposition.
So make sure MATLAB engine is installed.
We have complied one of the runs with sklearn and stored the data using pickle.
Hence it is not necessary to have sklearn. you can comment import sklearn lines in qlearningAgents.py

The main transfer learning algorithm is implemented in the qlearningAgents.py (transferLearning agent)

To compile the transfer learning algorithm, use - python pacman.py -p tla -l newtarget
