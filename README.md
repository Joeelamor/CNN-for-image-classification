CNN-for-image-classification
============================

Execution
---------

Executing the program using command:

    python3 main.py

For example, with folder like this

    |____final
    | |____main.py
    | |____input.py
    | |____data
    | | |____cifar-10-batches-py
  
We can execute like:

    cd OneDrive/UTDallas/Course6375 Machine Learning/final
    final git:(master) python3 main.py

The output with 10000 iterations will be produced. After every 50 iterations,
the accuracies of train and test will be printed.

Note
----

To execute the project, you firstly need to install libraries, `tensorflow`,
`sklearn`, `prettytensor`, `pickle` and `numpy`, and then download the
dataset `CIFAR-10` into project category.
In the code `main.py`, the iterations could be modified. And the execution
of program will be slow, so please be patient.
In the meanwhile, a file named `checkpoints` which is used to store
trained neurons before will be produced.
