# System Identification by RNN

Different from the existed repo, this RNN is trained on multiple trajectories of underwater robots. To reduce the computation cost, we use high dimension tensor to store the data. So the original RNN module offered by Pytorch can't be used normally, we redefine the RNNCell in the [\model\_\_init\_\_.py](https://github.com/dream-oyh/System-Identification-by-RNN/blob/master/model/__init__.py).

- [x] TODO: code the training process of the model
- [x] TODO: improve the presentation of prediction results and improve the format of std_out
- [ ] TODO: add more cell, like the GRU cell and LSTM cell to enhance the utility of this repo and test more RNN-Like models.

P.s.

[\datasets_test](https://github.com/dream-oyh/System-Identification-by-RNN/tree/master/datasets_test) is a part of our own datasets for test. Each csv file includes a trajectory sampled by 10Hz.

[main.py](https://github.com/dream-oyh/System-Identification-by-RNN/blob/master/main.py)is the train entrance.

[\output](https://github.com/dream-oyh/System-Identification-by-RNN/tree/master/output) will store the log message and the image of model prediction results.
