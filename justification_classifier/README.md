# cnn_run_classifier
- Here, a CNN model with three convolution layers and one linear layer is used.
- Data loading, preprocessing and training loop with loss calculation is implemented.


 - To run with CUDA:

    ```python cnn_run_classifier.py --data_dir justification --output_dir cnn_output --do_train --do_eval```

- To run without CUDA:

 ```python cnn_run_classifier.py --data_dir justification --output_dir cnn_output --do_train --do_eval --no_cuda```