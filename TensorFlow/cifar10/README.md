CIFAR-10 is a common benchmark in machine learning for image recognition.

http://www.cs.toronto.edu/~kriz/cifar.html

this project is used to move cifar10 code in tensorflow tutoials to meituan yun, prepare to use multi-gpus at last

1. first run singel cpu version
2. run with multi gpus

some problem may lead to failed:
1. data format not support by meituan yun, this may need to convert data to tensorflow default data format: tfrecords
2. tensorboard support may not work, this can be solved by debug

