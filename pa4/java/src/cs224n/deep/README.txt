

Below are the classes implementing training and testing for different models.

Baseline.java:            Baseline NER model based on exact string match
WindowModel.java:         Base Neural Network  using SGD with decaying learning rate
WindowModel_Cap.java:     Base Neural Network with Capitalization Indicator using standard SGD
WindowModel_Batch.java:   Base Neural Network  with MiniBatch SGD
WindowModel_Deeper.java:  Deeper Network with one more hidden layer using standard SGD
