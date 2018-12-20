# Sandwiching Transferred Models of Sparse Vectors in Deep Learning

The task assigned was to predict the trading action that the user is likely to read according to app contents about users by using transfer learning. While implementing deep learning architectures, we observed the non-convergence of common loss functions. Hence we proposed masked loss function which could mask the sparsity information in the loss function. Further, we proposed an autoen- coder based model by sandwiching two distinctly trained networks together. We then introduced a novel architecture, bi-directional autoencoders which produces two outputs for a single input vectors. The backward pass of the network based on the alternating gradient optimization methods. Finally, through our experiments, bi-directional autoencoders outperforms the rest of the given models.

![Poster](https://github.com/pranav-ust/transfer/blob/master/poster.png)

View the detailed report [here](https://github.com/pranav-ust/transfer/blob/master/sandwiching-transferred-models.pdf)

## Data and Models

The vectorized processed data is in `data` folder which contains `app.txt` showing app vectors, `l1.txt` showing coarse level category and `l2.txt` showing finer level category.

The training, validation and test splits are prepended with `tr`, `val` and `test` respectively.

The models are in the form of pickle files in `models` folder. We provide weights from:
1. app to coarse level vectors in `app2l1.pkl`
2. app to fine level vectors in `app2l2.pkl`
3. coarse to fine level vectors in `l12l2.pkl`

## Baseline

The baseline is a simple feedforward network with our masked loss. The code is in `baseline.py`. With MSE Loss the category accuracy is 0.2 and with masked loss the category accuracy is 0.22

## Autoencoder

This model is based on combining transfer learning with autoencoders in unsupervised learning and finetune the weights of this whole model. This model is in `autoencoders.py` which gives category accuracy with 0.35

## Bi-directional "Sandwiched" Autoencoder

Firstly, we back prop from L1 to App vectors(encoders) and L2 to App vectors(decoders) respectively to update the weights. Then we update the weights of sandwich layers using the gradients from both directions. Finally, we back prop from encoder part to decoder part to update the weights of decoder part, and in opposite, we back prop from decoder part to encoder part to update weights of encoders. This model is in `bidirectional.py` which gives category accuracy with 0.4

## Thoughts

It worked well with classification accuracy. However, when it came for recommendation, our MAP (Mean Average Precision) was extremely low (around 0.04), (code is in `precision`). Maybe we made some fundamental mistake, but at least we learned how to build architectures and transfer weights!
