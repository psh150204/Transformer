# Attention Is All You Need (Transformer)

PyTorch implementation of the Transformer

### Requirements
* Python 3.6 >
* PyTorch 1.4 >
* Dataset: Multi30k
* Features: Model, Framework agnostic

### Results
* Prediction result : results/pred.txt
* 12 Epochs (with train loss 1.3013) : BLEU = 30.59, 63.6/38.1/25.5/17.0 (BP=0.956, ratio=0.957, hyp_len=11710, ref_len=12242)
* 21 Epochs (with train loss 0.7675) : BLEU = 32.81, 63.5/39.1/26.6/17.9 (BP=0.994, ratio=0.994, hyp_len=12174, ref_len=12242)
* 39 Epochs (with train loss 0.2197) : BLEU = 31.77, 62.9/37.6/25.3/17.1 (BP=1.000, ratio=1.007, hyp_len=12327, ref_len=12242)

### References
[1] Transformer in DGL (https://github.com/dmlc/dgl/tree/master/examples/pytorch/transformer)

[2] Multi30k (https://www.aclweb.org/anthology/W16-3210/)
