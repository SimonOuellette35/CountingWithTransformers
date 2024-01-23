# CountingWithTransformers
Code for paper ["Counting and Algorithmic Generalization with Transformers"](https://arxiv.org/abs/2310.08661)

* trainStd_Transformer_Count.py: baseline Std-Transformer-Count experiment that attempts to learn to count pixels in a grid with a standard transformer.
* trainNo_LayerNorm_Count.py: the No-LayerNorm-Count experiment that succeeds at learning to count pixels.
* trainNo_LayerNormV2_Count.py: a variant of No-LayerNorm-Count that uses a 2-layer Feed-forward NN, does not generalize as well.
* trainLayerNorm_SA_Count.py: LayerNorm-SA-Count experiment.
* trainLayerNorm_SAV2_Count.py: a variant of LayerNorm-SA-Count that uses a 2-layer Feed-forward NN.
* trainLayerNorm_FF_Count.py: LayerNorm-FF-Count experiment.
* trainLayerNorm_FFV2_Count.py: a variant of LayerNorm-FF-Count that uses a 2-layer Feed-forward NN.
* trainLayerNorm_Identity.py: LayerNorm-Identity, experiment with generalizing on the identity function.
* trainNo_LayerNorm_Identity.py: No-LayerNorm-Identity, experiment with generalizing on the identity function.
* trainSeqtoSeqCounterV2.py: an experiment to show that a multi-layer standard transformer cannot even learn to count sequentially (removed from latest version of paper as being redundant)
* trainUTSeqtoSeqCounterV2.py: a failed attempt at getting the Universal Transformer to learn to count sequenitally (removed from latest version of paper due to being inconclusive)
