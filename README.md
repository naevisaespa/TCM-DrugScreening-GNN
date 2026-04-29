# TCMDA

This repository contains the training code for the TCMDA study on prioritizing TCM-derived small molecules for gastric cancer using heterogeneous graph learning.

## Train

```bash
python train.py \
  --graph-root ./path_to_graph_root \
  --save-dir ./path_to_output_dir \
  --device 0 \
  --seed 1105 \
  --epoch 400 \
  --patience 80
```

## Dual Test Evaluation


```bash
python evaluate.py \
  --graph-root ./path_to_graph_root \
  --checkpoint ./path_to_checkpoint \
  --save-dir ./path_to_eval_dir \
  --device 0 
```
