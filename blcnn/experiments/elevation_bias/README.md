Humans have a bias to estimate higher-frequency stimuli higher spatially. A lot of evidence points to this being a result of higher-level cognitive functions that result from our experiences with higher-frequency sounds often being further upward in nature. This effect is mitigated in music, where our understanding that instruments play from a stable sound source provides context that helps us localize elevation more correctly. We want to check if the DNN model exhibits a similar bias by experimenting on it similarly to what was done in [this paper](https://zenodo.org/records/17496602). However the current model implementation requires some concessions:

- The model is only able to estimate in a range from 0 to 60° instead of -25 to +25°.
- Current estimations of the model in elevation are much more error-prone than those of humans. So we may need to use chords for more spectral information rather than single notes.

# Execution

To train the regression model:
```
python blcnn/regression/train.py \
    --pretrained-model models/keras/net3.keras \
    --train-data data/cochleagrams/natural_selection_slab_kemar/train_cochleagrams.tfrecord \
    --output-dir models/regression \
    --epochs 50 \
    --batch-size 8 \
    --max-train-batches 500 \
    --max-val-batches 50 \
    --shuffle-buffer-size 500 \
    --validation-split 0.1 \
    --num-unfrozen-layers 3
```

To run the regression model:
```
python blcnn/regression/run.py \
    --model  models/regression/2026-03-01_14-52-51/final_model.keras \
    --data   data/cochleagrams/viola_slab_kemar/test_cochleagrams.tfrecord \
    --output data/output/regression_predictions_viola_test.csv \
--batch-size 8
```
