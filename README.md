# spec2wav

Experimenting with a neural vocoder derived from other researches
The skeleton is the same as SampleRNN, with modifications in architecture to suit my use case.

Will modify the architecture to make it extremely flexible for multiple purposes. Speech, Music, Any Other Audios, etc.

## TODO
### Training Block
  - [x] Saving Model
  - [x] Loading Model
  - [x] Evaluation?
  - [x] Train/Val loss visualisation
  - [x] Generation audio every epoch
### Model
  - [x] Model Input Output Tuning
  - [x] Weight/Hidden State Initialization
  - [ ] Weight Normalization
  - [x] Generator File/ Module
- [ ] HyperParameter Tuning
- [x] CUDA
- [ ] SalesForce QRNN
- [ ] Handle Variable Length Input
- [x] Gaussian Noise?
