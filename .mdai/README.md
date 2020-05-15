# MDAI Model Deployment

This folder enables the model to be integrated on the MD.AI platform.

## Requirements

The checkpoints and weights (`penet_best.pth.tar`) for PENet need to be downloaded in the root directory of the repo.

## Output

The model is series-scoped (an entire CT series is sent to the model server during a request) and produces a single global series-scoped output. A wrapper for gradcam has also been implemented that optionally produces a gradcam result for MD.AI platform if enabled.

## Arguments

The model accepts optional arguments given below

```json
{
    "probability_threshold": "0.5",  // threshold for creating output
    "gradcam": "0"                   // "0" (off) or "1" (generates gif over best sliding window)
    "input_slice_number": "24"       // number of input slices per sliding window
}
```
 
