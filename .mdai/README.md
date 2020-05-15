# MDAI Model Deployment

This folder enables the model to be integrated on the MD.AI platform.

## Requirements
The checkpoints and weights for PENet need to be downloaded in the root directory of the model

## Output
The model is series-scoped and produces a single global series-scoped output. A wrapper for gradcam has also been implemented that optionally produces a gradcam result for MD.AI platform if enabled

## Arguments
The model accepts optional arguments given below

```javascript
{
    probability_threshold: "str"   // default 0.5
    gradcam: "str"                 //0(off) / 1(best window). default 0
    slices_per_window: "str"       //default 24
}
```
 
