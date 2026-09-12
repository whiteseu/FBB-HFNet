# FBB-HFNet

Project code for FBB-HFNet, a foreground-background-boundary aware hierarchical
fusion network for polyp segmentation.

| file | module |
|---|---|
| `gcaeb.py` | GCAEB, grouped channel attention enhancement block |
| `fbbab.py` | FBBAB, foreground-background-boundary attention block |
| `fab.py` | FAB, feature aggregation block |

## GCAEB

Sits on the encoder side and enhances the stage-2, stage-3 and stage-4 features
of the Res2Net-50 backbone, unifying them to C = 256 channels.

It combines one direct branch with three parallel dual-branch sub-space
attention blocks. Each sub-space block splits its input in half along the
channel dimension, passes one half through a 1x1 convolution and ReLU, then
cross-concatenates the two halves into the two combinations P and Q, each
calibrated by its own channel attention before being merged. The four branch
outputs are concatenated, fused by a 3x3 convolution, and added to a residual
shortcut.

This lets the two channel subspaces inform each other before either is
calibrated, rather than re-weighting all channels uniformly.

## FBBAB

Sits on the decoder side and is applied once at each of levels 4, 3 and 2,
refining the prediction bottom-up.

It first decomposes the previous-stage prediction into three confidence maps for
foreground, background and boundary confusion; the three are non-negative and
sum to one at every pixel. The estimator takes a confidence-weighted average to
obtain one prototype vector per region. Cross-attention then uses the feature map
as queries and the three regional prototypes as keys and values, aggregating
context back to every pixel, which is concatenated with the original feature and
fused.

This treats the high-uncertainty region near the contour as an independent
semantic region instead of only separating foreground from background.

## FAB

Sits between the encoder and the decoder and starts the decoding path.

It aligns the three GCAEB outputs to H/8 resolution and concatenates them,
compresses and fuses them with a 3x3 convolution followed by a grouped
convolution, then rescales the result channel-wise with a gate produced by
global average pooling, giving the initial decoded feature.

This aggregates multi-level context without pixel-wise global self-attention.

## Weights

A trained checkpoint is available under
[Releases](https://github.com/whiteseu/FBB-HFNet/releases).

## License

MIT
