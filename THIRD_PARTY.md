# QTC author code

The wrapper in `cpr10/qtc.py` executes unchanged definitions from the authors'
[QTC repository](https://github.com/MLI-lab/recalibrating_conformal_prediction)
at commit `25b4b821539cce5dc6d0082da4f7db41858a4f41`:

- `get_sorted_softmax`;
- `QuantileConfidenceThreshold`;
- `ConformalModel` and `ThresholdedConformalModel`.

Run `python fetch_qtc.py` to obtain the three required source files. SHA-256
checks authenticate them before execution. The downloaded files are excluded
from this repository. The pinned upstream README states that all files are
provided under the Apache License, Version 2.0. That license applies to the
downloaded author code; the wrapper does not replace its attribution or terms.

Our wrapper reconstructs float32 logits by taking the logarithm of cached
probabilities. It does not recover the original classifier logits and does not
apply CP-R10's probability floor. It uses the paper's `min(gamma_T, gamma_S)`
combination, upstream linear quantile interpolation, and upstream thresholded
prediction sets. A quantile level exceeding one produces all classes in the
main comparison. The separately reported cap sensitivity uses the largest
source score at that boundary; all other calculations remain unchanged.
