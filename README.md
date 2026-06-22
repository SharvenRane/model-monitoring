# Model Monitoring and Drift Detection

Once a model is deployed, the question is no longer "is it accurate" but "is the
data still the data it was validated on". This watches for that. You give it a
frozen reference window, the training set or the last production window you
trusted, and it tells you when the live inputs or the model's own scores have
moved far enough that the model should be looked at again.

That is the trigger in a governance or change control story. Drift past a
threshold means revalidate before you keep trusting the predictions.

## What it checks

PSI, the Population Stability Index, for continuous distributions, with the usual
model risk thresholds where anything over 0.25 counts as a real shift. The KS
test for continuous features and scores. A chi square test for categorical
features.

## Using it

```python
from monitor import DriftMonitor

monitor = DriftMonitor(reference={"age": ref_age, "pred_score": ref_scores})
report = monitor.report(current={"age": cur_age, "pred_score": cur_scores})
print(report["drift_detected"], report["drifted_features"])
```

## Tests

```
pip install -r requirements.txt
pytest tests/ -q
```

The tests check both directions: no false alarm when the current window is drawn
from the same distribution, and a clear signal when the mean shifts or a
category gets reweighted. Nothing to download.
