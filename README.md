# mlc

ML training monitor. Wraps your training script, streams output, and surfaces alerts when something goes wrong.

## Quick Start

```bash
mlc run python -c "
import time, math, random
for i in range(1, 120):
    loss = 2.0 * math.exp(-0.15 * min(i, 30)) + 0.05 * random.random()
    print(f'Epoch {i}/60: loss={loss:.4f}, val_loss={loss + 0.05:.4f}')
    time.sleep(1.0)
"
```
