# mlc

ML training monitor. Wraps your training script, streams output, and gives some charts. Quick and easy. No Wandb, no changes to your code, just reading from terminal output.

## Quick Start

Install first:
```bash
curl -fsSL https://raw.githubusercontent.com/Malav-P/mlobs/main/install.sh | sh
```

Then run your training script:
```bash
mlc python train.py --lr 1e-3
# OR
mlc python main.py
# OR
mlc <whatever your command is>
```


![](./assets/run.png)