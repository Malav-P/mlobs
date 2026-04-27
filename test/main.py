import numpy as np
import time


# -----------------------------
# Data
# -----------------------------
def make_data(n=5000, d=10, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))

    w_true = rng.normal(size=(d,))
    logits = X @ w_true + 0.5 * rng.normal(size=n)

    y = (logits > 0).astype(np.float32)
    return X, y


# -----------------------------
# Utils
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss_fn(y, p):
    eps = 1e-8
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def accuracy(y, p):
    return np.mean((p > 0.5) == y)


# -----------------------------
# Training
# -----------------------------
def train(X, y, lr=0.05, epochs=10, batch_size=128):
    n, d = X.shape
    w = np.zeros(d)

    steps_per_epoch = n // batch_size
    global_step = 0

    print("event=training_start", flush=True)

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        X, y = X[perm], y[perm]

        epoch_loss = 0.0

        print(f"event=epoch_start epoch={epoch}", flush=True)

        for step in range(steps_per_epoch):
            time.sleep(0.1)  # simulate some delay
            global_step += 1

            xb = X[step * batch_size:(step + 1) * batch_size]
            yb = y[step * batch_size:(step + 1) * batch_size]

            # forward
            logits = xb @ w
            preds = sigmoid(logits)

            # loss
            loss = loss_fn(yb, preds)
            epoch_loss += loss

            # gradient
            grad = xb.T @ (preds - yb) / batch_size
            w -= lr * grad

            acc = accuracy(yb, preds)

            # -----------------------------
            # STEP LOG (parseable)
            # -----------------------------
            print(
                f"type=step "
                f"epoch={epoch} step={step} global_step={global_step} "
                f"loss={loss:.6f} acc={acc:.6f}",
                flush=True,
            )

        # epoch summary
        full_preds = sigmoid(X @ w)
        full_acc = accuracy(y, full_preds)
        avg_loss = epoch_loss / steps_per_epoch

        print(
            f"type=epoch_summary "
            f"epoch={epoch} avg_loss={avg_loss:.6f} full_acc={full_acc:.6f}",
            flush=True,
        )

        print(f"event=epoch_end epoch={epoch}", flush=True)

        time.sleep(0.1)

    print("event=training_complete", flush=True)
    return w


# -----------------------------
# Main
# -----------------------------
def main():
    X, y = make_data(n=8000, d=20)

    w = train(X, y)

    print("\nFINAL_WEIGHTS_BEGIN", flush=True)
    print(" ".join([f"{x:.6f}" for x in w]), flush=True)
    print("FINAL_WEIGHTS_END", flush=True)


if __name__ == "__main__":
    main()