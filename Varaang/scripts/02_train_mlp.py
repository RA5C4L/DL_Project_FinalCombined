"""
Step 2 - MLP + TF-IDF baseline: train, validate, test once, save results.
Owner: Varaang Tiwari (230953378)

Config (matches Interim Report Table, Section 3):
  TF-IDF unigrams+bigrams, 30,000 max features, min_df=2, max_df=0.95, sublinear_tf=True
  MLP: 30000 -> 256 (ReLU) -> dropout 0.5 -> 1 (sigmoid via BCEWithLogits)
  Adam (lr 1e-3), batch 128, early stopping on validation loss (patience 3), seed 42
"""
import os, sys, time, json, random
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metrics_utils import compute_metrics, save_json
from importlib import import_module
prep = import_module("01_prepare_data")

SEED = 42
MAX_FEATURES, MIN_DF, MAX_DF = 30000, 2, 0.95
HIDDEN, DROPOUT, LR, BATCH = 256, 0.5, 1e-3, 128
MAX_EPOCHS, PATIENCE = 30, 3
RES = os.path.join(HERE, "..", "results")
SPLIT = os.path.join(HERE, "..", "splits", "imdb_split.json")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, HIDDEN), nn.ReLU(),
                                 nn.Dropout(DROPOUT), nn.Linear(HIDDEN, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def to_dense(X_sparse, rows, device):
    return torch.from_numpy(X_sparse[rows].toarray().astype(np.float32)).to(device)


@torch.no_grad()
def predict(model, X, device, bs=512):
    model.eval()
    logits = []
    for i in range(0, X.shape[0], bs):
        rows = np.arange(i, min(i + bs, X.shape[0]))
        logits.append(model(to_dense(X, rows, device)).cpu())
    logits = torch.cat(logits)
    return logits, (logits > 0).long().numpy()


def main():
    set_seed(SEED)
    os.makedirs(RES, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ---- data + shared split -------------------------------------------------
    tr_x, tr_y, te_x, te_y = prep.load_imdb()
    split = json.load(open(SPLIT))
    tr_i, va_i = np.array(split["train_idx"]), np.array(split["val_idx"])
    clean = prep.clean_text
    X_train_txt = [clean(tr_x[i]) for i in tr_i]; y_train = tr_y[tr_i]
    X_val_txt   = [clean(tr_x[i]) for i in va_i]; y_val   = tr_y[va_i]
    X_test_txt  = [clean(t) for t in te_x];       y_test  = te_y
    print(f"train={len(X_train_txt)} val={len(X_val_txt)} test={len(X_test_txt)}")

    # ---- TF-IDF: fit on TRAIN ONLY ------------------------------------------
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=MAX_FEATURES, min_df=MIN_DF,
                          max_df=MAX_DF, sublinear_tf=True)
    Xtr = vec.fit_transform(X_train_txt).tocsr()
    Xva = vec.transform(X_val_txt).tocsr()
    Xte = vec.transform(X_test_txt).tocsr()
    print("TF-IDF feature dim:", Xtr.shape[1])
    joblib.dump(vec, os.path.join(RES, "tfidf_vectorizer.joblib"))   # needed for Phase-3 robustness tests

    # ---- train with early stopping on validation loss -----------------------
    model = MLP(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    ytr_t = torch.tensor(y_train, dtype=torch.float32)
    yva_t = torch.tensor(y_val, dtype=torch.float32)
    best, bad, best_state, history = float("inf"), 0, None, []
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = np.random.permutation(Xtr.shape[0]); run = 0.0
        for i in range(0, len(perm), BATCH):
            rows = perm[i:i + BATCH]
            xb, yb = to_dense(Xtr, rows, device), ytr_t[rows].to(device)
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
            run += loss.item() * len(rows)
        tr_loss = run / len(perm)
        va_logits, va_pred = predict(model, Xva, device)
        va_loss = loss_fn(va_logits, yva_t).item()
        va_acc = float((va_pred == y_val).mean())
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss, "val_acc": va_acc})
        print(f"epoch {ep:02d}  train_loss {tr_loss:.4f}  val_loss {va_loss:.4f}  val_acc {va_acc:.4f}")
        if va_loss < best - 1e-4:
            best, bad = va_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping."); break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(RES, "mlp_best.pt"))

    # ---- evaluate ONCE on official test set ---------------------------------
    _, te_pred = predict(model, Xte, device)
    _, va_pred = predict(model, Xva, device)
    results = {
        "model": "MLP + TF-IDF", "owner": "Varaang Tiwari (230953378)", "seed": SEED,
        "config": {"tfidf": {"ngram_range": [1, 2], "max_features": MAX_FEATURES, "min_df": MIN_DF,
                             "max_df": MAX_DF, "sublinear_tf": True, "actual_dim": int(Xtr.shape[1])},
                   "hidden_units": HIDDEN, "dropout": DROPOUT, "optimizer": "Adam", "lr": LR,
                   "batch_size": BATCH, "patience": PATIENCE},
        "epochs_run": len(history), "best_val_loss": best, "train_time_sec": round(train_time, 1),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "val": compute_metrics(y_val, va_pred),
        "test": compute_metrics(y_test, te_pred),
        "history": history,
    }
    save_json(results, os.path.join(RES, "mlp_results.json"))
    np.save(os.path.join(RES, "mlp_test_predictions.npy"), te_pred)   # for Phase-3 error analysis

    # ---- figures -------------------------------------------------------------
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train"); ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("MLP loss")
    ax[1].plot(ep, [h["val_acc"] for h in history]); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val accuracy"); ax[1].set_title("MLP validation accuracy")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "mlp_training_curves.png"), dpi=200); plt.close()

    cm = np.array(results["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(3.6, 3.2)); ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm): ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["neg", "pos"]); ax.set_yticklabels(["neg", "pos"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("MLP test confusion matrix")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "mlp_confusion_matrix.png"), dpi=200); plt.close()

    t = results["test"]
    print("\n===== TEST RESULTS (MLP + TF-IDF) =====")
    print(f"Accuracy  {t['accuracy']*100:.2f}%   Macro-P {t['precision_macro']*100:.2f}%   "
          f"Macro-R {t['recall_macro']*100:.2f}%   Macro-F1 {t['f1_macro']*100:.2f}%")
    print("Confusion matrix [[TN FP],[FN TP]]:", t["confusion_matrix"])
    write_snippet(results)


def write_snippet(r):
    t, c = r["test"], r["config"]
    txt = f"""# Varaang - text to paste into the Interim Report (auto-generated from mlp_results.json)

## Section 3 - Models table, MLP row
Owner: Varaang Tiwari | Status: Implemented; trained and evaluated on the common test set
Preliminary result: Accuracy {t['accuracy']*100:.1f}%; macro F1 {t['f1_macro']*100:.1f}%
Configuration: {c['tfidf']['actual_dim']:,}-dim TF-IDF (unigram+bigram) input; one {c['hidden_units']}-unit ReLU layer; dropout {c['dropout']}; Adam (lr {c['lr']}); early stopping (stopped after {r['epochs_run']} epochs).

## Section 3 - extra paragraph (optional, under the table)
The MLP baseline was trained on the 20,000-review training split, tuned by early stopping on the 5,000-review validation split, and evaluated once on the 25,000-review official test set. Test accuracy was {t['accuracy']*100:.2f}% (macro precision {t['precision_macro']*100:.2f}%, macro recall {t['recall_macro']*100:.2f}%, macro F1 {t['f1_macro']*100:.2f}%). The confusion matrix was [[{t['confusion_matrix'][0][0]}, {t['confusion_matrix'][0][1]}], [{t['confusion_matrix'][1][0]}, {t['confusion_matrix'][1][1]}]] (rows = true negative/positive). The model has {r['n_params']:,} trainable parameters and trained in {r['train_time_sec']} seconds. Results are preliminary and unoptimised; no hyper-parameter search was performed.

## Section 4 - Contribution log row
Varaang Tiwari | 230953378 | IMDB acquisition; shared split file; TF-IDF pipeline; MLP baseline; shared metric script. | Commit history on GitHub (add repo link + commit range) | [signature]

## Section 5 - Risk (MLP-specific, optional)
TF-IDF is vocabulary-bound, so words unseen at training time (typos, rare synonyms) are silently dropped; this is expected to make the MLP sensitive to character noise and cross-domain vocabulary, which Phase 3 will quantify.
"""
    with open(os.path.join(RES, "report_snippet.md"), "w") as f:
        f.write(txt)
    print("Wrote results/report_snippet.md")


if __name__ == "__main__":
    main()
