# ------------------------------------------------------------
# evaluate_all_checkpoints.py
# ------------------------------------------------------------
from pathlib import Path
import numpy as np, pandas as pd, torch, nltk, re, matplotlib.pyplot as plt
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from evaluate import load as load_metric
from torch.utils.data import Dataset

# ─── CONFIG ──────────────────────────────────────────────────
BASE_DIR = Path("./chatbot_model_two_t5_base_continued")  # folder that holds checkpoint-5000,‑10000…
BATCH    = 8
MAX_IN, MAX_OUT = 128, 64
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ─── DATA PREP (as before, once) ─────────────────────────────
print("Loading MultiWOZ test subset …")
dataset = load_dataset("multi_woz_v22")
def wanted(d): return any(s in d["services"] for s in ["hotel","restaurant","taxi"])
test_rows = [d for d in dataset["test"] if wanted(d)]

nltk.download("punkt", quiet=True)
lemm = nltk.stem.WordNetLemmatizer()
def preproc(t):
    from nltk.tokenize import word_tokenize
    return " ".join([lemm.lemmatize(w) for w in word_tokenize(t.lower()) if w.isalpha()])

def prep_convo(d):
    c, ctx = [], []
    for utt, spk in zip(d["turns"]["utterance"], d["turns"]["speaker"]):
        if spk==0:
            ctx.append("Language: English | "+preproc(utt))
        else:
            if ctx:
                c.append({"context":" [SEP] ".join(ctx[-6:]), "response":utt})
    return c

test_df = pd.DataFrame([x for d in test_rows for x in prep_convo(d)])

class ConvDS(Dataset):
    def __init__(s,df,tok): s.df, s.tok = df, tok
    def __len__(s): return len(s.df)
    def __getitem__(s,i):
        ctx, rsp = s.df.iloc[i]
        enc = s.tok(ctx, max_length=MAX_IN, truncation=True,
                    padding="max_length", return_tensors="pt")
        dec = s.tok(rsp,max_length=MAX_OUT,truncation=True,
                    padding="max_length", return_tensors="pt")
        labels = dec["input_ids"].squeeze()
        labels[labels==s.tok.pad_token_id] = -100
        return {"input_ids":enc["input_ids"].squeeze(),
                "attention_mask":enc["attention_mask"].squeeze(),
                "labels":labels}

# ─── METRIC FN ───────────────────────────────────────────────
rouge_metric = load_metric("rouge")
smooth = SmoothingFunction().method4
def metrics_fn(tokenizer):
    def inner(eval_pred):
        logits, labels = eval_pred
        preds = np.where(logits<0, tokenizer.pad_token_id, logits)
        dec_p = tokenizer.batch_decode(preds, skip_special_tokens=True)
        dec_r = tokenizer.batch_decode(np.where(labels==-100, tokenizer.pad_token_id, labels),
                                       skip_special_tokens=True)
        bleu = np.mean([sentence_bleu([r.split()], p.split(), smoothing_function=smooth)
                        for p,r in zip(dec_p, dec_r)])
        rouge = rouge_metric.compute(predictions=dec_p, references=dec_r, use_stemmer=True)
        rougeL = (rouge["rougeL"]
                  if isinstance(rouge["rougeL"], float) else rouge["rougeL"].mid.fmeasure)
        return {"bleu": bleu, "rougeL": rougeL}
    return inner

# ─── LOOP OVER CHECKPOINTS ───────────────────────────────────
results = []
for ckpt_dir in sorted(BASE_DIR.glob("checkpoint-*"),
                       key=lambda p:int(re.findall(r"\d+",p.name)[0])):
    step = int(re.findall(r"\d+", ckpt_dir.name)[0])
    print(f"\n▶ Evaluating checkpoint {step} …")
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir).to(DEVICE)

    test_ds = ConvDS(test_df, tok)
    args = Seq2SeqTrainingArguments(
        output_dir="tmp_eval", per_device_eval_batch_size=BATCH,
        predict_with_generate=True, do_train=False, report_to="none"
    )
    trainer = Seq2SeqTrainer(model=model, args=args,
                             eval_dataset=test_ds, tokenizer=tok,
                             compute_metrics=metrics_fn(tok))
    res = trainer.evaluate()
    results.append({"step":step,
                    "loss":res["eval_loss"],
                    "bleu":res["eval_bleu"],
                    "rougeL":res["eval_rougeL"]})
    # free memory
    del model; torch.cuda.empty_cache()

# ─── RESULTS TABLE & PLOT ────────────────────────────────────
df = pd.DataFrame(results).sort_values("step")
print("\n===== Summary =====")
print(df.round(4))

plt.figure(figsize=(7,4))
plt.plot(df["step"], df["loss"],   marker="o", label="Loss")
plt.plot(df["step"], df["rougeL"], marker="o", label="ROUGE‑L")
plt.xlabel("Checkpoint step")
plt.title("Checkpoint performance on MultiWOZ test set")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig("checkpoint_metrics.png")
plt.show()
