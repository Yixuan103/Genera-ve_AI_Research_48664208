# Genera-ve_AI_Research_48664208
torch>=2.0
transformers>=4.39
datasets>=2.19
accelerate>=0.27
tqdm
from pathlib import Path
from datasets import load_dataset, Dataset

def load_local_txt(path: str, split_ratio=0.9):
    """把本地 txt 拆成 train/valid 两份 HuggingFace Dataset"""
    txt = Path(path).read_text(encoding="utf‑8").splitlines()
    n_train = int(len(txt) * split_ratio)
    train_ds = Dataset.from_dict({"text": txt[:n_train]})
    val_ds   = Dataset.from_dict({"text": txt[n_train:]})
    return {"train": train_ds, "validation": val_ds}
import argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="../results/gpt2-finetuned")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)

    inputs = tok(args.prompt, return_tensors="pt").to(device)
    out_ids = model.generate(**inputs,
                             max_new_tokens=args.max_new_tokens,
                             do_sample=True, top_p=0.95, temperature=0.8)
    print(tok.decode(out_ids[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
