#!/usr/bin/env python
import os, json, glob, csv, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/home/vipuser/train_logs")
    ap.add_argument("--models", nargs="*", default=[
        "s1_ps1p5_h512_e5_cycle",
        "s1_ps1p5_h512_e5_cycle_mix",
        "s1_ps1p5_h512_e5_lr1e4",
        "s1_ps1p5_h512_e5_lr2e4_nockpt",
    ])
    args = ap.parse_args()

    rows=[]
    for m in args.models:
        d=os.path.join(args.logs, m)
        cands=glob.glob(os.path.join(d,"metrics*.json"))
        if not cands:
            rows.append({"model":m,"Top1_new":"","Top5_new":"","CLIP_cosine":"","MSE":"","LPIPS":"","Set":"","note":"no metrics"})
            continue
        with open(cands[0],"r") as f:
            k=json.load(f)
        rows.append({
            "model": m,
            "Top1_new": k.get("top1_new"),
            "Top5_new": k.get("top5_new"),
            "CLIP_cosine": k.get("clip_cosine"),
            "MSE": k.get("mse"),
            "LPIPS": k.get("lpips"),
            "Set": k.get("set","new_test"),
        })
    if not rows:
        print("no rows")
        return
    csv_path=os.path.join(args.logs,"metrics_summary.csv")
    md_path=os.path.join(args.logs,"metrics_summary.md")
    with open(csv_path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with open(md_path,"w") as f:
        hdr=list(rows[0].keys())
        f.write("| " + " | ".join(hdr) + " |\n")
        f.write("| " + " | ".join(["---"]*len(hdr)) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(k,"")) for k in hdr) + " |\n")
    print("wrote:", csv_path, "and", md_path)

if __name__=="__main__":
    main()
