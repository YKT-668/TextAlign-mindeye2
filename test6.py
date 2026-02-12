import json, re
p="src/final_evaluations.ipynb"
nb=json.load(open(p,"r",encoding="utf-8"))

MODEL="s1_textalign_stage1_FINAL_BEST_32"
SUBJ=1
DATA="/mnt/work/repos/TextAlign-mindeye2"
CACHE="/mnt/work/repos/TextAlign-mindeye2"

def rep(s):
    s=re.sub(r'model_name\s*=\s*".*?"', f'model_name="{MODEL}"', s)
    s=re.sub(r'subj\s*=\s*\d+', f'subj={SUBJ}', s)
    s=re.sub(r'data_path\s*=\s*".*?"', f'data_path="{DATA}"', s)
    s=re.sub(r'cache_dir\s*=\s*".*?"', f'cache_dir="{CACHE}"', s)
    return s

for c in nb["cells"]:
    if c.get("cell_type")=="code":
        s="".join(c.get("source",[]))
        c["source"]=rep(s).splitlines(True)

out=f"src/final_evaluations__patched_{MODEL}.ipynb"
json.dump(nb, open(out,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote:", out)