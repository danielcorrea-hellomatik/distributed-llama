#!/usr/bin/env python3
"""Generate the paper figures from data/bench/ and the documented stage values.

Run from anywhere:  python3 paper/figures/make_figures.py
Requires: matplotlib, numpy. Outputs <name>.pdf (for LaTeX) and <name>.png (for web).
"""
import csv, statistics, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE  = os.path.dirname(os.path.abspath(__file__))   # paper/figures
REPO  = os.path.dirname(os.path.dirname(HERE))        # repo root
OUT   = HERE
BENCH = os.path.join(REPO, "data", "bench")

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight",
})
INK="#1b2a4a"; ACC="#2f6fed"; GOOD="#1f9d57"; WARN="#d98a00"; GREY="#9aa3b2"

def load(*names):
    v=[]
    for n in names:
        with open(os.path.join(BENCH, n)) as f:
            for r in csv.DictReader(f):
                v.append(float(r["tok_per_s"]))
    return v

def save(fig, name):
    for ext in ("pdf","png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig); print(f"  wrote {name}.pdf/.png")

# ---- Fig 1: optimisation trajectory ----
labels=["Llama 3.1 8B\n(dense)","Qwen3-30B\nMoE switch","max-seq-len\n+ swap clean",
        "Stage 8\npatches+flags","Stage 9\nTIER-0 sysctls","Stage 10\nSILU·MUL fuse",
        "Stage 13\ntrue 1-pass","Stage 14-15\nchunk+NIC"]
vals=[5.70,11.40,12.71,13.720,14.011,14.081,14.270,14.449]
fig,ax=plt.subplots(figsize=(9,4.6))
colors=[GREY,ACC,ACC,ACC,ACC,ACC,ACC,GOOD]
ax.bar(range(len(vals)),vals,color=colors,width=0.72,zorder=3)
ax.axhline(13.04,ls="--",lw=1.3,color=WARN,zorder=2)
ax.text(len(vals)-0.5,13.04+0.12,"public ceiling 13.04 tok/s (b4rtaz #255)",
        color=WARN,ha="right",va="bottom",fontsize=9)
for i,v in enumerate(vals):
    ax.text(i,v+0.12,f"{v:.2f}",ha="center",va="bottom",fontsize=8.5,color=INK)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,fontsize=8)
ax.set_ylabel("Sustained throughput (tok/s)"); ax.set_ylim(0,15.6)
ax.set_title("Optimisation trajectory: Qwen3-30B-A3B on 4× Raspberry Pi 5",color=INK)
ax.grid(axis="y",ls=":",alpha=0.5,zorder=0)
save(fig,"fig_trajectory")

# ---- Fig 2: Stage 16 WFE vs yield A/B ----
y=load("bench_yield2.csv","bench_yield3.csv"); w=load("bench_wfe.csv","bench_wfe2.csv")
def st(a): return statistics.mean(a),1.96*statistics.stdev(a)/math.sqrt(len(a))
ym,yc=st(y); wm,wc=st(w)
fig,ax=plt.subplots(figsize=(6.4,4.6))
data=[y,w]; pos=[1,2]
bp=ax.boxplot(data,positions=pos,widths=0.5,patch_artist=True,showfliers=False,zorder=2,
              medianprops=dict(color=INK,lw=1.4))
for patch,c in zip(bp["boxes"],[GREY,GOOD]): patch.set(facecolor=c,alpha=0.35,edgecolor=c)
rng=np.random.default_rng(42)
for xp,arr,c in zip(pos,data,[GREY,GOOD]):
    ax.scatter(xp+rng.uniform(-0.13,0.13,len(arr)),arr,s=16,color=c,alpha=0.8,zorder=3,edgecolor="white",linewidth=0.4)
ax.errorbar(pos,[ym,wm],yerr=[yc,wc],fmt="D",color=INK,ms=6,capsize=4,zorder=4,label="mean ± 95% CI")
ax.set_xticks(pos); ax.set_xticklabels([f"yield spin\n(n={len(y)})",f"WFE/SEV\n(n={len(w)})"])
ax.set_ylabel("Throughput (tok/s)")
ax.set_title("Stage 16 A/B: WFE/SEV barrier vs yield-spin\n(bit-exact, same-session cold paired)",color=INK)
ax.annotate("+0.48%  (Welch t=2.45, p=0.014)",xy=(1.5,max(wm,ym)+0.18),ha="center",color=GOOD,fontsize=10)
ax.text(1,ym-0.02,f"{ym:.3f}",ha="center",va="top",fontsize=9,color=INK)
ax.text(2,wm+0.02,f"{wm:.3f}",ha="center",va="bottom",fontsize=9,color=GOOD)
ax.grid(axis="y",ls=":",alpha=0.5); ax.legend(frameon=False,loc="lower right",fontsize=9)
save(fig,"fig_stage16_ab")

# ---- Fig 3: wall-time bottleneck breakdown ----
cats=["Matmul Q40\n(MoE + F32)","Sync barrier\n(busy-spin)","Syscalls\n(send/recv)","Orchestration\n& other"]
pct=[56,17,4.5,22.5]; cols=[ACC,WARN,GREY,"#c7cfdb"]
fig,ax=plt.subplots(figsize=(7.2,3.2))
left=0
for c,p,col in zip(cats,pct,cols):
    ax.barh(0,p,left=left,color=col,edgecolor="white",zorder=3)
    ax.text(left+p/2,0,f"{c}\n{p}%",ha="center",va="center",fontsize=8.5,
            color="white" if col in (ACC,WARN) else INK)
    left+=p
ax.set_xlim(0,100); ax.set_ylim(-0.6,0.6); ax.set_yticks([])
ax.set_xlabel("Share of per-token wall-clock (%)")
ax.set_title("Where the time goes: DRAM-bandwidth-bound (49% backend-stalled cycles)",color=INK,fontsize=11)
ax.spines["left"].set_visible(False)
save(fig,"fig_bottleneck")

# ---- Fig 4: same-class comparison ----
fig,ax=plt.subplots(figsize=(5.6,4.4))
names=["b4rtaz #255\n(public ceiling)","This work\n(Stage 14-15)"]; v=[13.04,14.449]
bars=ax.bar(names,v,color=[GREY,GOOD],width=0.55,zorder=3)
for b,val in zip(bars,v): ax.text(b.get_x()+b.get_width()/2,val+0.1,f"{val:.2f}",ha="center",fontsize=11,color=INK)
ax.annotate("+10.81%",xy=(1,14.449),xytext=(0.5,15.0),ha="center",color=GOOD,fontsize=12,
            arrowprops=dict(arrowstyle="->",color=GOOD))
ax.set_ylabel("Sustained throughput (tok/s)"); ax.set_ylim(0,16)
ax.set_title("Same model & hardware class\nQwen3-30B-A3B Q40, 4× Raspberry Pi 5",color=INK,fontsize=11)
ax.grid(axis="y",ls=":",alpha=0.5,zorder=0)
save(fig,"fig_comparison")

print("DONE. yield mean=%.3f±%.3f  wfe mean=%.3f±%.3f"%(ym,yc,wm,wc))
