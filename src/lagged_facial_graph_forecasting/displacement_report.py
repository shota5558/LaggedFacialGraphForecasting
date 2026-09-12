"""PNG figures and Japanese Markdown draft from serialized new-metric tables."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import numpy as np
import pandas as pd

from .displacement_analysis import BANDS, CONDITIONS, LABELS, NOTICE, POINTS, REGIONS, write_csv

NAMES = dict(zip(REGIONS, LABELS)) | {"whole_face": "全顔", "selected_targets": "評価可能な選択target"}
COND_NAMES = {"speaking": "本人のみ発話", "non_speaking": "本人非発話"}
MODEL_NAMES = {"persistence": "Persistence", "self": "Self", "full": "Full", "pcmci": "PCMCI-block"}
SCALE = 1000
UNIT_LABEL = "眼角中点間距離比 ×10⁻³"
INTERVAL = "帯と区間は固定OOF値のgroup構成に対する名目95%区間。学習全体の不確かさ・同時区間ではない。"


def _markdown_path(root: Path, path: Path) -> str:
    """Return a report-local POSIX path for Markdown links and images."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def render_report(root: Path, config: dict) -> Path:
    plt.rcParams.update({"font.family": ["Meiryo", "DejaVu Sans"], "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "savefig.facecolor": "white"})
    tables, figures = root/"tables", root/"figures"
    figures.mkdir(exist_ok=True)
    read = lambda name: pd.read_csv(tables/(name+".csv"))
    band = read("N_ST2_band_summary")
    band_subject = read("N_ST2_band_subject")
    cell = read("N_ST2_cell_summary")
    centered = read("N_ST2_centered_summary")
    centered_subject = read("N_ST2_centered_subject")
    errors = read("N_T2_model_errors")
    effects = read("N_T2_paired_effects")
    model_subject = read("N_ST2_model_subject")
    effect_subject = read("N_ST2_paired_subject")
    enrich = read("N_T3_enrichment")
    enrich_subject = read("N_ST2_enrichment_subject")
    repeats = pd.read_csv(tables/"N_ST2_enrichment_repeats.csv.gz")
    selection = pd.read_csv(root/"input/selection.csv")
    quality = pd.read_csv(root/"input/quality.csv")
    subjects = pd.read_csv(root/"input/subjects.csv")
    captions = {}
    supplement_images = []

    def save(fig, name, title, sources, caption=""):
        fig.suptitle(title+"  模擬データ", fontsize=13, y=0.995)
        fig.text(0.5, 0.008, NOTICE, ha="center", fontsize=8, color="#9a3412")
        fig.tight_layout(rect=(0, 0.03, 1, 0.975))
        path = figures/(name+".png")
        fig.savefig(path, dpi=170)
        plt.close(fig)
        captions[name] = dict(title=title, caption=caption, source_tables=sources,
                              unit=UNIT_LABEL, uncertainty=INTERVAL, synthetic_notice=NOTICE)
        return path

    def heat(ax, part, value, vmax, *, percent=False, label=""):
        matrix = part.pivot(index="target", columns="source", values=value).reindex(index=REGIONS, columns=REGIONS).to_numpy(float)
        cmap = plt.get_cmap("viridis" if percent else "RdBu_r").copy()
        cmap.set_bad("#dddddd")
        image = ax.imshow(matrix if percent else matrix*SCALE, cmap=cmap, vmin=0 if percent else -vmax, vmax=vmax)
        ax.set_xticks(range(8), LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(8), LABELS, fontsize=8)
        ax.set_title(label, fontsize=10)
        for i in range(8):
            ax.add_patch(Rectangle((i-.5, i-.5), 1, 1, facecolor="#eeeeee", edgecolor="#aaaaaa", hatch="///", linewidth=0))
        return image

    vmax = float(band["median"].abs().max()*SCALE)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for c, condition in enumerate(CONDITIONS):
        for b, name in enumerate(BANDS):
            part = band[(band.condition == condition) & (band.band == name)]
            image = heat(axes[c, b], part, "median", vmax, label=COND_NAMES[condition]+"  "+name)
            for r in part.itertuples():
                if np.isfinite(r.median):
                    axes[c,b].text(REGIONS.index(r.source), REGIONS.index(r.target), f"{r.median*SCALE:.2f}", ha="center", va="center", fontsize=6,
                                   color="white" if abs(r.median*SCALE)>vmax*.65 else "black")
            fig.colorbar(image, ax=axes[c,b], shrink=.75, label="G  "+UNIT_LABEL)
    save(fig, "N_F1_landscape", "N-F1 全候補の帯域G  横source 縦target", ["N_ST2_band_summary.csv"],
         "被験者内の全lag平均後に被験者間中央値。斜線は自己対、灰色は評価不能。")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for c, condition in enumerate(CONDITIONS):
        for b, name in enumerate(BANDS):
            part = band[(band.condition == condition) & (band.band == name)]
            image = heat(axes[c,b], part, "fraction_positive", 1, percent=True, label=COND_NAMES[condition]+"  "+name)
            for r in part.itertuples():
                if r.n_subjects:
                    axes[c,b].text(REGIONS.index(r.source), REGIONS.index(r.target), f"{r.n_positive}/{r.n_subjects}", ha="center", va="center", fontsize=6,
                                   color="white" if r.fraction_positive<.5 else "black")
            fig.colorbar(image, ax=axes[c,b], shrink=.75, label="G > 0 の被験者割合")
    save(fig, "N_F2_commonality", "N-F2 改善の符号と分母  横source 縦target", ["N_ST2_band_summary.csv"],
         "セル内は正の人数/評価人数。0は正に含めない。真の効果がある人口の割合ではない。")

    pair_order = [(t,s) for t in REGIONS for s in REGIONS if s != t]
    subject_order = subjects.sort_values(["outer_fold", "group_id", "subject_id"]).subject_id.tolist()
    vmax_subject = band_subject.gain.abs().max()*SCALE
    for condition in CONDITIONS:
        fig, axes = plt.subplots(1, 3, figsize=(18, 14))
        for ax, name in zip(axes, BANDS):
            part = band_subject[(band_subject.condition==condition) & (band_subject.band==name)]
            matrix = part.pivot(index=["target","source"], columns="subject_id", values="gain").reindex(index=pd.MultiIndex.from_tuples(pair_order), columns=subject_order)
            cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#aaaaaa")
            image = ax.imshow(matrix.to_numpy()*SCALE, aspect="auto", vmin=-vmax_subject, vmax=vmax_subject, cmap=cmap)
            ax.set_yticks(range(56), [f"{NAMES[s]}→{NAMES[t]}" for t,s in pair_order], fontsize=7)
            ax.set_xticks(range(30), [s[-2:] for s in subject_order], rotation=90, fontsize=7)
            for edge in (5.5,11.5,17.5,23.5): ax.axvline(edge, color="black", lw=1)
            ax.set_title(name); ax.set_xlabel("MOCK subject ID  縦線はfold境界")
            fig.colorbar(image, ax=ax, shrink=.55, label="G  "+UNIT_LABEL)
        name = "N_F2_subject_"+condition
        save(fig, name, "N-F2 全被験者の帯域G  "+COND_NAMES[condition], ["N_ST2_band_subject.csv"])
        supplement_images.append(name)

    absmax = np.nanmax(np.abs(cell[["ci_low","ci_high"]].to_numpy()))*SCALE
    for condition in CONDITIONS:
        fig, axes = plt.subplots(8, 8, figsize=(17, 14), sharex=True, sharey=True)
        for j, target in enumerate(REGIONS):
            for i, source in enumerate(REGIONS):
                ax = axes[j,i]
                if i == j:
                    ax.set_facecolor("#eeeeee"); ax.text(.5,.5,"自己対\n候補外",transform=ax.transAxes,ha="center",va="center",fontsize=8)
                else:
                    part = cell[(cell.condition==condition)&(cell.target==target)&(cell.source==source)].sort_values("lag")
                    x = part.lag*1000/config["fps"]
                    ax.axhline(0,color="#aaaaaa",lw=.6)
                    ax.fill_between(x,part.ci_low*SCALE,part.ci_high*SCALE,color="#c7dfec",alpha=.8)
                    ax.plot(x,part["median"]*SCALE,color="#155e75",lw=1.4)
                    ax.set_ylim(-absmax*1.05,absmax*1.05)
                if j==0: ax.set_title(NAMES[source],fontsize=9)
                if i==0: ax.set_ylabel(NAMES[target]+"\nG ×10⁻³",fontsize=8)
                if j==7: ax.set_xlabel("lag ms",fontsize=8)
                ax.set_xticks([100,250,500]); ax.tick_params(labelsize=7)
        save(fig,"N_F3_absolute_"+condition,"N-F3A 全56部位対の絶対lag  "+COND_NAMES[condition], ["N_ST2_cell_summary.csv"],
             "横source・縦target。全候補を固定順に表示。実上限500 ms、h=1。帯は点ごとの名目区間。")

    fig, axes = plt.subplots(1, 2, figsize=(11,4.5), sharey=True)
    for ax, condition in zip(axes, CONDITIONS):
        raw = centered_subject[centered_subject.condition==condition]
        for _, line in raw.groupby("subject_id"):
            line=line.sort_values("delta"); ax.plot(line.delta*1000/30,line.response*SCALE,color="#94a3b8",alpha=.25,lw=.8)
        part=centered[centered.condition==condition].sort_values("delta")
        ax.fill_between(part.delta*1000/30,part.ci_low*SCALE,part.ci_high*SCALE,color="#c7dfec")
        ax.plot(part.delta*1000/30,part["median"]*SCALE,"o-",color="#155e75",lw=2)
        ax.axhline(0,color="#888888",lw=.8); ax.set_title(COND_NAMES[condition]); ax.set_xlabel("選択lagからのずれ Δ ms")
        ax.set_ylabel("R  "+UNIT_LABEL)
    save(fig,"N_F3_centered","N-F3B 選択lag中心の応答",["N_ST2_centered_subject.csv","N_ST2_centered_summary.csv"],
         "Δ=0は定義上0。正はずらした時の悪化。全Δで同じedge・部位対・被験者。")

    contrasts = ["persistence-self", "self-full", "self-pcmci", "full-pcmci"]
    fig, axes = plt.subplots(2, 2, figsize=(12,8))
    for c, condition in enumerate(CONDITIONS):
        ax=axes[0,c]
        for i, model in enumerate(MODEL_NAMES):
            points=model_subject[(model_subject.condition==condition)&(model_subject.target=="whole_face")&(model_subject.model==model)]
            vals=errors[(errors.condition==condition)&(errors.target=="whole_face")&(errors.model==model)].iloc[0]
            ax.scatter(np.full(len(points),i)+np.linspace(-.16,.16,len(points)),points.error*SCALE,color="#64748b",s=13,alpha=.5)
            ax.plot([i,i],[vals.ci_low*SCALE,vals.ci_high*SCALE],color="#155e75",lw=3); ax.scatter(i,vals["median"]*SCALE,color="#155e75",s=45)
        ax.set_xticks(range(4),list(MODEL_NAMES.values()),fontsize=8); ax.set_ylabel("E  "+UNIT_LABEL); ax.set_title(COND_NAMES[condition]+"  全顔")
        ax=axes[1,c]
        for i, contrast in enumerate(contrasts):
            points=effect_subject[(effect_subject.condition==condition)&(effect_subject.target=="whole_face")&(effect_subject.contrast==contrast)]
            vals=effects[(effects.condition==condition)&(effects.target=="whole_face")&(effects.contrast==contrast)].iloc[0]
            ax.scatter(points.difference*SCALE,np.full(len(points),i)+np.linspace(-.15,.15,len(points)),color="#64748b",s=13,alpha=.5)
            ax.plot([vals.ci_low*SCALE,vals.ci_high*SCALE],[i,i],color="#155e75",lw=3); ax.scatter(vals["median"]*SCALE,i,color="#155e75",s=45)
        ax.axvline(0,color="#888888",lw=.8); ax.set_yticks(range(4),contrasts); ax.set_xlabel("対応差  "+UNIT_LABEL); ax.invert_yaxis()
    save(fig,"N_F4_models","N-F4 4条件の誤差と対応差",["N_T2_model_errors.csv","N_T2_paired_effects.csv","N_ST2_model_subject.csv","N_ST2_paired_subject.csv"],
         "全顔は8領域が揃う被験者のみ。点は被験者、太線は名目区間。差は被験者内で計算。")
    for condition in CONDITIONS:
        fig, axes=plt.subplots(1,4,figsize=(15,5),sharey=True)
        for ax, contrast in zip(axes,contrasts):
            part=effects[(effects.condition==condition)&(effects.contrast==contrast)].set_index("target").reindex(REGIONS)
            ax.hlines(range(8),part.ci_low*SCALE,part.ci_high*SCALE,color="#155e75",lw=2)
            ax.scatter(part["median"]*SCALE,range(8),color="#155e75"); ax.axvline(0,color="#888888",lw=.7)
            ax.set_yticks(range(8),LABELS); ax.set_title(contrast); ax.set_xlabel("対応差 ×10⁻³")
        axes[0].invert_yaxis()
        name="N_F4_targets_"+condition
        save(fig,name,"N-F4 target別対応差  "+COND_NAMES[condition],["N_T2_paired_effects.csv"])
        supplement_images.append(name)

    fig, axes=plt.subplots(1,2,figsize=(12,6),sharex=True,sharey=True)
    targets=list(REGIONS)+["selected_targets"]
    for ax, condition in zip(axes,CONDITIONS):
        for j,target in enumerate(targets):
            points=enrich_subject[(enrich_subject.condition==condition)&(enrich_subject.target==target)].dropna(subset=["enrichment"])
            row=enrich[(enrich.condition==condition)&(enrich.target==target)].iloc[0]
            ax.scatter(points.enrichment*SCALE,j+np.linspace(-.14,.14,len(points)),color="#64748b",s=12,alpha=.5)
            ax.plot([row.ci_low*SCALE,row.ci_high*SCALE],[j,j],lw=2.5,color="#155e75"); ax.scatter(row["median"]*SCALE,j,color="#155e75",s=35)
        ax.axvline(0,color="#888888",lw=.8); ax.set_yticks(range(9),[NAMES[x] for x in targets]); ax.set_title(COND_NAMES[condition]); ax.set_xlabel("Enrichment Z  "+UNIT_LABEL)
    axes[0].invert_yaxis()
    save(fig,"N_F5_enrichment","N-F5 選択cellの濃縮",["N_T3_enrichment.csv","N_ST2_enrichment_subject.csv"],
         "選択cell-G平均から1,000ランダム集合平均の算術平均を引く。空選択は不能。")
    for condition in CONDITIONS:
        fig, axes=plt.subplots(2,4,figsize=(17,12),sharey=True)
        for ax,target in zip(axes.flat,REGIONS):
            for j,subject in enumerate(subject_order):
                vals=repeats[(repeats.condition==condition)&(repeats.target==target)&(repeats.subject_id==subject)]
                if len(vals):
                    y=vals.random_mean.to_numpy()*SCALE
                    ax.scatter(y,np.full(len(y),j),s=.4,color="#94a3b8",alpha=.2,rasterized=True)
                    # Missing any repeat leaves the expectation undefined.
                    if np.isfinite(y).all(): ax.plot(np.mean(y),j,"|",color="#2563eb",ms=7)
                    if np.isfinite(vals.selected_mean.iloc[0]): ax.plot(vals.selected_mean.iloc[0]*SCALE,j,"o",color="#9a3412",ms=3)
            ax.set_title(NAMES[target]); ax.set_yticks(range(30),[s[-2:] for s in subject_order],fontsize=7); ax.set_xlabel("集合平均G ×10⁻³"); ax.axvline(0,color="#cccccc",lw=.5)
        axes[0,0].invert_yaxis()
        name="N_F5_random_"+condition
        save(fig,name,"N-F5 全被験者の1,000集合  "+COND_NAMES[condition]+"  灰点=random 茶点=selected 青線=random平均",["N_ST2_enrichment_repeats.csv.gz"],
             "横は集合平均G、縦はMOCK subject ID。反復は人数ではない。失敗・空選択を0にしない。")
        supplement_images.append(name)

    fig, axes=plt.subplots(1,3,figsize=(14,5))
    ax=axes[0]; ax.add_patch(Ellipse((0,0),1.6,2.2,fill=False,edgecolor="#64748b"))
    centers=[(.45,.65),(-.45,.65),(.45,.4),(-.45,.4),(.65,-.1),(-.65,-.1),(0,-.45),(0,-.85)]
    for (x,y),label,points in zip(centers,LABELS,POINTS):
        ax.scatter(np.linspace(x-.12,x+.12,len(points)),np.full(len(points),y),s=14,color="#155e75")
        ax.text(x,y+.13,f"{label} {len(points)}点",ha="center",fontsize=8)
    ax.scatter([-.65,-.25,.25,.65],[.4]*4,marker="x",color="#9a3412",s=24)
    ax.set_xlim(-1.05,1.05);ax.set_ylim(-1.2,1.15);ax.set_aspect("equal");ax.axis("off");ax.set_title("29点の配置概念図\n×は眼角参照  実測座標ではない")
    stages=["original_seconds","after_calibration_seconds","after_qc_seconds","after_boundary_history_seconds","common_support_seconds"]
    for c,condition in enumerate(CONDITIONS):
        vals=quality[quality.condition==condition][stages].sum().to_numpy()/60
        axes[1].plot(range(5),vals,"o-",label=COND_NAMES[condition],ls="-" if c==0 else "--")
    axes[1].set_xticks(range(5),["元記録","校正後","QC後","履歴後","共通support"],rotation=35);axes[1].set_ylabel("延べ時間 分");axes[1].set_title("工程別時間  模擬値");axes[1].legend(fontsize=8)
    axes[2].scatter(quality.missing_fraction*100,quality.observed_displacement_amplitude*SCALE,c=quality.outer_fold,cmap="viridis",s=25)
    axes[2].set_xlabel("欠損率 %  模擬値");axes[2].set_ylabel("顎の観測変位振幅 ×10⁻³");axes[2].set_title("品質記述  色はfold")
    save(fig,"N_SF1_quality","N-SF1 測定定義と品質",["../input/quality.csv","../input/config.json"],
         "点配置は概念図。実topologyや追跡品質の確認ではない。QC時間は模擬値。")
    supplement_images.append("N_SF1_quality")

    history=np.load(root/"input/mock_discovery_history.npz")
    stability=[]
    for key,part in selection.groupby(["condition","outer_fold","target","source"],sort=True):
        condition,fold,target,source=key
        for band_name,lower,upper in zip(BANDS,[0,100,250],[100,250,500]):
            members=part[(part.lag*1000/30>lower)&(part.lag*1000/30<=upper)]
            arr=np.stack([history[f"{condition}_{fold}_{target}_{source}_{lag}"] for lag in members.lag])
            opportunity=(arr!=-2).any(axis=0);failed=(arr==-1).any(axis=0)
            selected_any=(arr==1).any(axis=0);n=int(opportunity.sum());k=int(selected_any.sum());fail=int(failed.sum())
            stability.append(dict(condition=condition,outer_fold=fold,target=target,source=source,band=band_name,n_lags=len(members),
                                  opportunities=n,selected_count=k,failures=fail,
                                  frequency=k/n if n and not fail else np.nan,
                                  frequency_low=k/n if n else np.nan,frequency_high=(k+fail)/n if n else np.nan,
                                  primary_selected=bool(members.selected.any())))
    stability=pd.DataFrame(stability)
    write_csv(stability,tables/"N_ST2_stability_band.csv")
    for condition in CONDITIONS:
        fig,axes=plt.subplots(3,5,figsize=(17,10))
        for b,band_name in enumerate(BANDS):
            for fold in range(5):
                part=stability[(stability.condition==condition)&(stability.outer_fold==fold)&(stability.band==band_name)]
                image=heat(axes[b,fold],part,"frequency_low",1,percent=True,label=f"fold {fold}  {band_name}")
                for r in part[part.primary_selected].itertuples():
                    axes[b,fold].plot(REGIONS.index(r.source),REGIONS.index(r.target),"s",ms=3,mfc="none",mec="white",mew=.8)
                if part.failures.max(): axes[b,fold].set_title(f"fold {fold}  {band_name}\n失敗あり 下限表示",fontsize=9,color="#9a3412")
                fig.colorbar(image,ax=axes[b,fold],shrink=.6)
        name="N_SF2_stability_"+condition
        save(fig,name,"N-SF2 帯域選択頻度  "+COND_NAMES[condition]+"  白枠は元の選択",["N_ST2_stability_band.csv"],
             "100回の模擬選択記録。PCMCI再探索を実行した結果ではない。失敗時は下限k/N、上限はCSV。")
        supplement_images.append(name)
        fig,ax=plt.subplots(figsize=(17,12))
        columns=[(fold,lag) for fold in range(5) for lag in range(1,16)]
        part=selection[selection.condition==condition]
        matrix=part.pivot(index=["target","source"],columns=["outer_fold","lag"],values="frequency_low").reindex(index=pd.MultiIndex.from_tuples(pair_order),columns=pd.MultiIndex.from_tuples(columns))
        cmap=plt.get_cmap("viridis").copy();cmap.set_bad("#dddddd")
        im=ax.imshow(matrix,aspect="auto",cmap=cmap,vmin=0,vmax=1)
        ax.set_yticks(range(56),[f"{NAMES[s]}→{NAMES[t]}" for t,s in pair_order],fontsize=8)
        ax.set_xticks([fold*15+lag-1 for fold in range(5) for lag in (1,8,15)], [f"F{fold} τ{lag}" for fold in range(5) for lag in (1,8,15)],fontsize=8)
        for edge in (14.5,29.5,44.5,59.5):ax.axvline(edge,color="white",lw=1.5)
        fig.colorbar(im,ax=ax,shrink=.75,label="選択頻度 失敗時は下限")
        name="N_SF2_exact_lag_"+condition
        save(fig,name,"N-SF2 全lagの選択記録  "+COND_NAMES[condition],["../input/selection.csv"])
        supplement_images.append(name)

    (root/"captions.json").write_text(json.dumps(captions,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return write_markdown(root,config,captions,supplement_images)


def write_markdown(root: Path, config: dict, captions: dict, supplement_images: list[str]) -> Path:
    tables=root/"tables"
    link=lambda label,path:f"[{label}]({_markdown_path(root,path)})"
    def img(name):
        caption=captions[name]
        path = root/"figures"/(name+".png")
        return f"![{caption['title']}]({_markdown_path(root,path)})\n\n{caption['caption']}\n"
    def table(frame):
        return frame.to_markdown(index=False,floatfmt=".3f",missingval="評価不能")
    def fmt(value):
        return "評価不能" if pd.isna(value) else f"{value*SCALE:.3f}"
    errors=pd.read_csv(tables/"N_T2_model_errors.csv")
    effects=pd.read_csv(tables/"N_T2_paired_effects.csv")
    enrich=pd.read_csv(tables/"N_T3_enrichment.csv")
    boundary=pd.read_csv(tables/"N_T3_boundary.csv")
    band=pd.read_csv(tables/"N_ST2_band_summary.csv")
    t1=pd.read_csv(tables/"N_T1_dataset_support.csv")
    model_input=pd.read_csv(root/"input/metrics.csv")
    main=[]
    for condition in CONDITIONS:
        for model in MODEL_NAMES:
            row=errors[(errors.condition==condition)&(errors.target=="whole_face")&(errors.model==model)].iloc[0]
            main.append({"条件":COND_NAMES[condition],"モデル":MODEL_NAMES[model],"E中央値":fmt(row["median"]),
                         "名目95%区間":f"{fmt(row.ci_low)} ～ {fmt(row.ci_high)}","人数":int(row.n_subjects),"group数":int(row.n_groups)})
    contrasts=[]
    for row in effects[effects.target=="whole_face"].itertuples():
        contrasts.append({"条件":COND_NAMES[row.condition],"対応差":row.contrast,"中央値":fmt(row.median),"名目95%区間":f"{fmt(row.ci_low)} ～ {fmt(row.ci_high)}","人数":row.n_subjects})
    enrichment_rows=[]
    for row in enrich[enrich.target=="selected_targets"].itertuples():
        enrichment_rows.append({"条件":COND_NAMES[row.condition],"selected平均Gの中央値":fmt(row.selected_mean_median),
                                "random期待値の中央値":fmt(row.random_mean_median),"Z中央値":fmt(row.median),
                                "名目95%区間":f"{fmt(row.ci_low)} ～ {fmt(row.ci_high)}","人数":row.n_subjects,"group数":row.n_groups})
    overall_boundary=boundary.groupby("condition",as_index=False)[["n_selected","n_interior","n_boundary","failed_edge_subjects"]].sum()
    overall_boundary["境界除外率 %"]=100*overall_boundary.n_boundary/overall_boundary.n_selected
    overall_boundary.condition=overall_boundary.condition.map(COND_NAMES)
    overall_boundary=overall_boundary.rename(columns={"condition":"条件","n_selected":"選択edge数","n_interior":"対称grid適格","n_boundary":"境界除外","failed_edge_subjects":"失敗edge被験者単位"})
    sizes=[]
    for (condition,model),part in model_input.groupby(["condition","model"],sort=True):
        if model=="persistence":continue
        good=part[part.status=="evaluable"].drop_duplicates(["outer_fold","target"])
        sizes.append({"条件":COND_NAMES[condition],"モデル":MODEL_NAMES[model],
                      "Self列数範囲":f"{good.self_scalar_count.min()}–{good.self_scalar_count.max()}",
                      "追加block数範囲":f"{good.added_blocks.min()}–{good.added_blocks.max()}",
                      "追加scalar列数範囲":f"{good.added_scalar_count.min()}–{good.added_scalar_count.max()}"})
    dataset=t1[["condition","n_subjects","n_groups","original_seconds","after_calibration_seconds","after_qc_seconds","after_boundary_history_seconds","common_support_seconds"]].copy()
    dataset.condition=dataset.condition.map(COND_NAMES)
    dataset.columns=["条件","人数","group数","元時間 秒","校正後 秒","QC後 秒","境界履歴後 秒","共通support 秒"]
    text=f"""# 顔部位間の遅延予測構造 解析レポート草案

作成日 2026-09-11  
版 {config['protocol_id']}  
**{NOTICE}**

本書は、正規化変位の平均ユークリッド誤差を主指標とする解析の、図表と文章の草案である。30人・15組の架空groupから作った模擬予測を集約し、全候補G、被験者間の共通性、時間構造、4条件の予測、選択集合の濃縮を同じ指標で表示した。数値と形状は生成規則の結果であり、顔運動に関する実験上の発見ではない。

**今回実際に行った処理**は、模擬座標・予測からの主誤差計算、全候補Gの集約、1,000ランダム集合との比較、選択lag中心の集約、group単位10,000回のbootstrap、図表・本文の生成である。PCMCI探索、Ridge学習・inner調整、映像からの点抽出、実データのQCは実行していない。選択履歴100回とQC値も模擬値である。

## 1 解析対象と主指標

主誤差は、各点のx・yの予測差のユークリッド距離を取り、領域内の点と被験者内の評価時点を等重みで平均したEとする。点をまとめた重心の誤差や速度RMSEではない。全顔は先に8領域を等重み平均する。

`E = mean_time mean_point sqrt((pred_x − true_x)^2 + (pred_y − true_y)^2)`

`G(source→target, lag) = E_Self − E_(Self＋単一source領域lagブロック)`

G>0は追加ブロックにより誤差が小さくなることを示す。cell-Gは加算可能な効果ではなく、選択集合の同時入力による改善量とも異なる。表のE・G・Z・Rは眼角中点間距離比の **×10⁻³** 表示であり、改善率%やmmではない。

模擬設定は30 fps、h=1、他部位lagは1〜15 frame（33.3〜500 ms）、Self履歴30 frame。帯域は0〜100、100超〜250、250超〜500 msで、それぞれ3・4・8 lagを含む。点数は8領域29点58成分。このfps・履歴・点mappingが実データで確定・検証済みという意味ではない。

### N-T1 データと評価support

{table(dataset)}

各被験者の各条件は900評価frameで、各図表は同じsupport hashを使う。上表の工程時間は表示確認用の模擬QC台帳。最後のfoldでは右頬を学習側使用不能とする例を置いたため、右頬関連は24人、他の部位対は原則30人で評価する。全顔4条件は24人・12groupとなる。1個のcell計算失敗も注入し、対応する帯域の部分平均を禁止している。

## 2 全候補の追加予測情報

{img('N_F1_landscape')}

各条件168個の部位対×帯域セルをすべて描いた。模擬値は正・負・ほぼ0が混在するよう生成した。本文を実データで書き換える際は、部位対と帯域、Gの大きさ、正の割合と分母を併記する。結果から最大lagや良かったcellだけを選び直さない。

{img('N_F2_commonality')}

同じ中央値でも被験者ごとの符号は異なり得るため、G>0人数と全被験者の値を併せて報告する。分母が異なるセルの大小を直接比較する場合は、共通被験者で対応差を再計算する。{link('全被験者のGとfold別の補足図',root/'supplement.md')}に個別分布を示した。

## 3 絶対lagと中心化応答

{img('N_F3_absolute_speaking')}

{img('N_F3_absolute_non_speaking')}

絶対lag曲線は全56部位対を保持した。生成規則には広いピーク、上限付近まで続く形、口→顎の平坦な形を含む。これは図の読み分けを確認するための形状である。絶対lagの上限に改善が続く場合は終端未確認と記載する。

{img('N_F3_centered')}

中心化はSelf＋単一ブロックの誤差差を、同じ部位対の選択lag平均、被験者内の部位対平均、被験者間中央値の順で集約した。±3 frame（±100 ms）の全gridで対象を固定した。選択lagからずらして誤差が下がる負値も残す。Δ=0の0は定義上の値で、時間特異性の証拠にはしない。

### N-T3B 中心化対象と境界除外

{table(overall_boundary)}

edge数はfold・条件・部位対・lagの一意な選択数であり、被験者数ではない。失敗edge被験者単位は境界除外後の異なる分母なので別列にした。foldごとの分子・分母と各Δの被験者・部位対・edge数は数値表に保存した。

## 4 四条件の予測と入力数

{img('N_F4_models')}

### N-T2A 全顔の主誤差

{table(pd.DataFrame(main))}

### N-T2B 被験者内の対応差

{table(pd.DataFrame(contrasts))}

正の対応差は左側のモデルから右側のモデルへの改善を示す。対応差の中央値は、モデル別中央値の差とは限らない。空選択の場合はPCMCI-blockをSelfと同じ予測にして0差を残した。Fullとの同等性・非劣性や疎性の合格判定は置いていない。

### N-T2C 入力規模

{table(pd.DataFrame(sizes))}

範囲の単位は評価可能なfold×targetである。Persistenceは現在のtarget変位をそのまま使い、Ridge用の履歴列数表からは分けた。図の模型予測器は実際にRidgeで学習していないため、入力数は選択mappingから計算した設計上の列数である。

## 5 選択集合の濃縮

{img('N_F5_enrichment')}

### N-T3A 評価可能な選択targetの要約

{table(pd.DataFrame(enrichment_rows))}

targetごとの選択cell-G平均から、ブロック数と実scalar成分数を揃えた1,000ランダム集合の平均Gの算術平均を引いた。各被験者でtargetを等重み平均してから集団中央値を取る。表の3列はそれぞれ個別値から要約しているため、表示された中央値同士の差がZ中央値に一致するとは限らない。

ランダム集合は模擬test値の生成前に固定し、同じfoldの全被験者で共有した。集合内重複なし、選択集合との重なりと反復間重複は許した。空選択はZ=0とせず評価不能。予定targetの計算失敗は成功targetだけの要約へ置き換えず、集約失敗として残した。1,000回を独立人数や厳密な置換p値には用いない。

{link('全被験者とtargetの1,000集合分布',root/'supplement.md')}も保存した。正のZでもselected平均Gが負なら、Selfを改善したという結論にはならない。

## 6 不確かさと解釈の範囲

全図表はouter-fold内で独立groupを復元抽出する同一の10,000抽出indexを使う。選ばれたgroupの全被験者をまとめて残し、各集約順序を保った。区間は固定OOF値における被験者構成の変動を示す名目95%区間である。学習集合の重複や探索・学習をやり直す不確かさを解決するものではない。10group未満では区間を出さない。曲線は点ごとの区間で、多重比較補正済み・同時区間ではない。

本草案では有意性検定、星印、p/q値、結果に基づく表示の足切りを使わない。本人発話と本人非発話は別条件の記述であり、差を発話の因果効果とは解釈しない。source→targetは遅延予測関係として述べ、生理学的な直接因果や伝達時間とはしない。

## 7 実データ版への差し替え箇所

データ利用と独立group、採用点と測定品質、fps・Self履歴・共通support、実際の探索・Ridge調整・再学習を確認した後、承認protocolの実artifactから全指標を再計算する。旧速度RMSE集計を名称変更して読み込むことはできない。本草案生成は本実験freezeや科学的受容の完了ではない。

{link('補足資料 図表 設定 全数値への索引',root/'supplement.md')}<br>
{link('生成設定',root/'input/config.json')}<br>
{link('出力registry',root/'analysis_artifact_registry.csv')}<br>
{link('出力manifest',root/'analysis_manifest.json')}

再生成コマンド（repository root、analysis依存関係を導入したPython）:

```powershell
& .venv/Scripts/python.exe scripts/generate_displacement_mock_report.py
```
"""
    report=root/"analysis_report_draft.md"
    report.write_text(text,encoding="utf-8")
    settings=pd.DataFrame({"項目":["主指標","予測","他部位lag","Self履歴","中心化幅","ランダム集合","集団区間","選択履歴","実学習","科学的受容"],
                           "値":[config["primary_metric"],"h=1 / 30 fps 模擬設定","1–15 frame","30 frame","±3 frame","1,000 / 実成分数層化","10,000 / fold内group抽出","100回の模擬記録","未実行","未実施"]})
    write_csv(settings,tables/"N_ST1_configuration.csv")
    mapping=pd.DataFrame({"領域":LABELS,"点index":[", ".join(map(str,p)) for p in POINTS],"点数":list(map(len,POINTS)),"成分数":[len(p)*2 for p in POINTS]})
    write_csv(mapping,tables/"N_ST1_point_mapping.csv")
    supplemental=["# 解析レポート草案 補足資料",f"**{NOTICE}**",link("本文へ",report),
                  "## N-ST1 設定と測定点",table(settings),table(mapping),
                  "全点の具体的配置と抽出品質は未確認。図は概念図である。以下の実数値表を先に保存し、その値から描画した。",
                  "## 補足図"]
    for name in supplement_images:
        supplemental.extend(["### "+captions[name]["title"],img(name)])
    supplemental.append("## N-ST2 全数値表と再生成元")
    for path in sorted(tables.iterdir()):
        supplemental.append("- "+link(path.name,path))
    supplemental.append("## 模擬予測の再構成")
    supplemental.append("input/coordinates.npzに正解・直前値・単位残差を保存した。各metrics/cells行のcoordinates_keyで配列を引き、正解＋residual_scale×単位残差で予測を再構成できる。Persistenceはprevious配列である。supportは共通900frame。失敗・不能行の残差scaleを有効な成果として使わない。")
    for path in sorted((root/"input").iterdir()):supplemental.append("- "+link(path.name,path))
    supplemental.append("- "+link("共有bootstrap抽出index",root/"bootstrap_subject_indices.npy"))
    supplemental.append("- "+link("用途別seed",root/"seed_registry.json"))
    supplemental.append("- "+link("図のcaptionと出典表",root/"captions.json"))
    (root/"supplement.md").write_text("\n\n".join(supplemental)+"\n",encoding="utf-8")
    return report
