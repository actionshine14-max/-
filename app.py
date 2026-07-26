
from __future__ import annotations
import json, math, re
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
import streamlit as st

SCENARIOS=("S1","S2","S3")
def clamp(x,lo=0.0,hi=10.0): return float(max(lo,min(hi,x)))
def logistic(x): return 1/(1+math.exp(-x))
def comb(*ps):
    q=1.0
    for p in ps: q*=1-min(max(float(p),0),1)
    return 1-q
def nrank(pos,n):
    if pos is None or not n:return None
    return clamp(10*(1-(pos-1)/max(1,n-1)))
def wmean(v,w):
    if not v:return None
    a=np.asarray(v,float); b=np.asarray(w[:len(v)],float)
    return float(np.average(a,weights=b)) if b.sum()>0 else float(a.mean())

HEADER=re.compile(r"(20\d{2})年(\d{1,2})月(\d{1,2})日.*?(札幌|函館|福島|新潟|東京|中山|中京|京都|阪神|小倉).*?(\d{1,2})レース",re.S)
DIST=re.compile(r"コース：?\s*([\d,]+)メートル（(芝|ダート)・(右|左)")
HORSE=re.compile(r"枠(\d+)[^\n]*\n(\d+)\s*(?:\nブリンカー着用)?\n([^\n]+)")
DATE=re.compile(r"(20\d{2})年(\d{1,2})月(\d{1,2})日")
COURSE=re.compile(r"(\d{3,4})(芝|ダ)")
PASS=re.compile(r"^\s*(\d{1,2})(?:\s+(\d{1,2}))?(?:\s+(\d{1,2}))?(?:\s+(\d{1,2}))?\s*$",re.M)
UP=re.compile(r"3F\s*(\d{2}\.\d)")
BW=re.compile(r"(\d{3})kg\((初出走|[+-]\d+|0)\)")
ODDS=re.compile(r"(\d+(?:\.\d+)?)\s*\n\((\d+)番人気\)")
LOAD=re.compile(r"(\d{2}\.\d)kg")

def parse_runs(block):
    ds=list(DATE.finditer(block)); out=[]
    for i,m in enumerate(ds):
        c=block[m.start():(ds[i+1].start() if i+1<len(ds) else len(block))]
        co=COURSE.search(c); fs=re.search(r"(\d+)着\s+(\d+)頭",c); up=UP.search(c)
        pos=[]
        for pm in PASS.finditer(c):
            z=[int(x) for x in pm.groups() if x]
            if 2<=len(z)<=4 and all(1<=x<=30 for x in z): pos=z
        out.append({"date":f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}",
                    "distance":int(co.group(1)) if co else None,
                    "surface":"dirt" if co and co.group(2)=="ダ" else ("turf" if co else None),
                    "finish":int(fs.group(1)) if fs else None,
                    "field_size":int(fs.group(2)) if fs else None,
                    "positions":pos,"up3f":float(up.group(1)) if up else None})
    return out

def parse_text(text):
    race={}; warns=[]
    h=HEADER.search(text)
    if h: race={"date":f"{h.group(1)}-{int(h.group(2)):02d}-{int(h.group(3)):02d}","track":h.group(4),"race_no":int(h.group(5))}
    else: warns.append("開催情報を抽出できませんでした。")
    d=DIST.search(text)
    if d: race|={"distance":int(d.group(1).replace(",","")),"surface":"dirt" if d.group(2)=="ダート" else "turf","direction":d.group(3)}
    else: warns.append("距離・芝ダートを抽出できませんでした。")
    ms=list(HORSE.finditer(text)); horses=[]
    for i,m in enumerate(ms):
        b=text[m.start():(ms[i+1].start() if i+1<len(ms) else len(text))]
        bw=BW.search(b); od=ODDS.search(b); loads=list(LOAD.finditer(b))
        horses.append({"frame":int(m.group(1)),"number":int(m.group(2)),"name":m.group(3).strip(),
                       "load":float(loads[0].group(1)) if loads else 57.0,
                       "body_weight":int(bw.group(1)) if bw else None,
                       "body_weight_delta":None if not bw or bw.group(2)=="初出走" else int(bw.group(2)),
                       "debut":bool(bw and bw.group(2)=="初出走"),"blinker":"ブリンカー着用" in b,
                       "odds":float(od.group(1)) if od else None,"popularity":int(od.group(2)) if od else None,
                       "runs":parse_runs(b)})
    race["field_size"]=len(horses)
    if not horses: warns.append("馬ブロックを抽出できませんでした。")
    return race,horses,warns

def derive(race,horses):
    n=max(1,len(horses)); td=float(race.get("distance",1800)); surf=race.get("surface","dirt")
    rows=[]
    for h in horses:
        first=[]; last=[]; ws=[]; ups=[]; fins=[]; front=[]
        for i,r in enumerate(h.get("runs",[])[:5]):
            fs=int(r.get("field_size") or n); sim=max(.35,1-abs((r.get("distance") or td)-td)/1000)
            if r.get("surface") and r.get("surface")!=surf: sim*=.45
            w=sim*math.exp(-.22*i)
            if r.get("positions"):
                first.append(nrank(r["positions"][0],fs)); last.append(nrank(r["positions"][-1],fs)); ws.append(w)
                front.append(1 if r["positions"][0]<=max(3,math.ceil(fs*.25)) else 0)
            if r.get("up3f") is not None: ups.append(float(r["up3f"]))
            if r.get("finish") is not None: fins.append(nrank(r["finish"],fs))
        conf=clamp(10*(.55*min(len(h.get("runs",[])),5)/5+.45*min(len(ups),5)/5))
        ep=wmean(first,ws) if first else 5.0; cp=wmean(last,ws) if last else ep
        close=clamp(10-(np.mean(ups)-34)*.95) if ups else 5.0
        fq=float(np.mean(fins)) if fins else 5.0
        inner=clamp(10*(1-(h["number"]-1)/max(1,n-1))); outer=10-inner
        load=float(h.get("load") or 57); relief=clamp(5+(57-load)*.9)
        delta=float(h.get("body_weight_delta") or 0); stable=clamp(10-abs(delta)*.22)
        cruise=.5*cp+.25*fq+.25*stable; reacc=.45*cp+.35*close+.2*relief; straight=.55*close+.25*fq+.2*relief
        if h.get("debut"): conf=min(conf,2)
        miss=clamp((10-ep)*.055+(1-conf/10)*.18,.03,.70)
        trap=clamp((inner/10)*(.12+max(0,5-ep)*.025),.02,.48); wide=clamp((outer/10)*.42,.02,.52)
        block=clamp(.10+(1-conf/10)*.15+trap*.30,.04,.55)
        fatigue=clamp(4.8+max(0,td-1600)/500+(10-cruise)*.23-(relief-5)*.18)
        loss=clamp(logistic((fatigue-straight)*.45),.05,.85)
        rows.append({"馬番":h["number"],"馬名":h["name"],"枠":h["frame"],"斤量":load,"馬体重":h.get("body_weight"),
                     "増減":h.get("body_weight_delta"),"初出走":h.get("debut",False),"有効近走数":len(h.get("runs",[])),
                     "入力信頼度":round(conf,3),"初速":round(ep,3),"巡航":round(clamp(cruise),3),
                     "再加速":round(clamp(reacc),3),"直線余力":round(clamp(straight),3),"内枠優位":round(inner,3),
                     "外枠負荷":round(outer,3),"逃げ番手率":round(float(np.mean(front)) if front else .25,4),
                     "出遅れ推定率":round(miss,4),"内包まれ率":round(trap,4),"外回し率":round(wide,4),
                     "進路詰まり率":round(block,4),"疲労EV":round(fatigue,3),"ラスト失速率":round(loss,4),
                     "value_type":"DERIVED/ESTIMATE"})
    return pd.DataFrame(rows).sort_values("馬番").reset_index(drop=True)

def worlds(base):
    d=base.copy(); n=len(d)
    raw=np.exp(.35*d["初速"]/10+.25*d["逃げ番手率"]+.15*d["内枠優位"]/10-.20*d["出遅れ推定率"]-.15*d["外枠負荷"]/10)
    d["ハナ取得確率基礎"]=raw/raw.sum()
    fc=int((d["初速"]>=d["初速"].quantile(.70)).sum())
    p=clamp(fc*1.15+d.nlargest(min(5,n),"初速")["逃げ番手率"].sum()*1.4+d["外枠負荷"].mean()*.22)
    clos=int((d["直線余力"]-d["初速"]>=1).sum()); unc=clamp(10-d["入力信頼度"].mean()+d["初速"].std(ddof=0))
    w=pd.DataFrame([
      ["S1","自然隊列・競り緩和",clamp(p-1.6),clamp(.12+.025*clos+.015*p,0,1),.85,.82,.82],
      ["S2","前圧継続・中団差し接続",clamp(p+1.3),clamp(.28+.04*clos+.025*p,0,1),1.10,1.22,1.05],
      ["S3","反相関・構造穴",clamp(p+(.8 if p>=5 else -1)),clamp(.22+.045*clos+.035*unc,0,1),1.45,1.10,1.38]],
      columns=["シナリオ","説明","前列圧","差し接続率","位置分散倍率","前列失速倍率","進路摩擦倍率"])
    rr=np.array([max(.1,8-.55*p-.25*unc),max(.1,1.5+.72*p+.35*clos),max(.1,1+.65*unc+.25*abs(fc-2))])
    w["発生確率"]=rr/rr.sum()
    w["荒れ前残り発生率"]=clamp(logistic(-.7+.18*fc+.10*(10-p)+.08*(d["斤量"]<=54).sum()),.05,.85)
    return d,w

def scenario_inputs(base,w):
    out=[]
    for _,z in w.iterrows():
      for _,h in base.iterrows():
        f=h["初速"]/10; closer=max(0,(h["直線余力"]-h["初速"])/10)
        recv=clamp(z["前列圧"]*(.4+.75*f)+h["外枠負荷"]*.14+h["内包まれ率"]*2)
        gen=clamp(.58*h["初速"]+2.4*h["逃げ番手率"]+.18*h["外枠負荷"])
        if z["シナリオ"]=="S1": early=clamp(h["初速"]+.45*h["内枠優位"]/5-.25*recv); dc=z["差し接続率"]*(.65+.7*closer)
        elif z["シナリオ"]=="S2": early=clamp(h["初速"]-.42*recv+.30*h["再加速"]); dc=z["差し接続率"]*(.70+1.15*closer)
        else:
          structural=(57-h["斤量"])*.22+h["内枠優位"]*.06+(10-h["入力信頼度"])*.05
          early=clamp(h["初速"]-.30*recv+structural); dc=z["差し接続率"]*(.75+1.2*closer)
        fail=comb(h["出遅れ推定率"],h["内包まれ率"]*z["進路摩擦倍率"],h["外回し率"]*.45,min(.7,recv/18))
        loss=clamp(h["ラスト失速率"]*z["前列失速倍率"]*(.82+.45*f),.03,.92)
        remain=clamp(h["直線余力"]+.32*h["巡航"]+.28*h["再加速"]-.36*recv-1.8*loss)
        route=clamp(1-comb(h["進路詰まり率"]*z["進路摩擦倍率"],h["外回し率"]*.45,h["内包まれ率"]*.35),.05,.98)
        out.append({"馬番":h["馬番"],"馬名":h["馬名"],"シナリオ":z["シナリオ"],"シナリオ発生確率":z["発生確率"],
                    "ハナ取得確率":h["ハナ取得確率基礎"],"先手参加率":clamp(.35*h["ハナ取得確率基礎"]+.65*logistic((early-5.2)/1.2),0,1),
                    "前列圧発生EV":gen,"前列圧被害EV":recv,"序盤位置不発率":fail,"差し接続確率":clamp(dc,0,1),
                    "残存エネルギーEV":remain,"進路実現確率":route,"ラスト失速率":loss,
                    "位置分散倍率":z["位置分散倍率"],"荒れ前残り発生率":z["荒れ前残り発生率"],
                    "入力信頼度":h["入力信頼度"],"初速":h["初速"],"巡航":h["巡航"],"再加速":h["再加速"],"直線余力":h["直線余力"]})
    return pd.DataFrame(out)

def simulate(inp,trials,seeds):
    rec=[]
    for sid in SCENARIOS:
      d=inp[inp["シナリオ"]==sid].reset_index(drop=True); n=len(d); rows=[]; per=max(50,trials//seeds)
      for seed in range(seeds):
        rng=np.random.default_rng(20260726+seed*1009+SCENARIOS.index(sid))
        for _ in range(per):
          sig=d["位置分散倍率"].to_numpy()*(.45+(1-d["入力信頼度"].to_numpy()/10)*.85)
          start=d["初速"].to_numpy()+rng.normal(0,sig)-rng.binomial(1,d["序盤位置不発率"].to_numpy())*rng.uniform(.7,2.2,n)
          mid=.46*start+.34*d["巡航"].to_numpy()+.20*d["再加速"].to_numpy()-.26*d["前列圧被害EV"].to_numpy()
          conn=rng.binomial(1,d["差し接続確率"].to_numpy())
          cs=mid+conn*d["再加速"].to_numpy()*.22+rng.normal(0,sig*.55)
          order=np.argsort(-cs); cr=np.empty(n,int); cr[order]=np.arange(1,n+1)
          blocked=rng.binomial(1,1-d["進路実現確率"].to_numpy()); faded=rng.binomial(1,d["ラスト失速率"].to_numpy())
          fs=.34*cs+.42*d["残存エネルギーEV"].to_numpy()+.24*d["直線余力"].to_numpy()-blocked*rng.uniform(.5,1.8,n)-faded*rng.uniform(.7,2.5,n)+rng.normal(0,sig*.45)
          order=np.argsort(-fs); fr=np.empty(n,int); fr[order]=np.arange(1,n+1)
          for i in range(n): rows.append([d.loc[i,"馬番"],d.loc[i,"馬名"],cr[i],fr[i],blocked[i],faded[i]])
      r=pd.DataFrame(rows,columns=["馬番","馬名","4角順位","着順","進路不発","失速"])
      for (num,name),g in r.groupby(["馬番","馬名"],sort=False):
        rec.append({"シナリオ":sid,"馬番":num,"馬名":name,"4角順位平均":g["4角順位"].mean(),"4角順位SD":g["4角順位"].std(ddof=0),
                    "4角3位内率":(g["4角順位"]<=3).mean(),"勝率":(g["着順"]==1).mean(),"複勝率":(g["着順"]<=3).mean(),
                    "平均着順":g["着順"].mean(),"実測進路不発率":g["進路不発"].mean(),"実測失速率":g["失速"].mean()})
    return pd.DataFrame(rec)

def audit(res,inp):
    rows=[]
    for a,b in [("S1","S2"),("S1","S3"),("S2","S3")]:
      x=res[res["シナリオ"]==a].set_index("馬番"); y=res[res["シナリオ"]==b].set_index("馬番"); c=x.index.intersection(y.index)
      corr=x.loc[c,"4角順位平均"].corr(y.loc[c,"4角順位平均"]) if len(c)>=3 else np.nan
      overlap=len(set(x.nsmallest(min(5,len(x)),"平均着順").index)&set(y.nsmallest(min(5,len(y)),"平均着順").index))
      pa=inp[inp["シナリオ"]==a]["前列圧被害EV"].mean(); pb=inp[inp["シナリオ"]==b]["前列圧被害EV"].mean(); diff=abs(pa-pb)
      status="NG: 分離不足" if pd.notna(corr) and corr>=.90 and overlap>=4 and diff<.75 else ("WARN" if pd.notna(corr) and corr>=.82 and overlap>=4 else "OK")
      rows.append({"比較":f"{a}-{b}","4角順位相関":corr,"上位5頭重複":overlap,"平均前列圧差":diff,"判定":status})
    return pd.DataFrame(rows)

st.set_page_config(page_title="OADPシナリオ分離シミュレーター",layout="wide")
st.title("OADP シナリオ分離型レースシミュレーター")
st.caption("試作版 / FACT・DERIVED・ESTIMATE分離 / 人気・オッズを物理計算に不使用")
with st.sidebar:
    trials=st.slider("各シナリオ試行数",300,10000,3000,300)
    seeds=st.slider("シード数",1,30,10)
tabs=st.tabs(["入力","基礎数値","シミュレーション"])
with tabs[0]:
    text=st.text_area("JRA出馬表テキストを貼り付け",height=420)
    f=st.file_uploader("またはTXT",type=["txt"])
    if f: text=f.getvalue().decode("utf-8",errors="replace")
    if text:
      race,horses,warns=parse_text(text)
      for w in warns: st.warning(w)
      st.json(race)
      if horses: st.dataframe(pd.DataFrame([{k:v for k,v in h.items() if k!="runs"}|{"近走数":len(h["runs"])} for h in horses]),use_container_width=True)
      if st.button("基礎数値を生成",type="primary"):
        st.session_state["race"]=race; st.session_state["base"]=derive(race,horses); st.success("生成しました")
with tabs[1]:
    if "base" not in st.session_state: st.info("入力タブから生成してください。")
    else:
      st.session_state["base"]=st.data_editor(st.session_state["base"],use_container_width=True,num_rows="fixed")
      st.download_button("基礎CSV",st.session_state["base"].to_csv(index=False).encode("utf-8-sig"),"oadp_base.csv")
with tabs[2]:
    if "base" not in st.session_state: st.info("基礎数値がありません。")
    else:
      if st.button("S1/S2/S3生成・実行",type="primary"):
        b,w=worlds(st.session_state["base"]); i=scenario_inputs(b,w); r=simulate(i,trials,seeds); a=audit(r,i)
        st.session_state.update(world=w,inputs=i,result=r,audit=a)
      if "world" in st.session_state:
        st.subheader("シナリオ世界"); st.dataframe(st.session_state["world"],use_container_width=True)
        st.subheader("全頭×S1/S2/S3入力"); st.dataframe(st.session_state["inputs"],use_container_width=True)
        st.subheader("結果"); st.dataframe(st.session_state["result"].sort_values(["シナリオ","平均着順"]),use_container_width=True)
        st.subheader("独立監査"); st.dataframe(st.session_state["audit"],use_container_width=True)
        if st.session_state["audit"]["判定"].str.startswith("NG").any(): st.error("分離監査NG。係数または入力世界の再調整が必要です。")
        export={"race":st.session_state.get("race",{}),"worlds":st.session_state["world"].to_dict("records"),
                "scenario_inputs":st.session_state["inputs"].to_dict("records"),"results":st.session_state["result"].to_dict("records"),
                "audit":st.session_state["audit"].to_dict("records"),"notice":"試作モデルのDERIVED/ESTIMATE"}
        st.download_button("結果JSON",json.dumps(export,ensure_ascii=False,indent=2).encode(),"oadp_simulation.json","application/json")
st.divider()
st.caption("未固定係数は暫定式です。公式結果による校正前のため、真の確率を保証しません。")
