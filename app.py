
from __future__ import annotations
import json, math, re
import requests
from urllib.parse import urlparse
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
import streamlit as st
from oadp_engine import parse_nar_html

APP_VERSION="0.2.0-step0-parser-integrated"
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
HORSE_STRICT=re.compile(r"枠\\s*(\\d+)[^\\n]*\\n\\s*(\\d{1,2})\\s*(?:\\n\\s*)*(?:ブリンカー着用\\s*(?:\\n\\s*)*)?([^\\n]+)")
FRAME_MARK=re.compile(r"(?m)^\\s*(?:枠\\s*(\\d+)|(\\d+)\\s*枠)\\s*$")
DATE=re.compile(r"(20\d{2})年(\d{1,2})月(\d{1,2})日")
COURSE=re.compile(r"(\d{3,4})(芝|ダ)")
PASS=re.compile(r"^\s*(\d{1,2})(?:\s+(\d{1,2}))?(?:\s+(\d{1,2}))?(?:\s+(\d{1,2}))?\s*$",re.M)
UP=re.compile(r"3F\s*(\d{2}\.\d)")
BW=re.compile(r"(\d{3})kg\((初出走|[+-]\d+|0)\)")
ODDS=re.compile(r"(\d+(?:\.\d+)?)\s*\n\((\d+)番人気\)")
LOAD=re.compile(r"(\d{2}\.\d)kg")


def fetch_nar_html(race_url):
    """
    STEP0作成アプリ v3.8 と同じ考え方で地方競馬公式URLを取得する。
    SSRF対策として http/https と keiba.go.jp 系だけを許可する。
    """
    url = (race_url or "").strip()
    if not url:
        raise ValueError("URLが空です。")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("正しいhttp/https URLを入力してください。")
    host = (parsed.hostname or "").lower()
    if not (host == "keiba.go.jp" or host.endswith(".keiba.go.jp")):
        raise ValueError("地方競馬公式 keiba.go.jp の出馬表URLを入力してください。")

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; OADPSimulator/0.2)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.7,en;q=0.5",
        "Cache-Control": "no-cache",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=25, allow_redirects=True)
    except requests.RequestException as e:
        raise ValueError(f"URL取得に失敗しました: {e}") from e

    if resp.status_code >= 400:
        raise ValueError(f"URL取得に失敗しました: HTTP {resp.status_code}")
    if not resp.encoding or resp.encoding.lower() in {"iso-8859-1", "ascii"}:
        resp.encoding = resp.apparent_encoding or "utf-8"
    html = resp.text
    if len(html.strip()) < 500:
        raise ValueError("取得HTMLが短すぎます。出馬表公開前、URL誤り、またはアクセス制限の可能性があります。")
    return html


def nar_race_to_sim_input(race_obj):
    """
    STEP0作成アプリの parse_nar_html() が返す Race/Horse/PastRun を
    本シミュレーター共通入力へ変換する。
    """
    course_text = str(getattr(race_obj, "course", "") or "")
    surface = "turf" if "芝" in course_text else "dirt"
    race = {
        "date": str(getattr(race_obj, "date_text", "") or ""),
        "track": str(getattr(race_obj, "track", "") or ""),
        "race_no": getattr(race_obj, "race_no", None),
        "title": str(getattr(race_obj, "title", "") or ""),
        "distance": getattr(race_obj, "distance", None),
        "surface": surface,
        "direction": str(getattr(race_obj, "direction", "") or ""),
        "weather": str(getattr(race_obj, "weather", "") or ""),
        "going": str(getattr(race_obj, "going", "") or ""),
        "field_size": len(getattr(race_obj, "horses", []) or []),
        "source": "NAR_URL_STEP0_PARSER",
    }

    horses = []
    for h in getattr(race_obj, "horses", []) or []:
        runs = []
        for r in getattr(h, "past_runs", []) or []:
            pos = list(getattr(r, "passing_positions", []) or [])
            course = str(getattr(r, "course", "") or "")
            runs.append({
                "date": str(getattr(r, "date", "") or ""),
                "distance": getattr(r, "distance", None),
                "surface": "turf" if "芝" in course else ("dirt" if course else None),
                "finish": getattr(r, "rank", None),
                "field_size": getattr(r, "heads", None),
                "positions": pos,
                "up3f": getattr(r, "agari", None),
            })

        horses.append({
            "frame": getattr(h, "frame_no", None),
            "number": getattr(h, "horse_no", None),
            "name": str(getattr(h, "name", "") or ""),
            "load": getattr(h, "carried_weight", None),
            "body_weight": getattr(h, "body_weight", None),
            "body_weight_delta": getattr(h, "body_weight_diff", None),
            "debut": len(runs) == 0,
            "blinker": "ブリンカー" in str(getattr(h, "equipment_notes", "") or ""),
            "odds": getattr(h, "odds", None),
            "popularity": getattr(h, "popularity", None),
            "runs": runs,
            "jockey": str(getattr(h, "jockey", "") or ""),
            "trainer": str(getattr(h, "trainer", "") or ""),
            "sex_age": str(getattr(h, "sex_age", "") or ""),
            "source_type": "FACT",
        })
    return race, horses


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


def _clean_lines(text):
    """
    改行・タブ・全角空白・NBSPを正規化する。
    splitlines()を使うことでWindows/Unix/Mac改行を同時に処理する。
    """
    if text is None:
        return []
    normalized = (
        str(text)
        .replace("\ufeff", "")
        .replace("\u00a0", " ")
        .replace("\u200b", "")
    )
    return [
        re.sub(r"[ \t\u3000]+", " ", line).strip()
        for line in normalized.splitlines()
    ]


def _horse_blocks_flexible(text):
    """
    JRA出馬表コピー形式の枠・馬番・馬名を抽出する。

    対応形式:
      枠1白<TAB>1
      枠1白 1
      枠1白
      1
      1枠 白 1

    ブリンカー着用が馬名の前に挿入されても処理する。
    """
    lines = _clean_lines(text)
    starts = []

    # 枠色が枠番号へ直結する形式を含む。
    start_re = re.compile(
        r"^(?:"
        r"枠\s*(?P<frame1>\d+)\s*(?:白|黒|赤|青|黄|緑|橙|桃)?\s*(?P<num1>\d{1,2})?"
        r"|"
        r"(?P<frame2>\d+)\s*枠\s*(?:白|黒|赤|青|黄|緑|橙|桃)?\s*(?P<num2>\d{1,2})?"
        r")$"
    )

    for i, line in enumerate(lines):
        m = start_re.fullmatch(line)
        if not m:
            continue
        frame = int(m.group("frame1") or m.group("frame2"))
        inline_num = m.group("num1") or m.group("num2")
        starts.append((i, frame, int(inline_num) if inline_num else None))

    blocks = []
    for j, (idx, frame, inline_num) in enumerate(starts):
        block_end = starts[j + 1][0] if j + 1 < len(starts) else len(lines)
        seg = lines[idx:block_end]

        number = inline_num
        number_i = 0 if inline_num is not None else None

        # 馬番が次行へ分離された形式
        if number is None:
            for k, line in enumerate(seg[1:16], start=1):
                if re.fullmatch(r"\d{1,2}", line):
                    v = int(line)
                    if 1 <= v <= 30:
                        number = v
                        number_i = k
                        break

        if number is None:
            continue

        search_from = 1 if number_i == 0 else number_i + 1
        name = None
        skip_words = {
            "ブリンカー着用", "取消", "除外", "競走除外",
            "勝負服の画像", "馬柱の見方", "着順で色分け", "同一レースで色分け",
        }

        for line in seg[search_from:search_from + 16]:
            if not line or line in skip_words:
                continue
            if re.fullmatch(r"(牡|牝|せん)\d+(?:/.*)?", line):
                continue
            if re.fullmatch(r"\d{2}(?:\.\d)?kg", line):
                continue
            if re.fullmatch(r"\d+(?:\.\d+)?", line):
                continue
            if re.fullmatch(r"\(\d+番人気\)", line):
                continue
            if re.fullmatch(r"\d{3}kg\((?:初出走|[+-]?\d+)\)", line):
                continue
            if line.startswith("枠") and "馬番" in line:
                continue
            name = line.strip()
            break

        if name:
            blocks.append((frame, number, name, "\n".join(seg)))

    return blocks

def _horse_blocks(text):
    flexible=_horse_blocks_flexible(text)
    if flexible:
        return flexible
    ms=list(HORSE_STRICT.finditer(text))
    out=[]
    for i,m in enumerate(ms):
        b=text[m.start():(ms[i+1].start() if i+1<len(ms) else len(text))]
        out.append((int(m.group(1)),int(m.group(2)),m.group(3).strip(),b))
    return out

def parse_text(text):
    race={}; warns=[]
    h=HEADER.search(text)
    if h:
        race={"date":f"{h.group(1)}-{int(h.group(2)):02d}-{int(h.group(3)):02d}","track":h.group(4),"race_no":int(h.group(5))}
    else:
        warns.append("開催情報を抽出できませんでした。")
    d=DIST.search(text)
    if d:
        race|={"distance":int(d.group(1).replace(",","")),"surface":"dirt" if d.group(2)=="ダート" else "turf","direction":d.group(3)}
    else:
        warns.append("距離・芝ダートを抽出できませんでした。")

    horses=[]
    seen=set()
    for frame,number,name,b in _horse_blocks(text):
        if number in seen:
            continue
        seen.add(number)
        bw=BW.search(b); od=ODDS.search(b); loads=list(LOAD.finditer(b))
        horses.append({
            "frame":frame,"number":number,"name":name,
            "load":float(loads[0].group(1)) if loads else 57.0,
            "body_weight":int(bw.group(1)) if bw else None,
            "body_weight_delta":None if not bw or bw.group(2)=="初出走" else int(bw.group(2)),
            "debut":bool(bw and bw.group(2)=="初出走"),
            "blinker":"ブリンカー着用" in b,
            "odds":float(od.group(1)) if od else None,
            "popularity":int(od.group(2)) if od else None,
            "runs":parse_runs(b)
        })
    horses=sorted(horses,key=lambda x:x["number"])
    race["field_size"]=len(horses)
    if not horses:
        warns.append("馬ブロックを抽出できませんでした。枠見出しが「枠1白 1」のように含まれているか確認してください。基礎数値生成は実行されません。")
    return race,horses,warns

def validate_base(df):
    required={"馬番","馬名","初速","巡航","再加速","直線余力","入力信頼度",
              "逃げ番手率","出遅れ推定率","内包まれ率","外回し率","進路詰まり率",
              "ラスト失速率","内枠優位","外枠負荷","斤量"}
    if df is None or not isinstance(df,pd.DataFrame) or df.empty:
        return False,"基礎データが空です。"
    missing=sorted(required-set(df.columns))
    if missing:
        return False,"必要列が不足しています: "+", ".join(missing)
    if df["馬番"].isna().any() or df["馬名"].isna().any():
        return False,"馬番または馬名に欠損があります。"
    return True,""

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
    """
    OADP Ver.2.35: 馬評価より先に馬群構成からS1/S2/S3を作る。
    人気・オッズは使用しない。
    """
    d=base.copy(); n=len(d)
    raw=np.exp(.42*d["初速"]/10+.22*d["逃げ番手率"]+.16*d["内枠優位"]/10-.12*d["出遅れ推定率"]-.08*d["外枠負荷"]/10)
    d["ハナ取得確率基礎"]=raw/raw.sum()

    front_count=int((d["逃げ番手率"]>=.40).sum())
    early_count=int((d["初速"]>=d["初速"].quantile(.65)).sum())
    closer_count=int((d["直線余力"]>=d["初速"]+.6).sum())
    uncertainty=clamp(10-d["入力信頼度"].mean()+d["初速"].std(ddof=0))
    pressure=clamp(1.0+front_count*1.15+early_count*.75+d["逃げ番手率"].sum()*.45)
    support=clamp(d.nlargest(min(3,n),"初速")["初速"].mean())
    k6_density=clamp((d["斤量"].le(55).sum()*1.2)+(d["内枠優位"].ge(7).sum()*.7)+closer_count*.8+uncertainty*.35)
    reversal=clamp(d["初速"].rank(ascending=False).corr(d["直線余力"].rank(ascending=False))*-3+5 if n>=3 else 5)

    s1_raw=max(.1, 8.5+.85*support-.72*pressure-.35*uncertainty)
    s2_raw=max(.1, 5.0+.95*pressure+.55*closer_count+.30*reversal)
    s3_raw=max(.1, 3.5+.80*k6_density+.55*uncertainty+.45*reversal-.30*support)
    rr=np.array([s1_raw,s2_raw,s3_raw],dtype=float)
    probs=rr/rr.sum()

    # 物理的に成立する世界線の下限監査
    floor=np.array([.15,.15,.15])
    probs=np.maximum(probs,floor)
    probs=probs/probs.sum()

    upset_front=clamp(logistic(-1.1+.18*front_count+.12*(10-pressure)+.10*d["斤量"].le(55).sum()),.05,.85)

    w=pd.DataFrame([
      ["S1","自然隊列・主導馬残存",clamp(pressure-1.8),clamp(.10+.018*closer_count+.010*pressure,0,1),.78,.78,.78,probs[0]],
      ["S2","前圧継続・好位中団接続",clamp(pressure+1.5),clamp(.30+.045*closer_count+.022*pressure,0,1),1.12,1.28,1.08,probs[1]],
      ["S3","反相関・構造穴（K6複合）",clamp(pressure+.4),clamp(.36+.050*closer_count+.025*uncertainty,0,1),1.65,1.18,1.48,probs[2]]],
      columns=["シナリオ","説明","前列圧","差し接続率","位置分散倍率","前列失速倍率","進路摩擦倍率","発生確率"])
    w["荒れ前残り発生率"]=upset_front
    w["構造荒れ指数"]=clamp(4.8+.35*uncertainty+.30*k6_density+.20*reversal)
    return d,w

def scenario_inputs(base,w):
    """
    S1/S2/S3で同じ能力式を流用せず、世界線ごとの4角到達構造を分離する。
    """
    out=[]
    for _,z in w.iterrows():
      sid=z["シナリオ"]
      for _,h in base.iterrows():
        early=h["初速"]; cruise=h["巡航"]; reacc=h["再加速"]; straight=h["直線余力"]
        conf=h["入力信頼度"]; inner=h["内枠優位"]; outer=h["外枠負荷"]
        front=h["逃げ番手率"]; light=clamp(5+(57-h["斤量"])*1.25)
        closer=clamp(straight-early+5)
        repro=clamp(10-h["出遅れ推定率"]*10-h["進路詰まり率"]*3)
        pressure_tol=clamp(.40*cruise+.30*reacc+.20*repro+.10*(10-outer))
        structural_f=clamp(.28*light+.22*inner+.20*early+.15*repro+.15*(10-front*5))
        structural_l=clamp(.30*straight+.24*reacc+.20*light+.14*(10-early)+.12*(10-inner))
        recv=clamp(z["前列圧"]*(.38+.70*front)+outer*.13+h["内包まれ率"]*2.2)

        if sid=="S1":
          early_ev=clamp(.34*early+.22*repro+.18*inner+.16*cruise+.10*pressure_tol)
          dc=clamp(z["差し接続率"]*(.55+.55*closer/10),0,1)
          reserve=clamp(.44*straight+.24*cruise+.16*reacc+.10*light+.06*repro-.16*recv)
          role=clamp(.45*early+.30*repro+.25*inner)
        elif sid=="S2":
          early_ev=clamp(.20*early+.25*pressure_tol+.20*cruise+.18*reacc+.17*closer-.22*recv)
          dc=clamp(z["差し接続率"]*(.72+1.10*closer/10),0,1)
          reserve=clamp(.34*straight+.25*reacc+.20*pressure_tol+.11*light+.10*cruise-.22*recv)
          role=clamp(.40*pressure_tol+.30*reacc+.30*straight)
        else:
          # S3は能力順の並べ替えではなく、軽斤量・内枠・差し接続・反相関複合
          early_ev=clamp(max(structural_f,structural_l)-.12*recv-.22*front*early)
          dc=clamp(z["差し接続率"]*(.82+1.25*closer/10),0,1)
          reserve=clamp(max(.30*straight+.25*reacc+.20*light+.15*(10-early)+.10*inner,
                            .28*straight+.20*cruise+.22*light+.18*inner+.12*repro)-.10*recv)
          role=clamp(.34*light+.24*straight+.20*inner+.22*(10-front*10))

        fail=comb(h["出遅れ推定率"],h["内包まれ率"]*z["進路摩擦倍率"],h["外回し率"]*.45,min(.75,recv/18))
        loss=clamp(h["ラスト失速率"]*z["前列失速倍率"]*(.78+.52*front),.03,.94)
        route=clamp(1-comb(h["進路詰まり率"]*z["進路摩擦倍率"],h["外回し率"]*.45,h["内包まれ率"]*.38),.05,.98)
        remain=clamp(reserve+.18*role-.22*loss*10)

        out.append({
          "馬番":h["馬番"],"馬名":h["馬名"],"シナリオ":sid,
          "シナリオ発生確率":z["発生確率"],"ハナ取得確率":h["ハナ取得確率基礎"],
          "先手参加率":clamp(.28*h["ハナ取得確率基礎"]+.72*logistic((early_ev-5.0)/1.05),0,1),
          "前列圧被害EV":recv,"序盤位置不発率":fail,"差し接続確率":dc,
          "残存エネルギーEV":remain,"進路実現確率":route,"ラスト失速率":loss,
          "位置分散倍率":z["位置分散倍率"],"初速":early_ev,"巡航":cruise,
          "再加速":reacc,"直線余力":straight,"入力信頼度":conf,
          "OADPシナリオ専用EV":role,"構造穴EV":max(structural_f,structural_l) if sid=="S3" else 0
        })
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
      status="NG: 分離不足" if pd.notna(corr) and corr>=.94 and overlap>=5 and diff<.55 else ("WARN" if pd.notna(corr) and corr>=.88 and overlap>=4 else "OK")
      rows.append({"比較":f"{a}-{b}","4角順位相関":corr,"上位5頭重複":overlap,"平均前列圧差":diff,"判定":status})
    return pd.DataFrame(rows)

st.set_page_config(page_title="OADPシナリオ分離シミュレーター",layout="wide")
st.title("OADP シナリオ分離型レースシミュレーター")
st.caption(f"Parser version: {APP_VERSION}")
st.caption(f"Version {APP_VERSION} / FACT・DERIVED・ESTIMATE分離 / 人気・オッズを物理計算に不使用")
with st.sidebar:
    trials=st.slider("各シナリオ試行数",300,10000,3000,300)
    seeds=st.slider("シード数",1,30,10)
tabs=st.tabs(["入力","基礎数値","シミュレーション"])
with tabs[0]:
    input_method = st.radio(
        "入力方法",
        ["地方競馬公式URL", "JRA出馬表テキスト"],
        horizontal=True,
    )

    race = {}
    horses = []
    warns = []
    source_note = ""

    if input_method == "地方競馬公式URL":
        race_url = st.text_input(
            "地方競馬公式 出馬表URL",
            placeholder="https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/DebaTable?...",
        )
        st.caption("STEP0作成アプリ v3.8 の地方競馬HTMLパーサーをそのまま利用します。")
        load_url = st.button("URLから出馬表データを取得", type="primary")

        if load_url:
            try:
                race_html = fetch_nar_html(race_url)
                nar_race = parse_nar_html(race_html)
                race, horses = nar_race_to_sim_input(nar_race)
                if not horses:
                    raise ValueError("公式HTMLは取得できましたが、出走馬を抽出できませんでした。出馬表公開前またはHTML構造変更の可能性があります。")
                st.session_state["pending_race"] = race
                st.session_state["pending_horses"] = horses
                st.session_state["pending_source"] = race_url
                st.success(f"地方競馬公式URLから {len(horses)}頭を取得しました。")
            except Exception as e:
                st.error(f"URL読込に失敗しました: {type(e).__name__}: {e}")

        race = st.session_state.get("pending_race", {})
        horses = st.session_state.get("pending_horses", [])
        source_note = st.session_state.get("pending_source", "")

    else:
        text = st.text_area("JRA出馬表テキストを貼り付け", height=420)
        f = st.file_uploader("またはTXT", type=["txt"])
        if f:
            text = f.getvalue().decode("utf-8", errors="replace")
        if text:
            race, horses, warns = parse_text(text)
            source_note = "JRA_TEXT"

    for w in warns:
        st.warning(w)

    if race:
        st.subheader("レース情報")
        st.json(race)
        if source_note:
            st.caption(f"入力元: {source_note}")

    if horses:
        st.write(f"抽出馬数: **{len(horses)}頭**")
        st.dataframe(
            pd.DataFrame([
                {k: v for k, v in h.items() if k != "runs"} | {"近走数": len(h.get("runs", []))}
                for h in horses
            ]),
            use_container_width=True,
        )
    elif race:
        st.error("馬が0頭のため、基礎数値は生成できません。")

    generate = st.button("基礎数値を生成", disabled=(len(horses) == 0))
    if generate:
        try:
            base = derive(race, horses)
            ok, msg = validate_base(base)
            if not ok:
                st.error(msg)
            else:
                st.session_state["race"] = race
                st.session_state["base"] = base
                for key in ("world", "inputs", "result", "audit"):
                    st.session_state.pop(key, None)
                st.success(f"{len(base)}頭の基礎数値を生成しました。")
        except Exception as e:
            st.error(f"基礎数値生成に失敗しました: {type(e).__name__}: {e}")

with tabs[1]:
    if "base" not in st.session_state: st.info("入力タブから生成してください。")
    else:
      st.session_state["base"]=st.data_editor(st.session_state["base"],use_container_width=True,num_rows="fixed")
      st.download_button("基礎CSV",st.session_state["base"].to_csv(index=False).encode("utf-8-sig"),"oadp_base.csv")
with tabs[2]:
    if "base" not in st.session_state: st.info("基礎数値がありません。")
    else:
      ok,msg=validate_base(st.session_state["base"])
      if not ok:
        st.error(msg)
      if st.button("S1/S2/S3生成・実行",type="primary",disabled=not ok):
        try:
          b,w=worlds(st.session_state["base"]); i=scenario_inputs(b,w); r=simulate(i,trials,seeds); a=audit(r,i)
          st.session_state.update(world=w,inputs=i,result=r,audit=a)
        except Exception as e:
          st.error(f"シミュレーション実行に失敗しました: {type(e).__name__}: {e}")
      if "world" in st.session_state:
        st.subheader("シナリオ世界")
        st.caption("OADP Ver.2.35：展開構造を先に生成し、S1/S2/S3で別の4角到達式を適用。人気・オッズは不使用。")
        st.dataframe(st.session_state["world"],use_container_width=True)
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
