
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

APP_VERSION="0.5.0-oadp-contract-consumption"
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
    """出馬表・近走事実から局所能力を分離して生成する。人気・オッズは不使用。"""
    n=max(1,len(horses)); td=float(race.get("distance",1800)); surf=race.get("surface","dirt")
    rows=[]
    for h in horses:
        first=[]; last=[]; ws=[]; ups=[]; fins=[]; front=[]; moves=[]; pos_var=[]
        for i,r in enumerate(h.get("runs",[])[:5]):
            fs=int(r.get("field_size") or n)
            sim=max(.35,1-abs((r.get("distance") or td)-td)/1000)
            if r.get("surface") and r.get("surface")!=surf: sim*=.45
            w=sim*math.exp(-.22*i)
            if r.get("positions"):
                a=nrank(r["positions"][0],fs); b=nrank(r["positions"][-1],fs)
                first.append(a); last.append(b); ws.append(w)
                moves.append(b-a)
                pos_var.append(float(np.std(r["positions"],ddof=0)) if len(r["positions"])>1 else 0.0)
                front.append(1 if r["positions"][0]<=max(3,math.ceil(fs*.25)) else 0)
            if r.get("up3f") is not None: ups.append(float(r["up3f"]))
            if r.get("finish") is not None: fins.append(nrank(r["finish"],fs))
        conf=clamp(10*(.55*min(len(h.get("runs",[])),5)/5+.45*min(len(ups),5)/5))
        ep=wmean(first,ws) if first else 5.0
        cp=wmean(last,ws) if last else ep
        close=clamp(10-(np.mean(ups)-34)*.95) if ups else 5.0
        fq=float(np.mean(fins)) if fins else 5.0
        inner=clamp(10*(1-(h["number"]-1)/max(1,n-1))); outer=10-inner
        load=float(h.get("load") or 57); relief=clamp(5+(57-load)*.9)
        delta=float(h.get("body_weight_delta") or 0); stable=clamp(10-abs(delta)*.22)
        move_ev=clamp(5+(np.average(moves,weights=ws[:len(moves)]) if moves else 0))
        pos_stability=clamp(10-(np.mean(pos_var)*1.15 if pos_var else 3.0))
        front_rate=float(np.mean(front)) if front else .25
        cruise=clamp(.42*cp+.25*fq+.20*stable+.13*pos_stability)
        reacc=clamp(.34*move_ev+.28*cp+.23*close+.15*relief)
        straight=clamp(.55*close+.25*fq+.20*relief)
        if h.get("debut"): conf=min(conf,2)
        miss=clamp((10-ep)*.055+(1-conf/10)*.18,.03,.70)
        trap=clamp((inner/10)*(.12+max(0,5-ep)*.025),.02,.48)
        wide=clamp((outer/10)*.42,.02,.52)
        block=clamp(.10+(1-conf/10)*.15+trap*.30,.04,.55)
        fatigue=clamp(4.8+max(0,td-1600)/500+(10-cruise)*.23-(relief-5)*.18)
        loss=clamp(logistic((fatigue-straight)*.45),.05,.85)

        # 能力は単一総合点にせず、区間別容量として保持する。
        start_cap=clamp(.62*ep+.20*front_rate*10+.18*pos_stability)
        cruise_cap=clamp(.58*cruise+.22*pos_stability+.20*stable)
        position_cap=clamp(.48*cp+.30*pos_stability+.22*ep)
        pressure_cap=clamp(.42*cruise+.28*reacc+.18*pos_stability+.12*(10-outer))
        corner_cap=clamp(.52*reacc+.28*move_ev+.20*cp)
        recovery_cap=clamp(.46*cruise+.30*straight+.24*stable)
        straight_cap=straight
        vmax_cap=clamp(.62*close+.23*reacc+.15*fq)

        rows.append({
            "馬番":h["number"],"馬名":h["name"],"枠":h["frame"],"斤量":load,
            "馬体重":h.get("body_weight"),"増減":h.get("body_weight_delta"),
            "初出走":h.get("debut",False),"有効近走数":len(h.get("runs",[])),
            "入力信頼度":round(conf,3),"初速":round(ep,3),"巡航":round(cruise,3),
            "再加速":round(reacc,3),"直線余力":round(straight,3),
            "内枠優位":round(inner,3),"外枠負荷":round(outer,3),
            "逃げ番手率":round(front_rate,4),"出遅れ推定率":round(miss,4),
            "内包まれ率":round(trap,4),"外回し率":round(wide,4),
            "進路詰まり率":round(block,4),"疲労EV":round(fatigue,3),
            "ラスト失速率":round(loss,4),
            "START_CAPACITY":round(start_cap,3),
            "CRUISE_CAPACITY":round(cruise_cap,3),
            "POSITION_CAPACITY":round(position_cap,3),
            "PRESSURE_CAPACITY":round(pressure_cap,3),
            "CORNER_CAPACITY":round(corner_cap,3),
            "RECOVERY_CAPACITY":round(recovery_cap,3),
            "STRAIGHT_CAPACITY":round(straight_cap,3),
            "VMAX_CAPACITY":round(vmax_cap,3),
            "位置上昇EV":round(move_ev,3),"位置安定EV":round(pos_stability,3),
            "value_type":"DERIVED/ESTIMATE"
        })
    return pd.DataFrame(rows).sort_values("馬番").reset_index(drop=True)

def worlds(base):
    """馬群構成だけから分岐世界を作る。人気・オッズ・総合能力は使わない。"""
    d=base.copy(); n=len(d)
    claim_raw=np.exp(
        .48*d["START_CAPACITY"]/10+
        .25*d["逃げ番手率"]+
        .17*d["内枠優位"]/10-
        .10*d["出遅れ推定率"]
    )
    d["ハナ取得確率基礎"]=claim_raw/claim_raw.sum()
    d["先行参加確率基礎"]=np.clip(
        .18+.55*d["逃げ番手率"]+.20*d["START_CAPACITY"]/10-.15*d["出遅れ推定率"],.05,.95
    )
    front_count=int((d["先行参加確率基礎"]>=.48).sum())
    strong_claim=int((d["ハナ取得確率基礎"]>=max(.12,1/max(1,n)*1.6)).sum())
    weak_front=int(((d["先行参加確率基礎"]>=.42)&(d["PRESSURE_CAPACITY"]<=5.4)).sum())
    closer_count=int((d["STRAIGHT_CAPACITY"]>=d["START_CAPACITY"]+.5).sum())
    uncertainty=clamp(10-d["入力信頼度"].mean()+d["START_CAPACITY"].std(ddof=0))
    pressure=clamp(1.3+front_count*.95+strong_claim*.85+weak_front*.55+d["先行参加確率基礎"].sum()*.22)
    control=clamp(d.nlargest(min(2,n),"ハナ取得確率基礎")["ハナ取得確率基礎"].sum()*10)
    k6d=clamp(2.0+max(0,strong_claim-1)*1.2+uncertainty*.38)
    k6e=clamp(1.5+d["斤量"].le(55).sum()*.55+d["内枠優位"].ge(7).sum()*.42)
    k6f=clamp(1.2+closer_count*.62+(d["位置上昇EV"]>=6).sum()*.55)

    raw=np.array([
        max(.1,8.0+.75*control-.63*pressure-.28*uncertainty),
        max(.1,4.8+.86*pressure+.38*weak_front+.42*closer_count),
        max(.1,3.8+.52*k6d+.48*k6e+.50*k6f+.34*uncertainty),
    ])
    probs=raw/raw.sum(); probs=np.maximum(probs,.12); probs=probs/probs.sum()
    s3parts=np.array([pressure+weak_front, k6d+k6e, k6f+closer_count],float)
    s3parts=s3parts/s3parts.sum()

    w=pd.DataFrame([
        ["S1","自然隊列・主逃げ成立",probs[0],"S1",.78,.25,.25,.20],
        ["S2","前列圧継続・複数回負荷",probs[1],"S2",1.35,.70,.65,.45],
        ["S3","構造穴分岐",probs[2],"S3",1.05,s3parts[0],s3parts[1],s3parts[2]],
    ],columns=["シナリオ","説明","発生確率","分岐型","圧力倍率","S3-Fa比率","S3-Fb比率","S3-L比率"])
    w["前列圧指数"]=pressure
    w["K6D指数"]=k6d; w["K6E指数"]=k6e; w["K6F指数"]=k6f
    return d,w

def scenario_inputs(base,w):
    """イベントエンジンへ渡す局所能力と条件付き確率を作る。"""
    out=[]
    for _,z in w.iterrows():
        sid=z["シナリオ"]
        for _,h in base.iterrows():
            conf=h["入力信頼度"]/10
            light=clamp(5+(57-h["斤量"])*1.2)
            claim=clamp(.48*h["START_CAPACITY"]+.25*h["逃げ番手率"]*10+.17*h["内枠優位"]+.10*light)
            resist=clamp(.40*h["PRESSURE_CAPACITY"]+.30*h["POSITION_CAPACITY"]+.20*h["CORNER_CAPACITY"]+.10*conf*10)
            move=clamp(.48*h["CORNER_CAPACITY"]+.28*h["位置上昇EV"]+.24*h["CRUISE_CAPACITY"])
            close_conn=clamp(.45*h["CORNER_CAPACITY"]+.35*h["STRAIGHT_CAPACITY"]+.20*h["VMAX_CAPACITY"])
            uncertainty=clamp(1-conf,.05,.85,)

            out.append({
                "馬番":h["馬番"],"馬名":h["馬名"],"シナリオ":sid,
                "シナリオ発生確率":z["発生確率"],
                "ハナ取得確率":h["ハナ取得確率基礎"],
                "先行参加率":h["先行参加確率基礎"],
                "主張EV":claim,"抵抗EV":resist,"3角進出EV":move,"差し接続EV":close_conn,
                "START_CAPACITY":h["START_CAPACITY"],"CRUISE_CAPACITY":h["CRUISE_CAPACITY"],
                "POSITION_CAPACITY":h["POSITION_CAPACITY"],"PRESSURE_CAPACITY":h["PRESSURE_CAPACITY"],
                "CORNER_CAPACITY":h["CORNER_CAPACITY"],"RECOVERY_CAPACITY":h["RECOVERY_CAPACITY"],
                "STRAIGHT_CAPACITY":h["STRAIGHT_CAPACITY"],"VMAX_CAPACITY":h["VMAX_CAPACITY"],
                "出遅れ推定率":h["出遅れ推定率"],"内包まれ率":h["内包まれ率"],
                "外回し率":h["外回し率"],"進路詰まり率":h["進路詰まり率"],
                "ラスト失速率":h["ラスト失速率"],"内枠優位":h["内枠優位"],
                "外枠負荷":h["外枠負荷"],"斤量":h["斤量"],"入力信頼度":h["入力信頼度"],
                "圧力倍率":z["圧力倍率"],"S3-Fa比率":z["S3-Fa比率"],
                "S3-Fb比率":z["S3-Fb比率"],"S3-L比率":z["S3-L比率"],
                "value_type":"DERIVED/SCENARIO_ESTIMATE"
            })
    return pd.DataFrame(out)

SEGMENTS=("START","EARLY_POSITION","FIRST_CORNER","MID_CRUISE","THIRD_CORNER","FOURTH_CORNER_EXIT")

def _choose_s3_branch(row,rng):
    p=np.array([row["S3-Fa比率"],row["S3-Fb比率"],row["S3-L比率"]],float)
    p=p/p.sum()
    return rng.choice(["S3-Fa","S3-Fb","S3-L"],p=p)

def _rank_from_position(position, rng, jitter=.015):
    """連続位置から重複のない順位を作る。小さい値ほど前。"""
    score=np.asarray(position,float)+rng.normal(0,jitter,len(position))
    order=np.argsort(score,kind="mergesort")
    rank=np.empty(len(position),int); rank[order]=np.arange(1,len(position)+1)
    return rank

def _weighted_choice(rng, weights):
    w=np.asarray(weights,float)
    w=np.clip(w,1e-9,None); w=w/w.sum()
    return int(rng.choice(np.arange(len(w)),p=w))

def simulate(inp,trials,seeds):
    """イベント型Monte Carlo v0.4.1。
    重要: シナリオ差を消耗係数だけでなく、各区間の連続位置 state へ直接反映する。
    """
    summaries=[]; event_rows=[]; branch_rows=[]
    for sid in SCENARIOS:
        d=inp[inp["シナリオ"]==sid].reset_index(drop=True)
        n=len(d); per=max(20,trials//max(1,seeds)); total=per*seeds
        horse_acc={int(r["馬番"]):{"corner":[],"finish":[],"energy":[],"pressure":[],"moves":[],"blocked":[],"faded":[],"lead":0,
                                    "zone_front":0,"zone_stalk":0,"zone_mid":0,"zone_back":0} for _,r in d.iterrows()}
        event_counts={}; branch_counts={}
        for seed in range(seeds):
            rng=np.random.default_rng(20260726+seed*1009+SCENARIOS.index(sid)*100003)
            for _ in range(per):
                branch=sid if sid!="S3" else _choose_s3_branch(d.iloc[0],rng)
                branch_counts[branch]=branch_counts.get(branch,0)+1

                start_cap=d["START_CAPACITY"].to_numpy(float)
                cruise_cap=d["CRUISE_CAPACITY"].to_numpy(float)
                pos_cap=d["POSITION_CAPACITY"].to_numpy(float)
                press_cap=d["PRESSURE_CAPACITY"].to_numpy(float)
                corner_cap=d["CORNER_CAPACITY"].to_numpy(float)
                recovery_cap=d["RECOVERY_CAPACITY"].to_numpy(float)
                straight_cap=d["STRAIGHT_CAPACITY"].to_numpy(float)
                vmax_cap=d["VMAX_CAPACITY"].to_numpy(float)
                conf=d["入力信頼度"].to_numpy(float)/10
                noise=.28+(1-conf)*.65
                inner=d["内枠優位"].to_numpy(float)
                light=np.clip(5+(57-d["斤量"].to_numpy(float))*1.2,0,10)

                energy=np.ones(n)*100.0
                pressure=np.zeros(n)
                move_count=np.zeros(n,int)

                # START: 発馬・先行参加・主張を抽選
                miss=rng.random(n)<d["出遅れ推定率"].to_numpy(float)
                start_perf=start_cap+rng.normal(0,noise)-miss*rng.uniform(.9,2.3,n)
                participants=rng.random(n)<d["先行参加率"].to_numpy(float)
                if not participants.any():
                    participants[np.argmax(start_perf)]=True

                claim=np.exp((start_perf-5)/1.8) * (0.35+0.65*d["ハナ取得確率"].to_numpy(float)*n)
                claim=np.where(participants,claim,1e-9)

                # S3-Fbは「本来競る馬が控え、別の前受け馬が主導」を位置 state で作る
                if branch=="S3-Fb":
                    strong=np.argsort(-claim)[:max(1,min(2,n))]
                    participants[strong]=False
                    alt_score=(.34*start_cap+.28*inner+.22*light+.16*pos_cap
                               -1.0*(d["ハナ取得確率"].to_numpy(float)*10))
                    eligible=np.where(~miss)[0]
                    if len(eligible):
                        leader=int(eligible[np.argmax(alt_score[eligible]+rng.normal(0,.35,len(eligible)))])
                    else:
                        leader=int(np.argmax(alt_score))
                    participants[leader]=True
                else:
                    leader=_weighted_choice(rng,claim)

                # 連続位置。旧版の rank を使い回さず、イベントごとに更新する。
                base_order=np.argsort(-start_perf)
                pos=np.empty(n,float); pos[base_order]=np.arange(1,n+1,dtype=float)
                pos-=participants*rng.uniform(.35,1.05,n)
                pos[miss]+=rng.uniform(1.0,2.4,miss.sum())
                pos[leader]=0.35

                energy-=2.0+np.maximum(0,7-start_cap)*.20
                energy[participants]-=rng.uniform(.5,1.4,participants.sum())
                event_counts[("START","MISS")]=event_counts.get(("START","MISS"),0)+int(miss.sum())
                event_counts[("START","CLAIM")]=event_counts.get(("START","CLAIM"),0)+int(participants.sum())

                # EARLY_POSITION: 競争の成否が位置そのものを変える
                contenders=[i for i in np.where(participants)[0] if i!=leader]
                if sid=="S1": attack_p=.14
                elif sid=="S2": attack_p=.78
                elif branch=="S3-Fa": attack_p=.84
                elif branch=="S3-Fb": attack_p=.06
                else: attack_p=.58

                for i in contenders:
                    if rng.random()<attack_p:
                        intensity=rng.uniform(.75,1.55)*float(d.iloc[i]["圧力倍率"])
                        pressure[leader]+=intensity
                        pressure[i]+=intensity*.76
                        energy[leader]-=(1.0+max(0,intensity-press_cap[leader]/7)**2)
                        energy[i]-=(.85+max(0,intensity-press_cap[i]/7)**2)
                        resist_p=logistic((d.iloc[leader]["抵抗EV"]-d.iloc[i]["主張EV"])/1.15)
                        resisted=rng.random()<resist_p
                        event_counts[("EARLY_POSITION","OUTSIDE_PRESS")]=event_counts.get(("EARLY_POSITION","OUTSIDE_PRESS"),0)+1
                        if resisted:
                            pos[leader]=min(pos[leader],.40)
                            pos[i]=min(pos[i],1.10+rng.uniform(0,.55))
                        else:
                            old=leader; leader=i
                            pos[leader]=.35
                            pos[old]=1.15+rng.uniform(0,.70)
                            event_counts[("EARLY_POSITION","LEAD_CHANGE")]=event_counts.get(("EARLY_POSITION","LEAD_CHANGE"),0)+1
                    else:
                        # 控える馬は番手〜好位へ落ち着く
                        pos[i]=max(pos[i],1.2+rng.uniform(0,2.0))

                if sid=="S1":
                    # 自然隊列: 主導馬を固定し、競合は隊列化
                    pos[leader]=.25
                    for j,i in enumerate(sorted(contenders,key=lambda x:pos[x])):
                        pos[i]=max(pos[i],1.20+j*.75+rng.uniform(0,.25))
                elif sid=="S2" or branch=="S3-Fa":
                    # 横広がり: 複数馬が前列へ残りやすい
                    front_candidates=np.where(participants)[0]
                    pos[front_candidates]=np.minimum(pos[front_candidates],
                        rng.uniform(.35,max(1.25,1.25+.18*len(front_candidates)),len(front_candidates)))

                # FIRST_CORNER
                trapped=rng.random(n)<d["内包まれ率"].to_numpy(float)*(0.75 if branch=="S3-L" else 1.0)
                wide=rng.random(n)<d["外回し率"].to_numpy(float)
                pos+=trapped*rng.uniform(.25,.85,n)
                pos+=wide*rng.uniform(.08,.40,n)
                energy-=wide*rng.uniform(.4,1.1,n)
                event_counts[("FIRST_CORNER","TRAPPED")]=event_counts.get(("FIRST_CORNER","TRAPPED"),0)+int(trapped.sum())
                event_counts[("FIRST_CORNER","WIDE")]=event_counts.get(("FIRST_CORNER","WIDE"),0)+int(wide.sum())

                # MID_CRUISE: 現在位置から前列を再判定（旧版はSTART時rank固定だった）
                rank_mid=_rank_from_position(pos,rng)
                front_zone=rank_mid<=max(3,math.ceil(n*.30))
                pace_load=(1.05 if sid=="S1" else 1.85 if sid=="S2" else
                           1.75 if branch=="S3-Fa" else .82 if branch=="S3-Fb" else 1.78)
                demand=pace_load+front_zone*.7+pressure*.30
                overload=np.maximum(0,demand-cruise_cap/5.2)
                energy-=1.4+demand*.55+overload**2
                recover=(~front_zone)*recovery_cap/10*rng.uniform(.5,1.2,n)
                if branch=="S3-Fb":
                    recover+=front_zone*recovery_cap/10*.72
                    # 縦隊列を強化
                    pos=np.sort(pos)[rank_mid-1] + (rank_mid-1)*.05
                energy+=recover
                event_counts[("MID_CRUISE","RECOVERY")]=event_counts.get(("MID_CRUISE","RECOVERY"),0)+int((recover>.5).sum())

                # THIRD_CORNER: 後方接続/前残りを位置へ直接反映
                rank3=_rank_from_position(pos,rng)
                if branch=="S3-Fb":
                    move_p=.08+.18*d["3角進出EV"].to_numpy(float)/10
                elif branch=="S3-L":
                    move_p=.25+.62*d["差し接続EV"].to_numpy(float)/10
                elif sid=="S2" or branch=="S3-Fa":
                    move_p=.18+.46*d["3角進出EV"].to_numpy(float)/10
                else:
                    move_p=.13+.32*d["3角進出EV"].to_numpy(float)/10
                move_p=np.clip(move_p*(.60+energy/135),.02,.94)
                if branch=="S3-L":
                    move_p=np.clip(move_p + (rank3>max(3,n*.35))*.16,.02,.96)
                movers=rng.random(n)<move_p
                move_strength=corner_cap+rng.normal(0,noise)-np.maximum(0,55-energy)*.06
                shift=np.where(movers,np.clip((move_strength-4.2)/1.55,0,3.5),0)
                if branch=="S3-L":
                    shift*=np.where(rank3>max(3,n*.35),1.35,.70)
                if branch=="S3-Fb":
                    shift*=.55
                move_count+=np.round(shift).astype(int)
                pos-=shift
                energy-=movers*(.7+shift*.62+np.maximum(0,6-corner_cap)*.20)
                event_counts[("THIRD_CORNER","MOVE")]=event_counts.get(("THIRD_CORNER","MOVE"),0)+int(movers.sum())

                # FOURTH_CORNER_EXIT
                rank_pre4=_rank_from_position(pos,rng)
                front_now=rank_pre4<=max(3,math.ceil(n*.30))
                velocity=.34*cruise_cap+.42*corner_cap+.24*pos_cap-.34*pressure-.060*np.maximum(0,60-energy)
                velocity[leader]+=1.15
                if branch=="S3-L":
                    velocity+=.36*straight_cap-.22*start_cap
                    velocity+=np.where(~front_now,.38,-.28)
                elif branch=="S3-Fb":
                    velocity+=np.where(front_now,.92,-.18)
                elif sid=="S2" or branch=="S3-Fa":
                    velocity+=np.where(front_now,-.22,.20)

                # 位置を主、速度を補助にする。旧版は stale rank と能力scoreが支配していた。
                corner_metric=pos-.24*velocity+rng.normal(0,noise*.16)
                corner_rank=_rank_from_position(corner_metric,rng,jitter=.01)
                corner_order=np.argsort(corner_rank)
                leader4=int(corner_order[0])

                blocked=rng.random(n)<np.clip(d["進路詰まり率"].to_numpy(float)+trapped*.12+wide*.04,.02,.80)
                fade_p=np.clip(d["ラスト失速率"].to_numpy(float)+np.maximum(0,55-energy)/100+pressure*.025,.03,.95)
                faded=rng.random(n)<fade_p
                effective_straight=(.54*straight_cap+.24*vmax_cap+.22*corner_cap)*(energy/100)
                final_score=-.34*corner_rank+.66*effective_straight-blocked*rng.uniform(.5,1.7,n)-faded*rng.uniform(.7,2.2,n)+rng.normal(0,noise*.35)
                finish_order=np.argsort(-final_score); finish_rank=np.empty(n,int); finish_rank[finish_order]=np.arange(1,n+1)

                for i,row in d.iterrows():
                    k=int(row["馬番"]); a=horse_acc[k]
                    cr=int(corner_rank[i])
                    a["corner"].append(cr); a["finish"].append(int(finish_rank[i]))
                    a["energy"].append(float(energy[i])); a["pressure"].append(float(pressure[i]))
                    a["moves"].append(int(move_count[i])); a["blocked"].append(int(blocked[i])); a["faded"].append(int(faded[i]))
                    if cr==1: a["lead"]+=1
                    if cr<=max(2,math.ceil(n*.15)): a["zone_front"]+=1
                    elif cr<=max(4,math.ceil(n*.35)): a["zone_stalk"]+=1
                    elif cr<=max(6,math.ceil(n*.70)): a["zone_mid"]+=1
                    else: a["zone_back"]+=1

        for _,row in d.iterrows():
            k=int(row["馬番"]); a=horse_acc[k]
            zone_rates=np.array([a["zone_front"],a["zone_stalk"],a["zone_mid"],a["zone_back"]],float)/total
            zone_names=["先頭圏","好位圏","中団圏","後方圏"]
            summaries.append({
                "シナリオ":sid,"馬番":k,"馬名":row["馬名"],
                "4角順位平均":np.mean(a["corner"]),"4角順位SD":np.std(a["corner"]),
                "4角順位中央値":np.median(a["corner"]),
                "4角最頻位置帯":zone_names[int(np.argmax(zone_rates))],
                "4角先頭圏率":zone_rates[0],"4角好位圏率":zone_rates[1],
                "4角中団圏率":zone_rates[2],"4角後方圏率":zone_rates[3],
                "4角先頭率":a["lead"]/total,"4角3位内率":np.mean(np.array(a["corner"])<=3),
                "4角残存エネルギー平均":np.mean(a["energy"]),"累積圧力平均":np.mean(a["pressure"]),
                "3角進出回数平均":np.mean(a["moves"]),
                "勝率":np.mean(np.array(a["finish"])==1),"複勝率":np.mean(np.array(a["finish"])<=3),
                "平均着順":np.mean(a["finish"]),"実測進路不発率":np.mean(a["blocked"]),
                "実測失速率":np.mean(a["faded"])
            })
        for (segment,event),count in event_counts.items():
            event_rows.append({"シナリオ":sid,"区間":segment,"イベント":event,"1試行平均回数":count/total})
        for branch,count in branch_counts.items():
            branch_rows.append({"シナリオ":sid,"内部分岐":branch,"発生率":count/total})
    return pd.DataFrame(summaries),pd.DataFrame(event_rows),pd.DataFrame(branch_rows)


def _extract_numbers(s):
    return [int(x) for x in re.findall(r"\d{1,2}", str(s))]

def parse_oadp_phase_contract(text, base):
    """Phase0〜Phase3テキストから、S1/S2/S3の4角拘束と発生確率を抽出する。
    対応対象は本アプリ出力フォーマット:
      【シナリオ1...】/ 発生確率
      4角出口〜直線入口：
      先頭圏：...
      番手圏：...
      好位圏：...
      中団接続圏：...
      消耗圏：...
      後方圏：...
    """
    if not text or not str(text).strip():
        raise ValueError("OADP Phase0〜3テキストが空です。")
    b=base.copy()
    valid=set(pd.to_numeric(b["馬番"],errors="coerce").dropna().astype(int))
    name_map={int(r["馬番"]):str(r["馬名"]) for _,r in b.iterrows()}
    scen_map={"1":"S1","2":"S2","3":"S3"}
    starts=[]
    pat=re.compile(r"【シナリオ\s*([123])[^】]*】")
    for m in pat.finditer(text):
        starts.append((m.start(),scen_map[m.group(1)]))
    if len(starts)<3:
        # Phase1の通常見出しも許容
        pat2=re.compile(r"【シナリオ\s*([123])[:：]")
        starts=[]
        for m in pat2.finditer(text):
            starts.append((m.start(),scen_map[m.group(1)]))
    if len(starts)<3:
        raise ValueError("S1/S2/S3のシナリオ見出しを3件抽出できませんでした。")

    zone_rank={"先頭圏":1.0,"番手圏":2.5,"好位圏":5.0,"中団接続圏":7.0,
               "中団圏":7.0,"展開補助必要圏":8.5,"消耗圏":9.0,
               "後方流入圏":9.0,"後方圏":10.5}
    rows=[]; meta=[]
    for idx,(stpos,sid) in enumerate(starts[:3]):
        en=starts[idx+1][0] if idx+1<len(starts) else len(text)
        block=text[stpos:en]
        pm=re.search(r"発生確率[:：]\s*(\d+(?:\.\d+)?)\s*%",block)
        prob=float(pm.group(1))/100 if pm else 1/3
        if "競り継続" in block or "前列圧型" in block:
            mode="HIGH_PRESSURE"
        elif "無風主導" in block or "圧不発" in block:
            mode="LOW_PRESSURE_FRONT"
        else:
            mode="NATURAL"
        target={}
        zlabel={}
        # 4角出口以降だけを優先
        q=block.find("4角出口")
        tail=block[q:] if q>=0 else block
        for z,base_rank in zone_rank.items():
            mm=re.search(rf"{re.escape(z)}[:：]\s*([^\n\r]+)",tail)
            if not mm: continue
            nums=[n for n in _extract_numbers(mm.group(1)) if n in valid]
            for j,n in enumerate(nums):
                target[n]=base_rank+j*0.35
                zlabel[n]=z
        # 1行隊列表記へのフォールバック
        if not target:
            mm=re.search(r"4角出口[^：]*[:：]\s*\n?\s*([0-9,、 ＞>]+)",block)
            if mm:
                groups=re.split(r"[＞>]",mm.group(1))
                r=1.0
                for g in groups:
                    nums=[n for n in _extract_numbers(g) if n in valid]
                    for j,n in enumerate(nums):
                        target[n]=r+j*.25; zlabel[n]="隊列指定"
                    r+=max(1.5,len(nums)*.55)
        if not target:
            raise ValueError(f"{sid}の4角位置指定を抽出できませんでした。")
        # 未記載馬は後方へ、ただし元データ順で安定化
        missing=sorted(valid-set(target))
        rear=max(target.values())+1.0
        for j,n in enumerate(missing):
            target[n]=rear+j*.45; zlabel[n]="未記載後方補完"
        meta.append({"シナリオ":sid,"発生確率":prob,"圧力モード":mode,
                     "抽出頭数":len(target),"原文先頭":block[:180].replace("\n"," ")})
        for n in sorted(valid):
            rows.append({"シナリオ":sid,"馬番":n,"馬名":name_map.get(n,""),
                         "OADP目標4角位置":target[n],"OADP位置帯":zlabel[n],
                         "シナリオ発生確率":prob,"圧力モード":mode})
    c=pd.DataFrame(rows)
    # 確率を正規化
    probs=pd.DataFrame(meta)
    s=probs["発生確率"].sum()
    if s>0:
        probs["発生確率"]/=s
        pmap=dict(zip(probs["シナリオ"],probs["発生確率"]))
        c["シナリオ発生確率"]=c["シナリオ"].map(pmap)
    return c,probs

def simulate_from_oadp_contract(base, contract, trials=3000, seeds=10, anchor_strength=.88):
    """AI/OADPが決めた4角構造を上位契約として固定し、その内部で消耗・圧力・
    位置揺らぎをMonte Carloする。人気・オッズは使用しない。
    """
    base=base.copy()
    results=[]; components=[]; audits=[]
    n=len(base)
    per=max(20,int(trials)//max(1,int(seeds)))
    total=per*max(1,int(seeds))
    for sid in SCENARIOS:
        d=base.merge(contract[contract["シナリオ"]==sid],on=["馬番","馬名"],how="inner")
        if len(d)!=len(base):
            raise ValueError(f"{sid}: 基礎データとOADP契約の馬番が一致しません。")
        acc={int(r["馬番"]):{"rank":[],"energy":[],"cons":[],"pressure":[],
             "start":[],"early":[],"corner1":[],"cruise":[],"third":[],"fourth":[]} for _,r in d.iterrows()}
        mode=str(d.iloc[0]["圧力モード"])
        for seed in range(max(1,int(seeds))):
            rng=np.random.default_rng(20260727+seed*1013+SCENARIOS.index(sid)*100019)
            for _ in range(per):
                # OADP順位を主状態とし、信頼度・能力に応じた局所揺らぎだけ許す
                conf=d["入力信頼度"].to_numpy(float)/10
                target=d["OADP目標4角位置"].to_numpy(float)
                start=d["START_CAPACITY"].to_numpy(float)
                cruise=d["CRUISE_CAPACITY"].to_numpy(float)
                presscap=d["PRESSURE_CAPACITY"].to_numpy(float)
                corner=d["CORNER_CAPACITY"].to_numpy(float)
                recovery=d["RECOVERY_CAPACITY"].to_numpy(float)
                outer=d["外枠負荷"].to_numpy(float)
                trapped_p=d["内包まれ率"].to_numpy(float)
                wide_p=d["外回し率"].to_numpy(float)

                front=np.clip(1-(target-1)/max(1,n-1),0,1)
                uncertainty=(1-conf)*1.25+(1-anchor_strength)*1.8
                metric=target+rng.normal(0,uncertainty)
                # 局所能力は順位を作り直さず、契約周辺の微小補正だけ
                metric-= (corner-5)*.075+(d["POSITION_CAPACITY"].to_numpy(float)-5)*.045
                order=np.argsort(metric,kind="mergesort")
                rank=np.empty(n,int); rank[order]=np.arange(1,n+1)

                if mode=="HIGH_PRESSURE":
                    press_level=1.25+2.15*front
                elif mode=="LOW_PRESSURE_FRONT":
                    press_level=.45+.55*front
                else:
                    press_level=.75+1.05*front
                pressure=np.maximum(0,rng.normal(press_level,.28))
                pressure*=np.clip(1.12-(presscap-5)*.035,.72,1.30)

                # 区間別消耗。100は能力同一ではなく、残存率の基準。
                c_start=2.0+np.maximum(0,6.5-start)*.22+rng.uniform(.25,.75,n)
                c_early=.8+front*(1.05 if mode=="NATURAL" else 2.15 if mode=="HIGH_PRESSURE" else .55)
                c_early+=pressure*.55
                trapped=rng.random(n)<trapped_p
                wide=rng.random(n)<wide_p
                c_corner1=.45+trapped*rng.uniform(.35,1.0,n)+wide*rng.uniform(.35,1.15,n)+outer*.025
                demand=(1.0 if mode=="NATURAL" else 1.85 if mode=="HIGH_PRESSURE" else .70)+front*.75+pressure*.28
                overload=np.maximum(0,demand-cruise/5.4)
                c_cruise=1.55+demand*.62+overload**2
                rec=(1-front)*recovery/10*rng.uniform(.25,.85,n)
                if mode=="LOW_PRESSURE_FRONT":
                    rec+=front*recovery/10*rng.uniform(.35,.75,n)
                c_third=.65+np.maximum(0,(6-corner))*.18
                c_third+=np.where(rank<target,.65,.25)+rng.uniform(.15,.55,n)
                c_fourth=.70+pressure*.18+np.maximum(0,5.5-corner)*.15
                total_cons=np.maximum(0,c_start+c_early+c_corner1+c_cruise+c_third+c_fourth-rec)
                # 前列高圧時の非線形二段負荷
                if mode=="HIGH_PRESSURE":
                    total_cons+=np.maximum(0,pressure-1.5)**2*.48
                energy=np.clip(100-total_cons,0,100)
                for i,r in d.iterrows():
                    a=acc[int(r["馬番"])]
                    for k,v in [("rank",rank[i]),("energy",energy[i]),("cons",total_cons[i]),
                                ("pressure",pressure[i]),("start",c_start[i]),("early",c_early[i]),
                                ("corner1",c_corner1[i]),("cruise",c_cruise[i]),
                                ("third",c_third[i]),("fourth",c_fourth[i])]:
                        a[k].append(float(v))
        for _,r in d.iterrows():
            no=int(r["馬番"]); a=acc[no]
            results.append({"シナリオ":sid,"馬番":no,"馬名":r["馬名"],
                "OADP位置帯":r["OADP位置帯"],"OADP目標4角位置":r["OADP目標4角位置"],
                "4角順位平均":np.mean(a["rank"]),"4角順位SD":np.std(a["rank"]),
                "4角先頭率":np.mean(np.array(a["rank"])==1),
                "4角3位内率":np.mean(np.array(a["rank"])<=3),
                "残存エネルギー平均":np.mean(a["energy"]),
                "残存エネルギーP10":np.percentile(a["energy"],10),
                "残存エネルギーP90":np.percentile(a["energy"],90),
                "消耗率平均":np.mean(a["cons"]),
                "消耗率SD":np.std(a["cons"]),
                "累積圧力平均":np.mean(a["pressure"]),
                "シナリオ発生確率":r["シナリオ発生確率"],
                "圧力モード":r["圧力モード"]})
            components.append({"シナリオ":sid,"馬番":no,"馬名":r["馬名"],
                "発馬消耗":np.mean(a["start"]),"序盤位置取り消耗":np.mean(a["early"]),
                "1角消耗":np.mean(a["corner1"]),"巡航消耗":np.mean(a["cruise"]),
                "3角進出消耗":np.mean(a["third"]),"4角消耗":np.mean(a["fourth"]),
                "総消耗率":np.mean(a["cons"])})
        rr=pd.DataFrame([x for x in results if x["シナリオ"]==sid])
        target_order=list(rr.sort_values("OADP目標4角位置")["馬番"])
        sim_order=list(rr.sort_values("4角順位平均")["馬番"])
        audits.append({"シナリオ":sid,"OADP先頭馬":target_order[0],"計算先頭馬":sim_order[0],
                       "先頭一致":target_order[0]==sim_order[0],
                       "順位相関":rr["OADP目標4角位置"].corr(rr["4角順位平均"]),
                       "判定":"OK" if target_order[0]==sim_order[0] and rr["OADP目標4角位置"].corr(rr["4角順位平均"])>=.90 else "NG"})
    res=pd.DataFrame(results); comp=pd.DataFrame(components); aud=pd.DataFrame(audits)
    # 発生確率加重の統合消耗
    weighted=(res.assign(_w=lambda x:x["シナリオ発生確率"])
              .groupby(["馬番","馬名"],as_index=False)
              .apply(lambda g:pd.Series({
                  "統合消耗率":np.average(g["消耗率平均"],weights=g["_w"]),
                  "統合残存エネルギー":np.average(g["残存エネルギー平均"],weights=g["_w"]),
                  "統合4角順位":np.average(g["4角順位平均"],weights=g["_w"])
              }),include_groups=False).reset_index(drop=True))
    return res,comp,aud,weighted

def audit(res,inp,event_log,branches):
    rows=[]
    for a,b in [("S1","S2"),("S1","S3"),("S2","S3")]:
        x=res[res["シナリオ"]==a].set_index("馬番"); y=res[res["シナリオ"]==b].set_index("馬番")
        c=x.index.intersection(y.index)
        corr=x.loc[c,"4角順位平均"].corr(y.loc[c,"4角順位平均"]) if len(c)>=3 else np.nan
        lead_a=x["4角先頭率"].idxmax() if len(x) else None
        lead_b=y["4角先頭率"].idxmax() if len(y) else None
        ea=event_log[event_log["シナリオ"]==a].set_index(["区間","イベント"])["1試行平均回数"]
        eb=event_log[event_log["シナリオ"]==b].set_index(["区間","イベント"])["1試行平均回数"]
        idx=ea.index.union(eb.index); va=ea.reindex(idx,fill_value=0); vb=eb.reindex(idx,fill_value=0)
        process_diff=float(np.abs(va-vb).mean()) if len(idx) else 0
        same_leader=lead_a==lead_b
        order_a=list(x.sort_values("4角順位平均").index)
        order_b=list(y.sort_values("4角順位平均").index)
        exact_same_order=(order_a==order_b)
        mean_abs_gap=float(np.mean(np.abs(x.loc[c,"4角順位平均"]-y.loc[c,"4角順位平均"]))) if len(c) else np.nan
        if exact_same_order and pd.notna(mean_abs_gap) and mean_abs_gap<.12:
            status="NG: 4角位置が実質同一"
        elif pd.notna(corr) and corr>=.96 and process_diff<.18:
            status="NG: 過程分離不足"
        elif pd.notna(corr) and corr>=.90 and process_diff<.30:
            status="WARN"
        else:
            status="OK"
        rows.append({"比較":f"{a}-{b}","4角順位相関":corr,"4角平均順位差":mean_abs_gap,
                     "4角並び完全一致":exact_same_order,"過程差":process_diff,
                     "4角首位候補同一":same_leader,"首位候補":f"{lead_a}/{lead_b}","判定":status})
    # S3 internal branch existence
    s3=set(branches.loc[branches["シナリオ"]=="S3","内部分岐"]) if not branches.empty else set()
    rows.append({"比較":"S3内部","4角順位相関":np.nan,"4角平均順位差":np.nan,
                 "4角並び完全一致":"-","過程差":np.nan,
                 "4角首位候補同一":"-","首位候補":",".join(sorted(s3)),
                 "判定":"OK" if {"S3-Fa","S3-Fb","S3-L"}.issubset(s3) else "NG: S3分岐欠落"})
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
                for key in ("world", "inputs", "result", "event_log", "branches", "audit"):
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
      st.markdown("### OADP Phase0〜3シナリオ契約")
      st.caption("AIが作成したPhase0〜3 TXTをアップロードすると、4角隊列を上位契約として固定し、その内部で消耗率・残存エネルギー・累積圧力をMonte Carlo算出します。")
      phase_file=st.file_uploader("OADP Phase0〜3 TXT",type=["txt"],key="oadp_phase_contract")
      anchor_strength=st.slider("OADP 4角構造の拘束強度",0.70,0.99,0.90,0.01,
                                help="高いほどAI/OADPの4角隊列を保持し、局所的な揺らぎだけを許します。")
      if phase_file is not None:
        try:
          phase_text=phase_file.getvalue().decode("utf-8",errors="replace")
          contract,contract_meta=parse_oadp_phase_contract(phase_text,st.session_state["base"])
          st.session_state["oadp_contract"]=contract
          st.session_state["oadp_contract_meta"]=contract_meta
          st.success("OADPシナリオ契約を抽出しました。")
          st.dataframe(contract_meta,use_container_width=True)
          st.dataframe(contract.sort_values(["シナリオ","OADP目標4角位置"]),use_container_width=True)
        except Exception as e:
          st.session_state.pop("oadp_contract",None)
          st.session_state.pop("oadp_contract_meta",None)
          st.error(f"OADPシナリオ契約の抽出に失敗しました: {type(e).__name__}: {e}")

      if st.button("OADP契約から消耗率を算出",type="primary",
                   disabled=(not ok or "oadp_contract" not in st.session_state)):
        try:
          rr,cc,aa,ww=simulate_from_oadp_contract(
              st.session_state["base"],st.session_state["oadp_contract"],
              trials=trials,seeds=seeds,anchor_strength=anchor_strength)
          st.session_state.update(contract_result=rr,contract_components=cc,
                                  contract_audit=aa,contract_weighted=ww)
        except Exception as e:
          st.error(f"OADP契約シミュレーションに失敗しました: {type(e).__name__}: {e}")

      if "contract_result" in st.session_state:
        st.subheader("OADP契約シミュレーション結果")
        st.dataframe(st.session_state["contract_result"].sort_values(["シナリオ","4角順位平均"]),use_container_width=True)
        st.subheader("区間別消耗内訳")
        st.dataframe(st.session_state["contract_components"].sort_values(["シナリオ","総消耗率"],ascending=[True,False]),use_container_width=True)
        st.subheader("シナリオ確率加重・統合消耗")
        st.dataframe(st.session_state["contract_weighted"].sort_values("統合消耗率",ascending=False),use_container_width=True)
        st.subheader("OADP隊列保持監査")
        st.dataframe(st.session_state["contract_audit"],use_container_width=True)
        if st.session_state["contract_audit"]["判定"].eq("NG").any():
          st.error("OADP隊列保持監査NGです。拘束強度を上げるか、TXTの4角位置記述を確認してください。")
        export_contract={
          "race":st.session_state.get("race",{}),
          "contract_meta":st.session_state.get("oadp_contract_meta",pd.DataFrame()).to_dict("records"),
          "contract":st.session_state["oadp_contract"].to_dict("records"),
          "results":st.session_state["contract_result"].to_dict("records"),
          "consumption_components":st.session_state["contract_components"].to_dict("records"),
          "weighted":st.session_state["contract_weighted"].to_dict("records"),
          "audit":st.session_state["contract_audit"].to_dict("records"),
          "notice":"OADPの4角構造を上位契約とし、消耗率はDERIVED/SCENARIO_ESTIMATE"
        }
        st.download_button("OADP契約・消耗結果JSON",
          json.dumps(export_contract,ensure_ascii=False,indent=2).encode("utf-8"),
          "oadp_contract_consumption.json","application/json")
        st.download_button("OADP契約・消耗結果CSV",
          st.session_state["contract_result"].to_csv(index=False).encode("utf-8-sig"),
          "oadp_contract_consumption.csv","text/csv")

      st.divider()
      st.markdown("### 自動生成シナリオ（従来モード）")
      if st.button("S1/S2/S3生成・実行",type="primary",disabled=not ok):
        try:
          b,w=worlds(st.session_state["base"]); i=scenario_inputs(b,w); r,e,br=simulate(i,trials,seeds); a=audit(r,i,e,br)
          st.session_state.update(world=w,inputs=i,result=r,event_log=e,branches=br,audit=a)
        except Exception as e:
          st.error(f"シミュレーション実行に失敗しました: {type(e).__name__}: {e}")
      if "world" in st.session_state:
        st.subheader("シナリオ世界")
        st.caption("OADP Ver.2.35：展開構造を先に生成し、S1/S2/S3で別の4角到達式を適用。人気・オッズは不使用。")
        st.dataframe(st.session_state["world"],use_container_width=True)
        st.subheader("全頭×S1/S2/S3入力"); st.dataframe(st.session_state["inputs"],use_container_width=True)
        st.subheader("4角出口・直線結果")
        st.dataframe(st.session_state["result"].sort_values(["シナリオ","4角順位平均"]),use_container_width=True)
        st.subheader("区間イベント集計")
        st.dataframe(st.session_state["event_log"],use_container_width=True)
        st.subheader("S3内部分岐")
        st.dataframe(st.session_state["branches"],use_container_width=True)
        st.subheader("独立監査"); st.dataframe(st.session_state["audit"],use_container_width=True)
        if st.session_state["audit"]["判定"].str.startswith("NG").any(): st.error("分離監査NG。係数または入力世界の再調整が必要です。")
        export={"race":st.session_state.get("race",{}),"worlds":st.session_state["world"].to_dict("records"),
                "scenario_inputs":st.session_state["inputs"].to_dict("records"),"results":st.session_state["result"].to_dict("records"),
                "event_log":st.session_state["event_log"].to_dict("records"),"branches":st.session_state["branches"].to_dict("records"),
                "audit":st.session_state["audit"].to_dict("records"),"notice":"試作モデルのDERIVED/ESTIMATE"}
        st.download_button("結果JSON",json.dumps(export,ensure_ascii=False,indent=2).encode(),"oadp_simulation.json","application/json")
st.divider()
st.caption("未固定係数は暫定式です。公式結果による校正前のため、真の確率を保証しません。")
