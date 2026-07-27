
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

APP_VERSION="0.4.0-event-driven"
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

def simulate(inp,trials,seeds):
    """条件付きイベント型Monte Carlo。各試行で発馬・主張・抵抗・圧力・息入れ・3角進出を発生させる。"""
    summaries=[]; event_rows=[]; branch_rows=[]
    for sid in SCENARIOS:
        d=inp[inp["シナリオ"]==sid].reset_index(drop=True)
        n=len(d); per=max(20,trials//max(1,seeds)); total=per*seeds
        horse_acc={int(r["馬番"]):{"corner":[],"finish":[],"energy":[],"pressure":[],"moves":[],"blocked":[],"faded":[],"lead":0} for _,r in d.iterrows()}
        event_counts={}
        branch_counts={}
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

                energy=np.ones(n)*100.0
                pressure=np.zeros(n)
                move_count=np.zeros(n,int)

                # START
                miss=rng.random(n)<d["出遅れ推定率"].to_numpy(float)
                start_perf=start_cap+rng.normal(0,noise)-miss*rng.uniform(.8,2.1,n)
                participants=rng.random(n)<d["先行参加率"].to_numpy(float)
                if branch=="S3-Fb":
                    # 競るはずの一部が控える
                    front_idx=np.where(participants)[0]
                    if len(front_idx)>1:
                        drop=rng.choice(front_idx,size=max(1,len(front_idx)//2),replace=False)
                        participants[drop]=False
                if not participants.any():
                    participants[np.argmax(start_perf)]=True
                claim_score=np.where(participants,start_perf+.35*d["主張EV"].to_numpy(float),-999)
                leader=int(np.argmax(claim_score))
                order=np.argsort(-start_perf)
                rank=np.empty(n,int); rank[order]=np.arange(1,n+1)
                energy-=2.0+np.maximum(0,7-start_cap)*.20
                energy[participants]-=rng.uniform(.5,1.4,participants.sum())
                event_counts[("START","MISS")]=event_counts.get(("START","MISS"),0)+int(miss.sum())
                event_counts[("START","CLAIM")]=event_counts.get(("START","CLAIM"),0)+int(participants.sum())

                # EARLY_POSITION: pairwise pressure and reaction
                contenders=[i for i in np.where(participants)[0] if i!=leader]
                if sid=="S1":
                    attack_p=.18
                elif sid=="S2":
                    attack_p=.72
                elif branch=="S3-Fa":
                    attack_p=.78
                elif branch=="S3-Fb":
                    attack_p=.10
                else:
                    attack_p=.45
                for i in contenders:
                    if rng.random()<attack_p:
                        resist_p=logistic((d.iloc[leader]["抵抗EV"]-5)/1.1)
                        resisted=rng.random()<resist_p
                        intensity=rng.uniform(.6,1.4)*float(d.iloc[i]["圧力倍率"])
                        pressure[leader]+=intensity
                        pressure[i]+=intensity*.72
                        energy[leader]-=(1.0+max(0,intensity-press_cap[leader]/7)**2)
                        energy[i]-=(.8+max(0,intensity-press_cap[i]/7)**2)
                        event_counts[("EARLY_POSITION","OUTSIDE_PRESS")]=event_counts.get(("EARLY_POSITION","OUTSIDE_PRESS"),0)+1
                        if not resisted and claim_score[i]>claim_score[leader]-.4:
                            leader=i
                            event_counts[("EARLY_POSITION","LEAD_CHANGE")]=event_counts.get(("EARLY_POSITION","LEAD_CHANGE"),0)+1

                # FIRST_CORNER: lane loss / trap
                trapped=rng.random(n)<d["内包まれ率"].to_numpy(float)*(1.0 if branch!="S3-L" else .75)
                wide=rng.random(n)<d["外回し率"].to_numpy(float)
                rank=rank+trapped.astype(int)-wide.astype(int)*0
                energy-=wide*rng.uniform(.4,1.1,n)
                event_counts[("FIRST_CORNER","TRAPPED")]=event_counts.get(("FIRST_CORNER","TRAPPED"),0)+int(trapped.sum())
                event_counts[("FIRST_CORNER","WIDE")]=event_counts.get(("FIRST_CORNER","WIDE"),0)+int(wide.sum())

                # MID_CRUISE: pace load and recovery
                front_zone=rank<=max(3,math.ceil(n*.30))
                pace_load=(1.1 if sid=="S1" else 1.7 if sid=="S2" else 1.5 if branch=="S3-Fa" else .9 if branch=="S3-Fb" else 1.65)
                demand=pace_load+front_zone*.7+pressure*.28
                overload=np.maximum(0,demand-cruise_cap/5.2)
                energy-=1.4+demand*.55+overload**2
                recover=(~front_zone)*recovery_cap/10*rng.uniform(.5,1.2,n)
                if branch=="S3-Fb": recover+=front_zone*recovery_cap/10*.55
                energy+=recover
                event_counts[("MID_CRUISE","RECOVERY")]=event_counts.get(("MID_CRUISE","RECOVERY"),0)+int((recover>.5).sum())

                # THIRD_CORNER: conditional moves
                if branch=="S3-Fb":
                    move_p=.12+.22*d["3角進出EV"].to_numpy(float)/10
                elif branch=="S3-L":
                    move_p=.30+.55*d["差し接続EV"].to_numpy(float)/10
                else:
                    move_p=.20+.42*d["3角進出EV"].to_numpy(float)/10
                move_p=np.clip(move_p*(.65+energy/140),.03,.92)
                movers=rng.random(n)<move_p
                move_strength=corner_cap+rng.normal(0,noise)-np.maximum(0,55-energy)*.06
                shift=np.where(movers,np.clip(np.round((move_strength-4.5)/2),0,3),0).astype(int)
                move_count+=shift
                rank=np.maximum(1,rank-shift)
                energy-=movers*(.7+shift*.65+np.maximum(0,6-corner_cap)*.20)
                event_counts[("THIRD_CORNER","MOVE")]=event_counts.get(("THIRD_CORNER","MOVE"),0)+int(movers.sum())

                # FOURTH_CORNER_EXIT: position, velocity, energy — no final result yet
                front_bonus=np.where(np.arange(n)==leader,1.5,0)
                velocity=.38*cruise_cap+.42*corner_cap+.20*pos_cap+front_bonus-.32*pressure-.055*np.maximum(0,60-energy)
                if branch=="S3-L":
                    velocity+=.28*straight_cap-.18*start_cap
                if branch=="S3-Fb":
                    velocity+=front_zone*.75
                corner_score=velocity-.42*rank+rng.normal(0,noise*.42)
                corner_order=np.argsort(-corner_score); corner_rank=np.empty(n,int); corner_rank[corner_order]=np.arange(1,n+1)

                # Straight uses ability after structure is fixed
                blocked=rng.random(n)<np.clip(d["進路詰まり率"].to_numpy(float)+trapped*.12+wide*.04,.02,.80)
                fade_p=np.clip(d["ラスト失速率"].to_numpy(float)+np.maximum(0,55-energy)/100+pressure*.025,.03,.95)
                faded=rng.random(n)<fade_p
                effective_straight=(.54*straight_cap+.24*vmax_cap+.22*corner_cap)*(energy/100)
                final_score=.28*corner_score+.72*effective_straight-blocked*rng.uniform(.5,1.7,n)-faded*rng.uniform(.7,2.2,n)+rng.normal(0,noise*.35)
                finish_order=np.argsort(-final_score); finish_rank=np.empty(n,int); finish_rank[finish_order]=np.arange(1,n+1)

                for i,row in d.iterrows():
                    k=int(row["馬番"]); a=horse_acc[k]
                    a["corner"].append(int(corner_rank[i])); a["finish"].append(int(finish_rank[i]))
                    a["energy"].append(float(energy[i])); a["pressure"].append(float(pressure[i]))
                    a["moves"].append(int(move_count[i])); a["blocked"].append(int(blocked[i])); a["faded"].append(int(faded[i]))
                    if corner_rank[i]==1: a["lead"]+=1

        for _,row in d.iterrows():
            k=int(row["馬番"]); a=horse_acc[k]
            summaries.append({
                "シナリオ":sid,"馬番":k,"馬名":row["馬名"],
                "4角順位平均":np.mean(a["corner"]),"4角順位SD":np.std(a["corner"]),
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
        if pd.notna(corr) and corr>=.96 and process_diff<.18:
            status="NG: 過程分離不足"
        elif pd.notna(corr) and corr>=.90 and process_diff<.30:
            status="WARN"
        else:
            status="OK"
        rows.append({"比較":f"{a}-{b}","4角順位相関":corr,"過程差":process_diff,
                     "4角首位候補同一":same_leader,"首位候補":f"{lead_a}/{lead_b}","判定":status})
    # S3 internal branch existence
    s3=set(branches.loc[branches["シナリオ"]=="S3","内部分岐"]) if not branches.empty else set()
    rows.append({"比較":"S3内部","4角順位相関":np.nan,"過程差":np.nan,
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
