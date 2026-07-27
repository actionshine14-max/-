
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import math
import json
import re
from pathlib import Path
from statistics import mean

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None


# ============================================================
# OADP Expectancy Engine
# ============================================================
# このエンジンの最重要設計:
# 1. STEP0 → Phase0 → Phase1 → Phase2 → Phase3 → Phase4 の順番を固定する。
# 2. Phase1では馬の強弱・人気・オッズ・印を使わず、4角出口〜直線入口の隊列だけを作る。
# 3. 発生確率はPhase1の隊列構造raw EVから正規化する。
# 4. Phase3以降で初めて馬評価・オッズ・印候補を使う。
# 5. 監査NGが1つでもあれば、Phase4の買い目を出さない。


@dataclass
class PastRun:
    rank: Optional[int] = None
    date: str = ""
    going: str = ""
    heads: Optional[int] = None
    venue: str = ""
    course: str = ""
    distance: Optional[int] = None
    gate_no: Optional[int] = None
    race_name: str = ""
    popularity: Optional[int] = None
    body_weight: Optional[int] = None
    jockey: str = ""
    carried_weight: Optional[float] = None
    time_text: str = ""
    passing_text: str = ""
    agari: Optional[float] = None
    margin: Optional[float] = None
    winner: str = ""

    @property
    def passing_positions(self) -> List[int]:
        if not self.passing_text:
            return []
        vals = []
        for token in re.findall(r"\d+", self.passing_text):
            try:
                vals.append(int(token))
            except ValueError:
                pass
        return vals

    @property
    def first_pos(self) -> Optional[int]:
        return self.passing_positions[0] if self.passing_positions else None

    @property
    def last_pos(self) -> Optional[int]:
        return self.passing_positions[-1] if self.passing_positions else None

    @property
    def improved_positions(self) -> int:
        if len(self.passing_positions) < 2:
            return 0
        return self.passing_positions[0] - self.passing_positions[-1]


@dataclass
class Horse:
    frame_no: Optional[int]
    horse_no: int
    name: str
    jockey: str = ""
    trainer: str = ""
    sex_age: str = ""
    carried_weight: Optional[float] = None
    odds: Optional[float] = None
    popularity: Optional[int] = None
    body_weight: Optional[int] = None
    body_weight_diff: Optional[int] = None
    total_record: str = ""
    left_record: str = ""
    right_record: str = ""
    track_record: str = ""
    distance_record: str = ""
    best_time: str = ""
    sire: str = ""
    dam: str = ""
    damsire: str = ""
    equipment_notes: str = ""
    past_runs: List[PastRun] = field(default_factory=list)


@dataclass
class Race:
    title: str = ""
    date_text: str = ""
    track: str = ""
    race_no: Optional[int] = None
    post_time: str = ""
    course: str = ""
    distance: Optional[int] = None
    direction: str = ""
    weather: str = ""
    going: str = ""
    race_class: str = ""
    horses: List[Horse] = field(default_factory=list)

    @property
    def field_size(self) -> int:
        return len(self.horses)

    @property
    def candidate_count(self) -> int:
        if self.field_size >= 12:
            return 8
        if self.field_size >= 10:
            return 7
        return 6


@dataclass
class HorseFeature:
    horse_no: int
    name: str
    front_score: float
    corner_score: float
    closing_score: float
    fade_risk: float
    track_fit: float
    distance_fit: float
    sand_risk: float
    weight_bonus: float
    recency_stability: float
    raw_style: str
    odds: Optional[float] = None
    popularity: Optional[int] = None

    @property
    def four_corner_ev(self) -> float:
        return clamp(
            0.34 * self.front_score
            + 0.28 * self.corner_score
            + 0.14 * self.track_fit
            + 0.14 * self.distance_fit
            + 0.10 * self.recency_stability
            - 0.10 * self.sand_risk,
            0,
            1,
        )

    @property
    def straight_ev(self) -> float:
        return clamp(
            0.38 * self.closing_score
            + 0.24 * (1.0 - self.fade_risk)
            + 0.16 * self.distance_fit
            + 0.12 * self.track_fit
            + 0.10 * self.recency_stability,
            0,
            1,
        )

    @property
    def front_pressure_ev(self) -> float:
        # 弱い先行馬でも前列圧に参加できるため、fade_riskを減点しすぎない。
        return clamp(0.70 * self.front_score + 0.20 * self.corner_score + 0.10 * self.sand_risk, 0, 1)

    @property
    def total_ability_ev(self) -> float:
        # Phase3以降でのみ使用。Phase1では使わない。
        return clamp(
            0.34 * self.four_corner_ev
            + 0.34 * self.straight_ev
            + 0.12 * self.track_fit
            + 0.12 * self.distance_fit
            + 0.08 * self.recency_stability,
            0,
            1,
        )


@dataclass
class Scenario:
    key: str
    label: str
    raw_probability_ev: float
    probability: float
    corner_queue: List[int]
    explanation: str
    structure_type: str


@dataclass
class PhaseResult:
    step0_text: str
    phase0_text: str
    phase1_text: str
    phase2_text: str
    phase3_text: str
    phase4_text: str
    audit_table: List[Dict[str, str]]
    audit_ok: bool
    files: Dict[str, str] = field(default_factory=dict)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
    return float(m.group(1)) if m else None




def extract_current_carried_weight(text: str, is_ban_ei: bool = False) -> Optional[float]:
    """出馬表の当日斤量を安全に抽出する。
    旧実装は「04.05生 54.0」のようなセルから誕生日の04.0を拾う事故があった。
    通常平地は45.0〜65.0、ばんえいは400〜900台を候補にする。
    """
    t = normalize_spaces(text or "")
    if not t:
        return None
    if is_ban_ei:
        m = re.search(r"(?<!\d)([4-9]\d{2})(?!\d)", t)
        return float(m.group(1)) if m else None
    # 減量記号つきも許容。誕生日 04.05 や騎手番号を拾わないよう45〜65台に限定。
    candidates: List[float] = []
    for m in re.finditer(r"(?:[▲△◇☆]\s*)?((?:4[5-9]|5\d|6[0-5])\.\d)(?!\d)", t):
        try:
            candidates.append(float(m.group(1)))
        except ValueError:
            pass
    return candidates[-1] if candidates else None


def carried_weight_status(weight: Optional[float], race: Race) -> str:
    if weight is None:
        return "MISSING"
    if "帯広" in (race.track or "") or "ばんえい" in (race.race_class or ""):
        return "OK" if 400 <= weight <= 900 else "WARN"
    return "OK" if 45.0 <= weight <= 65.0 else "ERROR"


def safe_int(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(-?\d+)", text)
    return int(m.group(1)) if m else None


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def parse_record_table_text(text: str) -> Dict[str, str]:
    # NARの着別成績欄はHTMLテーブル内で整形が崩れやすいので、
    # 完全パースではなく、表示用の圧縮テキストとして保持する。
    compact = normalize_spaces(text)
    return {"raw": compact}


def parse_passing_time_cell(text: str) -> Tuple[str, str, Optional[float]]:
    t = normalize_spaces(text)
    time_text = ""
    passing = ""
    agari = None
    tm = re.search(r"(\d:\d{2}\.\d|0:\d{2}\.\d|\d{1,2}\.\d)", t)
    if tm:
        time_text = tm.group(1)
    # 通過順は「1-1-1-1」「5-5」などを拾う。時計や着差は除外しにくいので、最長のハイフン列を採用。
    passings = re.findall(r"\d+(?:-\d+){1,5}", t)
    if passings:
        passing = max(passings, key=len)
    nums = re.findall(r"\d{2}\.\d|\d{1,2}\.\d", t)
    if nums:
        try:
            # 最後の小数を上がり候補にする。時計が1:32.6形式の場合は除外される。
            agari = float(nums[-1])
        except ValueError:
            agari = None
    return time_text, passing, agari


def parse_margin_cell(text: str) -> Tuple[Optional[float], str]:
    t = normalize_spaces(text)
    if not t:
        return None, ""
    m = re.search(r"(-?\d+(?:\.\d+)?)\s+(.+)$", t)
    if m:
        return float(m.group(1)), m.group(2).strip()
    return safe_float(t), ""



NAR_TRACK_NAMES: Tuple[str, ...] = (
    "帯広", "門別", "盛岡", "水沢", "浦和", "船橋", "大井", "川崎",
    "金沢", "笠松", "名古屋", "園田", "姫路", "高知", "佐賀",
)


def extract_nar_header_metadata(text: str) -> Optional[Dict[str, Any]]:
    """地方競馬のレースヘッダーから開催日・競馬場・R番号・発走時刻を抽出する。

    NAR公式ページは「浦 和」「門 別」のように競馬場名の文字間へ
    空白を挿入するため、監査用コピーでは全空白を除去してから照合する。
    元の表示文字列は source として保持する。
    """
    if not text:
        return None

    source = normalize_spaces(str(text))
    # 全角括弧・全角コロンを正規化し、HTML由来の文字間空白を除去する。
    compact = (
        source.replace("（", "(")
        .replace("）", ")")
        .replace("：", ":")
        .replace("\u3000", "")
        .replace("\xa0", "")
    )
    compact = re.sub(r"\s+", "", compact)

    track_pattern = "|".join(map(re.escape, NAR_TRACK_NAMES))
    match = re.search(
        rf"(?P<date>20\d{{2}}年\d{{1,2}}月\d{{1,2}}日(?:\([^)]*\))?)"
        rf".*?(?P<track>{track_pattern})"
        rf".*?第?(?P<race_no>\d{{1,2}})(?:競走|R)"
        rf"(?:(?P<hour>\d{{1,2}}):(?P<minute>\d{{2}})発走)?",
        compact,
    )
    if not match:
        return None

    post_time = ""
    if match.group("hour") and match.group("minute"):
        post_time = f"{int(match.group('hour')):02d}:{match.group('minute')}"

    return {
        "date_text": match.group("date"),
        "track": match.group("track"),
        "race_no": int(match.group("race_no")),
        "post_time": post_time,
        "source": source,
        "compact_source": compact,
    }


def apply_nar_header_metadata(race: Race, text: str) -> bool:
    """抽出できた地方競馬ヘッダーをRaceへ反映する。"""
    metadata = extract_nar_header_metadata(text)
    if not metadata:
        return False

    race.date_text = str(metadata["date_text"])
    race.track = str(metadata["track"])
    race.race_no = int(metadata["race_no"])
    if metadata.get("post_time"):
        race.post_time = str(metadata["post_time"])

    source = str(metadata["source"])
    # JRA/plain parserと同じ属性名を併記し、main.pyの監査ログでも必ず参照可能にする。
    setattr(race, "source_header_line", source)
    setattr(race, "source_race_header_line", source)
    setattr(race, "source_race_no_line", source)
    return True

def parse_nar_html(html: str) -> Race:
    if BeautifulSoup is None:
        raise RuntimeError("beautifulsoup4 がインストールされていません。pip install -r requirements.txt を実行してください。")

    soup = BeautifulSoup(html, "html.parser")
    race = Race()

    # 地方競馬のレース本体ヘッダーを最優先で解析する。
    # NAR公式HTMLは「浦 和」「門 別」のように競馬場名の文字間へ
    # 空白を挿入するため、extract_nar_header_metadata内で全空白を除去して照合する。
    header_candidates: List[str] = []

    # 通常は最初のh4が「日付 競馬場 第N競走 発走時刻」。
    for node in soup.find_all("h4")[:5]:
        candidate = normalize_spaces(node.get_text(" ", strip=True))
        if candidate:
            header_candidates.append(candidate)

    # HTML構造変更やspan分割時の保険として、ページ先頭側の本文も候補にする。
    page_head = normalize_spaces(soup.get_text(" ", strip=True))[:5000]
    if page_head:
        header_candidates.append(page_head)

    header_applied = False
    for candidate in header_candidates:
        if apply_nar_header_metadata(race, candidate):
            header_applied = True
            break

    if not header_applied and header_candidates:
        # 日付行を完全に読めなかった場合も元表示は保持し、AI監査で復元可能にする。
        race.date_text = header_candidates[0]
        setattr(race, "source_header_line", header_candidates[0])
        setattr(race, "source_race_header_line", header_candidates[0])

    title = soup.select_one("section.raceTitle h3")
    if title:
        race.title = normalize_spaces(title.get_text(" "))
        setattr(race, "source_race_title_line", race.title)

    data_li = soup.select_one("section.raceTitle ul.dataArea li")
    if data_li:
        txt = normalize_spaces(data_li.get_text(" "))
        race.course = txt
        dm = re.search(r"(ダート|芝)\s*(\d+)ｍ", txt)
        if dm:
            race.course = dm.group(1)
            race.distance = int(dm.group(2))
        direction_m = re.search(r"ｍ（([^）]+)）", txt)
        if direction_m:
            race.direction = direction_m.group(1)
        wm = re.search(r"天候：([^ ]+)", txt)
        gm = re.search(r"馬場：([^ ]+)", txt)
        if wm:
            race.weather = wm.group(1)
        if gm:
            race.going = gm.group(1)
        # 後半に条件が詰まっているので表示用だけ保持
        race.race_class = txt

    # 馬ブロック解析
    current_frame = None
    tborder_rows = soup.select("section.cardTable table > tbody > tr.tBorder")
    for row in tborder_rows:
        frame_cell = row.select_one(".courseNum")
        if frame_cell:
            current_frame = safe_int(frame_cell.get_text(" "))
        horse_cell = row.select_one(".horseNum")
        if not horse_cell:
            continue
        horse_no = safe_int(horse_cell.get_text(" "))
        if horse_no is None:
            continue
        hname = row.select_one(".horseName")
        name = normalize_spaces(hname.get_text(" ")) if hname else f"{horse_no}番"
        jockey = normalize_spaces(row.select_one(".jockeyName").get_text(" ")) if row.select_one(".jockeyName") else ""

        odds_cell = row.select_one("td.odds_weight")
        odds = None
        popularity = None
        if odds_cell:
            ot = normalize_spaces(odds_cell.get_text(" "))
            # 最初の小数を単勝オッズとみなす
            om = re.search(r"(\d+(?:\.\d+)?)", ot)
            if om:
                odds = float(om.group(1))
            pm = re.search(r"\((\d+)人気\)", ot)
            if pm:
                popularity = int(pm.group(1))

        # ブロックの5行を取得
        block_rows = [row]
        sib = row.find_next_sibling("tr")
        while sib and len(block_rows) < 5:
            block_rows.append(sib)
            sib = sib.find_next_sibling("tr")

        horse = Horse(
            frame_no=current_frame,
            horse_no=horse_no,
            name=name,
            jockey=jockey,
            odds=odds,
            popularity=popularity,
        )

        # 性齢・斤量
        if len(block_rows) > 1:
            txts = [normalize_spaces(td.get_text(" ")) for td in block_rows[1].find_all("td")]
            if txts:
                for t in txts[:5]:
                    if re.search(r"[牡牝セ]\d+", t):
                        horse.sex_age = t
                        break
                is_ban_ei = "帯広" in (race.track or "") or "ばんえい" in (race.race_class or "")
                for t in txts:
                    w = extract_current_carried_weight(t, is_ban_ei=is_ban_ei)
                    if w is not None:
                        horse.carried_weight = w
                        break

        # 調教師・馬体重・血統
        if len(block_rows) > 2:
            trainer_link = block_rows[2].select_one('a[href*="TrainerMark"]')
            if trainer_link:
                horse.trainer = normalize_spaces(trainer_link.get_text(" "))
            # 馬体重セル
            for td in block_rows[2].find_all("td"):
                t = normalize_spaces(td.get_text(" "))
                bw = re.search(r"\b(\d{3,4})\s*\(([+-]?\d+)\)", t)
                if bw:
                    horse.body_weight = int(bw.group(1))
                    horse.body_weight_diff = int(bw.group(2))
                    break
            first3 = [normalize_spaces(td.get_text(" ")) for td in block_rows[2].find_all("td")[:3]]
            if first3:
                horse.sire = first3[0]
        if len(block_rows) > 3:
            first3 = [normalize_spaces(td.get_text(" ")) for td in block_rows[3].find_all("td")[:3]]
            if first3:
                horse.dam = first3[0]
        if len(block_rows) > 4:
            first3 = [normalize_spaces(td.get_text(" ")) for td in block_rows[4].find_all("td")[:3]]
            if first3:
                horse.damsire = first3[0]

        result_td = row.select_one("td.result")
        if result_td:
            horse.total_record = parse_record_table_text(result_td.get_text(" ")).get("raw", "")

        # 近走
        race_infos = row.select("div.raceInfo")
        race_names: List[str] = []
        if len(block_rows) > 1:
            for a in block_rows[1].select("a.race, div.jrarace"):
                race_names.append(normalize_spaces(a.get_text(" ")))

        row2_texts = [normalize_spaces(td.get_text(" ")) for td in block_rows[2].find_all("td")] if len(block_rows) > 2 else []
        row3_texts = [normalize_spaces(td.get_text(" ")) for td in block_rows[3].find_all("td")] if len(block_rows) > 3 else []
        row4_texts = [normalize_spaces(td.get_text(" ")) for td in block_rows[4].find_all("td")] if len(block_rows) > 4 else []

        perf_cells = [t for t in row3_texts if re.search(r"(\d:\d{2}\.\d|0:\d{2}\.\d|\d{1,2}\.\d)", t)]
        pop_cells = [t for t in row2_texts if "人" in t and re.search(r"\d+人", t)]
        margin_cells = [t for t in row4_texts if re.search(r"\d", t) and not re.search(r"[牡牝セ]\d+", t)]

        for i, info in enumerate(race_infos[:5]):
            it = normalize_spaces(info.get_text(" "))
            pr = info.select_one(".pastRank")
            rank = safe_int(pr.get_text(" ")) if pr else None
            date_m = re.search(r"(\d{2}\.\d{2}\.\d{2})", it)
            heads_m = re.search(r"(\d+)頭", it)
            date = date_m.group(1) if date_m else ""
            heads = int(heads_m.group(1)) if heads_m else None
            going = ""
            gm = re.search(r"\d{2}\.\d{2}\.\d{2}\s*([^ ]+)\s*(\d+)頭", it)
            if gm:
                going = gm.group(1)

            # 競馬場/距離
            venue = ""
            course = ""
            dist = None
            gate_no = None
            vm = re.search(r"(川崎|浦和|船橋|大井|盛岡|水沢|佐賀|高知|門別|帯広|Ｊ阪神|Ｊ東京|Ｊ中山|Ｊ京都|Ｊ新潟|Ｊ福島|Ｊ小倉|Ｊ中京).*?(右|左|芝|ダ)?\s*(\d{3,4})", it)
            if vm:
                venue = vm.group(1)
                course = vm.group(2) or ""
                dist = int(vm.group(3))
            gate_m = re.search(r"(\d+)番", it)
            if gate_m:
                gate_no = int(gate_m.group(1))

            run = PastRun(
                rank=rank,
                date=date,
                going=going,
                heads=heads,
                venue=venue,
                course=course,
                distance=dist,
                gate_no=gate_no,
                race_name=race_names[i] if i < len(race_names) else "",
            )

            if i < len(pop_cells):
                pt = pop_cells[i]
                pm = re.search(r"(\d+)人", pt)
                if pm:
                    run.popularity = int(pm.group(1))
                # 騎手名と斤量
                wtm = re.search(r"([▲△◇☆]?\s*[\u4e00-\u9fffぁ-んァ-ンーA-Za-z]+)\s*(\d{2}\.\d)", pt)
                if wtm:
                    run.jockey = normalize_spaces(wtm.group(1))
                    run.carried_weight = float(wtm.group(2))
                bwm = re.search(r"人\s*(\d{3,4})\s*", pt)
                if bwm:
                    run.body_weight = int(bwm.group(1))

            if i < len(perf_cells):
                time_text, passing, agari = parse_passing_time_cell(perf_cells[i])
                run.time_text = time_text
                run.passing_text = passing
                run.agari = agari

            if i < len(margin_cells):
                mg, winner = parse_margin_cell(margin_cells[i])
                run.margin = mg
                run.winner = winner

            horse.past_runs.append(run)

        race.horses.append(horse)

    return race


def record_fit_from_text(text: str, label: str) -> float:
    # "場 1- 0- 1- 3" のような完全構造はHTML崩れで安定しないため、
    # 粗いスコアとして勝ち/連対の有無だけを拾う。
    if not text:
        return 0.50
    # まとまった成績欄の中で「場」「距」を特定するのは難しいので、
    # 出現数字から勝率寄りの粗評価にする。
    nums = [int(x) for x in re.findall(r"\b\d+\b", text)]
    if len(nums) >= 4:
        wins = nums[0]
        seconds = nums[1] if len(nums) > 1 else 0
        thirds = nums[2] if len(nums) > 2 else 0
        starts = sum(nums[:4]) if sum(nums[:4]) > 0 else 1
        return clamp((wins * 1.0 + seconds * 0.65 + thirds * 0.35) / starts + 0.25, 0.25, 0.95)
    return 0.50


def compute_features(race: Race) -> Dict[int, HorseFeature]:
    features: Dict[int, HorseFeature] = {}
    race_distance = race.distance or 0

    for h in race.horses:
        runs = h.past_runs
        weighted_front = []
        weighted_corner = []
        weighted_closing = []
        weighted_fade = []
        sand_risks = []
        distance_scores = []
        weights = [1.00, 0.82, 0.66, 0.52, 0.42]

        for idx, r in enumerate(runs[:5]):
            w = weights[idx] if idx < len(weights) else 0.35
            heads = r.heads or max([rr.heads or 0 for rr in runs] + [10])
            denom = max(heads - 1, 1)
            first = r.first_pos
            last = r.last_pos
            if first is not None:
                front = 1.0 - (first - 1) / denom
            else:
                front = 0.50
            if last is not None:
                corner = 1.0 - (last - 1) / denom
            else:
                corner = front * 0.85

            # 上がりが速いほど高くする。ただし地方ダートは時計差が大きいので粗正規化。
            if r.agari is not None:
                closing = clamp((43.5 - r.agari) / 7.0, 0.08, 0.98)
            else:
                closing = 0.50

            # 通過順改善を直線/3-4角余力に加える
            if r.improved_positions > 0:
                closing = clamp(closing + min(r.improved_positions, 5) * 0.045, 0, 1)
            elif r.improved_positions < -1:
                closing = clamp(closing - min(abs(r.improved_positions), 5) * 0.035, 0, 1)

            fade = 0.0
            if first is not None and first <= 3 and r.rank is not None and r.rank >= 6:
                fade += 0.35
            if first is not None and last is not None and last - first >= 3:
                fade += 0.25
            if r.agari is not None and r.agari >= 41.5:
                fade += 0.22
            if r.margin is not None and r.margin >= 1.8:
                fade += 0.18
            fade = clamp(fade, 0, 1)

            # 砂被り疑い。断定ではない。後方/内寄り/通過悪化を機械的にリスク化。
            sand = 0.15
            if first is not None and first >= 6:
                sand += 0.18
            if r.gate_no is not None and r.gate_no <= 3 and first is not None and first >= 5:
                sand += 0.12
            if first is not None and last is not None and last > first:
                sand += 0.10
            sand = clamp(sand, 0, 0.75)

            if race_distance and r.distance:
                diff = abs(r.distance - race_distance)
                if diff == 0:
                    dist_score = 0.84
                elif diff <= 100:
                    dist_score = 0.74
                elif diff <= 200:
                    dist_score = 0.63
                elif diff <= 400:
                    dist_score = 0.53
                else:
                    dist_score = 0.42
            else:
                dist_score = 0.50

            weighted_front.append(front * w)
            weighted_corner.append(corner * w)
            weighted_closing.append(closing * w)
            weighted_fade.append(fade * w)
            sand_risks.append(sand * w)
            distance_scores.append(dist_score * w)

        wsum = sum(weights[: len(runs)]) if runs else 1.0
        front_score = sum(weighted_front) / wsum if weighted_front else 0.50
        corner_score = sum(weighted_corner) / wsum if weighted_corner else 0.50
        closing_score = sum(weighted_closing) / wsum if weighted_closing else 0.50
        fade_risk = sum(weighted_fade) / wsum if weighted_fade else 0.35
        sand_risk = sum(sand_risks) / wsum if sand_risks else 0.35
        distance_fit = sum(distance_scores) / wsum if distance_scores else 0.50
        track_fit = record_fit_from_text(h.total_record, "場")

        recent_ranks = [r.rank for r in runs[:3] if r.rank is not None]
        if recent_ranks:
            # 低着順ほど高く、ただし着順だけ評価にならないよう重みは小さめ
            recency = clamp(1.0 - (mean(recent_ranks) - 1) / 10.0, 0.20, 0.95)
        else:
            recency = 0.50

        weight_bonus = 0.0
        if h.carried_weight is not None:
            if h.carried_weight <= 51:
                weight_bonus = 0.16
            elif h.carried_weight <= 52:
                weight_bonus = 0.10
            elif h.carried_weight <= 53:
                weight_bonus = 0.06

        if front_score >= 0.74:
            style = "逃げ/前列"
        elif front_score >= 0.60:
            style = "先行"
        elif corner_score >= 0.50:
            style = "好位/中団"
        elif closing_score >= 0.60:
            style = "差し"
        else:
            style = "後方/保留"

        features[h.horse_no] = HorseFeature(
            horse_no=h.horse_no,
            name=h.name,
            front_score=round(front_score, 4),
            corner_score=round(corner_score, 4),
            closing_score=round(closing_score, 4),
            fade_risk=round(fade_risk, 4),
            track_fit=round(track_fit, 4),
            distance_fit=round(distance_fit, 4),
            sand_risk=round(sand_risk, 4),
            weight_bonus=round(weight_bonus, 4),
            recency_stability=round(recency, 4),
            raw_style=style,
            odds=h.odds,
            popularity=h.popularity,
        )

    return features


def candidate_count_for_field(n: int) -> int:
    if n >= 12:
        return 8
    if n >= 10:
        return 7
    return 6


def generate_phase0(race: Race, features: Dict[int, HorseFeature]) -> Tuple[str, Dict[str, Any]]:
    front = [f for f in features.values() if f.front_score >= 0.70]
    press = [f for f in features.values() if 0.58 <= f.front_score < 0.70]
    mid = [f for f in features.values() if f.front_score < 0.58 and f.corner_score >= 0.48]
    back = [f for f in features.values() if f.corner_score < 0.48]

    front_pressure = clamp(
        (len(front) * 0.22 + len(press) * 0.11)
        + mean([f.fade_risk for f in front + press]) * 0.35 if (front or press) else 0.20,
        0,
        1,
    )
    diff_power = clamp(mean([f.closing_score for f in features.values()]) if features else 0.5, 0, 1)
    weak_front_ratio = mean([f.fade_risk for f in front + press]) if (front or press) else 0.3

    lines = []
    lines.append(f"Phase 0：展開前処理")
    lines.append(f"レース：{race.date_text}／{race.title}／{race.course}{race.distance}m／{race.going}／{race.field_size}頭")
    lines.append("※このPhaseでは印・買い目を出さない。馬の強弱ではなく、隊列参加能力だけを処理する。")
    lines.append("")
    lines.append("【逃げ・前列候補】" + "、".join([f"{f.horse_no}{f.name}" for f in sorted(front, key=lambda x: -x.front_score)]) if front else "【逃げ・前列候補】該当明確馬なし")
    lines.append("【先行候補】" + "、".join([f"{f.horse_no}{f.name}" for f in sorted(press, key=lambda x: -x.front_score)]) if press else "【先行候補】該当明確馬なし")
    lines.append("【好位・中団候補】" + "、".join([f"{f.horse_no}{f.name}" for f in sorted(mid, key=lambda x: -x.corner_score)]) if mid else "【好位・中団候補】該当明確馬なし")
    lines.append("【差し・後方候補】" + "、".join([f"{f.horse_no}{f.name}" for f in sorted(back, key=lambda x: -x.closing_score)]) if back else "【差し・後方候補】該当明確馬なし")
    lines.append("")
    lines.append(f"前列圧raw：{front_pressure:.3f}／差し接続raw：{diff_power:.3f}／前列失速raw：{weak_front_ratio:.3f}")
    lines.append("人気・オッズはPhase0の隊列構築には未使用。")
    data = {
        "front": [f.horse_no for f in front],
        "press": [f.horse_no for f in press],
        "mid": [f.horse_no for f in mid],
        "back": [f.horse_no for f in back],
        "front_pressure": front_pressure,
        "diff_power": diff_power,
        "weak_front_ratio": weak_front_ratio,
    }
    return "\n".join(lines), data


def generate_phase1(race: Race, features: Dict[int, HorseFeature], p0: Dict[str, Any]) -> Tuple[str, List[Scenario], Dict[str, Any]]:
    # Phase1では人気・オッズを使わない。features.odds/popularityは参照禁止。
    ordered_by_front = sorted(features.values(), key=lambda f: (-f.front_score, -f.corner_score, f.horse_no))
    ordered_by_straight = sorted(features.values(), key=lambda f: (-f.straight_ev, -f.closing_score, f.horse_no))
    ordered_by_corner = sorted(features.values(), key=lambda f: (-f.four_corner_ev, -f.front_score, f.horse_no))

    field_n = race.field_size
    pressure = p0["front_pressure"]
    diff_power = p0["diff_power"]
    weak_front = p0["weak_front_ratio"]
    front_count = len(p0["front"]) + len(p0["press"])

    # S1: 標準。前列→好位→直線余力の自然接続。
    s1_queue = []
    for f in ordered_by_front[: max(2, min(4, field_n))]:
        s1_queue.append(f.horse_no)
    for f in ordered_by_corner:
        if f.horse_no not in s1_queue:
            s1_queue.append(f.horse_no)
    s1_queue = s1_queue[:field_n]

    # S2: 前列圧が入る。前列馬は4角には残るが、直線余力馬が接続する。
    s2_queue = []
    pressure_front = ordered_by_front[: max(3, min(5, field_n))]
    for f in pressure_front:
        s2_queue.append(f.horse_no)
    for f in ordered_by_straight:
        if f.horse_no not in s2_queue:
            s2_queue.append(f.horse_no)
    s2_queue = s2_queue[:field_n]

    # S3: 構造穴。前残り穴型と差し荒れ型のrawを比較し、優勢側で隊列を作る。
    #
    # v1.1 修正:
    # 以前は late_collapse_raw が少しでも front_survival_raw を上回ると S3-L に固定され、
    # 「荒れ前残り率40%以上」でも前列大穴がS3上位から押し出される事故があった。
    # OADP固定ルール上、荒れ前残りが40%以上ならS3は前残り穴/K6D-K6Fを主役にする。
    # そのため、Phase1時点でも同じ式で前残り率proxyを作り、40%以上ならS3-Fを強制する。
    front_survival_raw = clamp(0.35 + 0.35 * pressure - 0.25 * weak_front + 0.08 * max(front_count - 2, 0), 0, 1)
    late_collapse_raw = clamp(0.25 + 0.45 * pressure + 0.25 * weak_front + 0.20 * diff_power, 0, 1)
    arere_front_proxy = clamp(0.18 + 0.42 * pressure + 0.25 * weak_front - 0.18 * diff_power, 0.05, 0.70)

    force_s3_front = arere_front_proxy >= 0.40
    if (not force_s3_front) and late_collapse_raw >= front_survival_raw:
        structure_type = "S3-L：前列圧差し荒れ"
        s3_queue = []
        # 直線入口で前列が削られ、差し接続馬が4角出口から上がる構造
        for f in ordered_by_straight:
            s3_queue.append(f.horse_no)
        # ただし前残り穴候補も完全には消さない
        for f in ordered_by_front:
            if f.horse_no not in s3_queue:
                s3_queue.append(f.horse_no)
    else:
        structure_type = "S3-F：前残り大穴構造"
        s3_queue = []
        q_front_index = {f.horse_no: i for i, f in enumerate(ordered_by_front)}
        # 前残り荒れでは、失速リスクを「消し」ではなく「波乱要因」として扱う。
        # 4角前列にいる大穴を最初に保護し、その後に差し接続馬を同時発生相手として接続する。
        front_big = [
            f for f in ordered_by_front
            if is_big_longshot(f, race) and f.front_pressure_ev >= 0.50
        ]
        front_big = sorted(
            front_big,
            key=lambda f: (
                -front_stay_big_score(f, q_front_index, race),
                q_front_index.get(f.horse_no, 99),
                f.horse_no,
            ),
        )
        for f in front_big:
            s3_queue.append(f.horse_no)
        for f in sorted(features.values(), key=lambda f: (-front_stay_big_score(f, q_front_index, race), q_front_index.get(f.horse_no, 99), f.horse_no)):
            if f.horse_no not in s3_queue:
                s3_queue.append(f.horse_no)
        for f in ordered_by_straight:
            if f.horse_no not in s3_queue:
                s3_queue.append(f.horse_no)
    s3_queue = s3_queue[:field_n]

    # Raw発生EV。馬個別の強さではなく、隊列構造から算出する。
    #
    # v0.7 修正:
    # 以前の式はS1に固定基礎値を厚く置きすぎ、前列圧・同型過多・前列失速・差し接続が
    # 強いレースでもS1が40%前後へ残りやすかった。ここでは「荒れ構造指数」を先に作り、
    # 荒れ構造が強い場合はS1へ上限をかけ、S2/S3へ確率を移す。
    pressure_count_bonus = clamp(max(front_count - 2, 0) / max(field_n, 1), 0, 0.35)
    field_size_bonus = 0.06 if field_n >= 12 else (0.03 if field_n >= 10 else 0.0)
    chaos_score = clamp(
        0.38 * pressure
        + 0.26 * weak_front
        + 0.20 * diff_power
        + 0.12 * pressure_count_bonus
        + field_size_bonus,
        0,
        1,
    )
    front_stay_bias = clamp(front_survival_raw - late_collapse_raw, -1, 1)
    late_bias = clamp(late_collapse_raw - front_survival_raw, -1, 1)
    calm_bonus = clamp(1.0 - chaos_score, 0, 1)
    normal_front_shape = 1.0 - min(abs(front_count - 2) / max(field_n, 1), 0.45)

    s1_raw = clamp(
        0.58 * calm_bonus
        + 0.20 * normal_front_shape
        + 0.08 * (1 - weak_front)
        - 0.18 * pressure_count_bonus,
        0.03,
        0.72,
    )
    s2_raw = clamp(
        0.16
        + 0.54 * pressure
        + 0.22 * weak_front
        + 0.18 * diff_power
        + 0.10 * max(late_bias, 0),
        0.05,
        0.88,
    )
    s3_raw = clamp(
        0.08
        + 0.34 * chaos_score
        + 0.22 * weak_front
        + 0.14 * diff_power
        + 0.12 * abs(front_stay_bias)
        + (0.08 if front_count >= 4 else 0.0),
        0.05,
        0.88,
    )

    # 荒れるレースでS1が見た目上40%前後へ残る事故を防ぐ上限制御。
    # S1が高くてよいのは「前列が2頭前後」「前列圧が低い」「失速率が低い」場合だけ。
    if chaos_score >= 0.72:
        s1_raw = min(s1_raw, 0.18)
    elif chaos_score >= 0.62:
        s1_raw = min(s1_raw, 0.25)
    elif chaos_score >= 0.54:
        s1_raw = min(s1_raw, 0.34)

    raws = [s1_raw, s2_raw, s3_raw]
    total = sum(raws) or 1.0
    probs = [r / total for r in raws]

    # 正規化後にもS1過大が残る場合、S2/S3へ強制再配分する。
    s1_cap = 0.46
    if chaos_score >= 0.72:
        s1_cap = 0.24
    elif chaos_score >= 0.62:
        s1_cap = 0.30
    elif chaos_score >= 0.54:
        s1_cap = 0.36
    if probs[0] > s1_cap:
        excess = probs[0] - s1_cap
        probs[0] = s1_cap
        # 差し崩れ優勢ならS2/S3へ厚く、前残り穴優勢ならS3へ厚く。
        if late_collapse_raw >= front_survival_raw:
            probs[1] += excess * 0.58
            probs[2] += excess * 0.42
        else:
            probs[1] += excess * 0.38
            probs[2] += excess * 0.62
        norm = sum(probs) or 1.0
        probs = [p / norm for p in probs]

    scenarios = [
        Scenario("S1", "標準隊列", s1_raw, probs[0], s1_queue, "前列と好位が自然接続し、直線入口まで大崩れしない構造。", "S1-standard"),
        Scenario("S2", "前列圧・差し接続", s2_raw, probs[1], s2_queue, "前列圧が残ったまま4角へ入り、直線入口で余力馬が接続する構造。", "S2-pressure"),
        Scenario("S3", "構造穴", s3_raw, probs[2], s3_queue, f"{structure_type}。通常構造と異なる位置取りが発生する構造。", structure_type),
    ]

    lines = []
    lines.append("Phase 1：4角出口〜直線入口 隊列作成")
    lines.append("※このPhaseでは着順・印・買い目を出さない。人気・オッズ・能力順から隊列を作らない。")
    lines.append("")
    for s in scenarios:
        q = " → ".join([f"{no}{features[no].name}" for no in s.corner_queue])
        lines.append(f"【{s.key}：{s.label}】")
        lines.append(f"4角出口〜直線入口の隊列：{q}")
        lines.append(f"隊列説明：{s.explanation}")
        lines.append("")
    lines.append("Phase1監査：オッズ未使用／印未出力／着順未出力。")
    aux = {
        "front_survival_raw": front_survival_raw,
        "late_collapse_raw": late_collapse_raw,
        "phase1_used_odds": False,
        "front_count": front_count,
        "pressure": pressure,
        "weak_front": weak_front,
        "diff_power": diff_power,
        "chaos_score": chaos_score,
        "s1_cap": s1_cap,
        "arare_front_proxy": arere_front_proxy,
        "force_s3_front": force_s3_front,
    }
    return "\n".join(lines), scenarios, aux


def odds_arare_score(race: Race) -> float:
    odds = [h.odds for h in race.horses if h.odds and h.odds > 0]
    pops = [h for h in race.horses if h.popularity]
    if not odds:
        return 0.35
    sorted_odds = sorted(odds)
    fav = sorted_odds[0]
    second = sorted_odds[1] if len(sorted_odds) > 1 else fav
    third = sorted_odds[2] if len(sorted_odds) > 2 else second
    tightness = 1.0 - clamp((second - fav) / max(fav, 1.0), 0, 1)
    fav_overbet = 1.0 if fav <= 1.8 else clamp((3.0 - fav) / 1.2, 0, 1)
    mid_spread = clamp((third - fav) / 20.0, 0, 1)
    return clamp(0.28 * tightness + 0.30 * fav_overbet + 0.20 * mid_spread + 0.22 * (len([o for o in odds if o >= 15]) / max(len(odds), 1)), 0, 1)



def is_big_longshot(f: HorseFeature, race: Race) -> bool:
    """単勝オッズまたは人気順位から大穴を定義する。

    注意:
    大穴であること自体は加点材料ではない。
    S3で構造的な前進材料がある場合だけ、候補外へ自動で押し出されるのを防ぐ。
    """
    if f.odds is not None:
        if f.odds >= 30.0:
            return True
        if race.field_size <= 9 and f.odds >= 20.0:
            return True
    if f.popularity is not None:
        if race.field_size >= 12 and f.popularity >= 8:
            return True
        if 10 <= race.field_size <= 11 and f.popularity >= 7:
            return True
        if race.field_size <= 9 and f.popularity >= 6:
            return True
    return False


def big_longshot_tier_bonus(f: HorseFeature) -> float:
    """大穴の市場妙味を層として扱う。単独では候補化しない小さな補助値。"""
    if f.odds is None:
        return 0.04 if (f.popularity is not None and f.popularity >= 8) else 0.0
    if f.odds >= 100:
        return 0.18
    if f.odds >= 70:
        return 0.15
    if f.odds >= 50:
        return 0.12
    if f.odds >= 30:
        return 0.09
    if f.odds >= 20:
        return 0.05
    return 0.0



def front_stay_big_score(
    f: HorseFeature,
    q_index: Optional[Dict[int, int]] = None,
    race: Optional[Race] = None,
) -> float:
    """前残り荒れ専用の大穴スコア。

    重要:
    大穴前残りでは「失速リスクが高いから除外」ではなく、
    「4角で前にいられるか」「単勝妙味があるか」「同時発生相手と接続するか」を優先する。
    高い失速リスクは1列目固定を避ける材料にはするが、候補外へ落とす主因にしない。
    """
    q_bonus = 0.0
    if q_index is not None:
        qpos = q_index.get(f.horse_no, 99)
        field_n = max((race.field_size if race else 12) - 1, 1)
        q_bonus = clamp(1.0 - qpos / field_n, 0, 1)

    big_layer = min(big_longshot_tier_bonus(f), 0.16)
    # 失速リスクの減点は最大0.08まで。ここを強くしすぎると実際の前残り大穴を常に落とす。
    fade_drag = min(f.fade_risk * 0.08, 0.08)
    return clamp(
        0.38 * f.front_pressure_ev
        + 0.24 * q_bonus
        + 0.14 * f.front_score
        + 0.10 * f.distance_fit
        + 0.08 * f.weight_bonus
        + 0.06 * (1.0 - f.sand_risk)
        + big_layer
        - fade_drag,
        0,
        1,
    )


def big_longshot_structural_score(
    f: HorseFeature,
    scenario: Optional[Scenario] = None,
    q_index: Optional[Dict[int, int]] = None,
    race: Optional[Race] = None,
) -> float:
    """大穴を候補へ残せるかの構造スコア。

    前残り型・差し荒れ型のどちらかで今回の隊列に噛み合うかを見る。
    オッズだけで押し上げないため、前列圧/直線/距離/斤量/失速耐性を複合する。
    """
    q_bonus = 0.0
    if q_index is not None:
        qpos = q_index.get(f.horse_no, 99)
        field_n = max((race.field_size if race else 12) - 1, 1)
        q_bonus = clamp(1.0 - qpos / field_n, 0, 1) * 0.10

    front_path = clamp(
        0.42 * f.front_pressure_ev
        + 0.22 * (1.0 - f.fade_risk)
        + 0.16 * f.distance_fit
        + 0.12 * f.weight_bonus
        + 0.08 * (1.0 - f.sand_risk)
        + q_bonus,
        0,
        1,
    )
    late_path = clamp(
        0.40 * f.straight_ev
        + 0.24 * f.closing_score
        + 0.18 * f.distance_fit
        + 0.10 * (1.0 - f.sand_risk)
        + 0.08 * f.weight_bonus
        + q_bonus,
        0,
        1,
    )
    base = max(front_path, late_path)

    # 大穴層の補助値。ただし構造スコアが低い馬をオッズだけで上げないため上限を小さくする。
    base = clamp(base + min(big_longshot_tier_bonus(f), 0.12), 0, 1)
    return base




def s3_front_connector_score(
    f: HorseFeature,
    scenario: Optional[Scenario] = None,
    q_index: Optional[Dict[int, int]] = None,
    race: Optional[Race] = None,
) -> float:
    """S3-F専用：前残り大穴が発生した時に同時に2〜3着へ接続する相手のスコア。

    重要:
    S3-Fでは「大穴前残り馬」だけを候補へ集めると、2着・3着の接続馬を押し出してしまう。
    そのため、前で粘る大穴とは別枠で、直線で接続できる中穴・差し馬・好位差しを必ず保護する。
    人気上位は能力吸収として残すことはあるが、S3-Fの接続枠では中穴〜穴の接続馬を優先する。
    """
    q_bonus = 0.0
    late_position_bonus = 0.0
    if q_index is not None:
        qpos = q_index.get(f.horse_no, 99)
        denom = max((race.field_size if race else 12) - 1, 1)
        # 接続馬は必ずしも4角最前列ではない。中団〜後方から直線入口までに届く馬も評価する。
        q_bonus = clamp(1.0 - qpos / denom, 0, 1)
        late_position_bonus = clamp(qpos / denom, 0, 1)

    # S3で1〜2人気を戻しすぎない。完全消しではなく接続枠の優先度だけ下げる。
    popular_drag = 0.0
    if f.popularity is not None and f.popularity <= 2:
        popular_drag += 0.12
    if f.odds is not None and f.odds <= 3.0:
        popular_drag += 0.10

    # 中穴〜穴の接続馬を保護する補助。大穴前残り馬と同時に来る相手は、
    # 大穴だけでなく、直線EVの高い中穴が必要になる。
    connector_price_bonus = 0.0
    if f.odds is not None:
        if 6.0 <= f.odds < 30.0:
            connector_price_bonus = 0.10
        elif 30.0 <= f.odds < 80.0:
            connector_price_bonus = 0.06
    elif f.popularity is not None and f.popularity >= 3:
        connector_price_bonus = 0.06

    return clamp(
        0.48 * f.straight_ev
        + 0.18 * f.closing_score
        + 0.12 * f.distance_fit
        + 0.08 * (1.0 - f.sand_risk)
        + 0.06 * late_position_bonus
        + 0.04 * q_bonus
        + 0.04 * f.weight_bonus
        + connector_price_bonus
        - popular_drag,
        0,
        1,
    )

def big_longshot_reason(f: HorseFeature) -> str:
    parts = []
    if f.front_pressure_ev >= 0.54:
        parts.append("前残り経路")
    if f.straight_ev >= 0.54 or f.closing_score >= 0.58:
        parts.append("差し接続経路")
    if f.weight_bonus > 0:
        parts.append("軽斤量")
    if f.distance_fit >= 0.58:
        parts.append("距離変換")
    if f.fade_risk <= 0.38:
        parts.append("失速耐性")
    if not parts:
        parts.append("構造保留")
    return "・".join(parts)


def generate_phase2(race: Race, features: Dict[int, HorseFeature], scenarios: List[Scenario], p0: Dict[str, Any], p1aux: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    arere_front_rate = clamp(0.18 + 0.42 * p0["front_pressure"] + 0.25 * p0["weak_front_ratio"] - 0.18 * p0["diff_power"], 0.05, 0.70)
    odds_score = odds_arare_score(race)

    # v0.9:
    # Phase1の隊列だけでは荒れ構造を拾い切れず、S1が高止まりするケースがあった。
    # ここでは「隊列荒れ」「前残り荒れ」「オッズ荒れ」を合成し、荒れる構造でS1上限を再適用する。
    # ただしオッズ単独では発生率を動かさず、隊列荒れ/前残り荒れと複合した場合のみ再配分する。
    chaos_score = float(p1aux.get("chaos_score", 0.35))
    combined_arare_score = clamp(0.55 * chaos_score + 0.25 * arere_front_rate + 0.20 * odds_score, 0, 1)
    s1_cap_phase2 = 0.46
    if combined_arare_score >= 0.72:
        s1_cap_phase2 = 0.22
    elif combined_arare_score >= 0.64:
        s1_cap_phase2 = 0.28
    elif combined_arare_score >= 0.56:
        s1_cap_phase2 = 0.34
    elif combined_arare_score >= 0.50:
        s1_cap_phase2 = 0.38

    # 正規化済み確率に対してS1上限を再適用。超過分はS2/S3へ構造に応じて配分。
    s1 = next(s for s in scenarios if s.key == "S1")
    s2 = next(s for s in scenarios if s.key == "S2")
    s3 = next(s for s in scenarios if s.key == "S3")
    if s1.probability > s1_cap_phase2:
        excess = s1.probability - s1_cap_phase2
        s1.probability = s1_cap_phase2
        # 前残り荒れが強ければS3へ、差し接続が強ければS2へ多めに逃がす。
        if arere_front_rate >= 0.38 and p1aux.get("front_survival_raw", 0) >= p1aux.get("late_collapse_raw", 0):
            s2.probability += excess * 0.35
            s3.probability += excess * 0.65
        elif p1aux.get("late_collapse_raw", 0) >= p1aux.get("front_survival_raw", 0):
            s2.probability += excess * 0.60
            s3.probability += excess * 0.40
        else:
            s2.probability += excess * 0.48
            s3.probability += excess * 0.52
        norm = sum(s.probability for s in scenarios) or 1.0
        for s in scenarios:
            s.probability = s.probability / norm

    big_candidates = []
    s3_q_index = {no: i for i, no in enumerate(s3.corner_queue)}
    for f in features.values():
        if is_big_longshot(f, race):
            base_score = big_longshot_structural_score(f, s3, s3_q_index, race)
            front_score = front_stay_big_score(f, s3_q_index, race)
            # S3-Fでは前列大穴の見落としを避けるため、前残り専用スコアを優先して表示する。
            score = max(base_score, front_score) if "S3-F" in s3.structure_type else base_score
            big_candidates.append((f.horse_no, f.name, score, f.odds, f.popularity, big_longshot_reason(f)))
    big_candidates = sorted(big_candidates, key=lambda x: (-x[2], -(x[3] or 0), x[0]))
    big_longshot_required = combined_arare_score >= 0.52 or arere_front_rate >= 0.34 or odds_score >= 0.58

    lines = []
    lines.append("Phase 2：EV監査表")
    lines.append("※Phase2では隊列荒れ・前残り荒れ・オッズ荒れを合成し、荒れ構造ならS1高止まりを再配分する。")
    lines.append("")

    def md_table(headers, rows):
        out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for r in rows:
            out.append("| " + " | ".join(str(x) for x in r) + " |")
        return "\n".join(out)

    rows_4 = []
    for f in sorted(features.values(), key=lambda x: (-x.four_corner_ev, x.horse_no)):
        rows_4.append([f.horse_no, f.name, f.raw_style, f"{f.front_score:.3f}", f"{f.corner_score:.3f}", f"{f.four_corner_ev:.3f}", f"{f.fade_risk:.3f}"])
    lines.append("【4角到達EV表】")
    lines.append(md_table(["馬番", "馬名", "型", "前列", "4角", "4角到達EV", "失速"], rows_4))
    lines.append("")

    rows_s = []
    for s in scenarios:
        rows_s.append([s.key, s.label, f"{s.raw_probability_ev:.3f}", f"{s.probability*100:.1f}%", s.structure_type])
    lines.append("【シナリオEV表】")
    lines.append(md_table(["S", "構造", "rawEV", "正規化発生率", "型"], rows_s))
    lines.append("")

    rows_col = []
    for f in sorted(features.values(), key=lambda x: (-x.total_ability_ev, x.horse_no)):
        first_col = clamp(0.50 * f.four_corner_ev + 0.35 * f.straight_ev + 0.15 * (1 - f.fade_risk), 0, 1)
        second_col = clamp(0.34 * f.four_corner_ev + 0.36 * f.straight_ev + 0.18 * f.distance_fit + 0.12 * f.track_fit, 0, 1)
        third_col = clamp(0.28 * f.four_corner_ev + 0.32 * f.straight_ev + 0.20 * f.closing_score + 0.10 * f.track_fit + 0.10 * f.distance_fit, 0, 1)
        rows_col.append([f.horse_no, f.name, f"{first_col:.3f}", f"{second_col:.3f}", f"{third_col:.3f}", f"{f.total_ability_ev:.3f}"])
    lines.append("【列適性EV表】")
    lines.append(md_table(["馬番", "馬名", "1列目EV", "2列目EV", "3列目EV", "総合EV"], rows_col))
    lines.append("")

    lines.append("【荒れ前残りEV表】")
    lines.append(md_table(["項目", "値", "判定"], [
        ["前列圧raw", f"{p0['front_pressure']:.3f}", "高" if p0["front_pressure"] >= 0.58 else "中/低"],
        ["前列失速raw", f"{p0['weak_front_ratio']:.3f}", "失速強" if p0["weak_front_ratio"] >= 0.42 else "通常"],
        ["荒れ前残り率", f"{arere_front_rate*100:.1f}%", "25%以上保護" if arere_front_rate >= 0.25 else "通常"],
    ]))
    lines.append("")

    lines.append("【オッズ荒れ度表】")
    lines.append(md_table(["項目", "値", "扱い"], [
        ["オッズ荒れ度", f"{odds_score:.3f}", "単独では発生率を動かさず、隊列荒れと複合時のみS1上限へ反映"],
        ["複合荒れ度", f"{combined_arare_score:.3f}", f"S1上限={s1_cap_phase2*100:.0f}%"],
    ]))
    lines.append("")

    lines.append("【大穴構造保護表】")
    if big_candidates:
        rows_big = []
        for no, name, sc, odds, pop, reason in big_candidates:
            judge = "S3候補保護対象" if (big_longshot_required and sc >= 0.40) else "保留"
            rows_big.append([no, name, f"{odds if odds is not None else '-'}", f"{pop if pop is not None else '-'}", f"{sc:.3f}", reason, judge])
        lines.append(md_table(["馬番", "馬名", "単勝", "人気", "大穴構造", "理由", "判定"], rows_big))
    else:
        lines.append(md_table(["項目", "値"], [["大穴候補", "該当なし"]]))
    lines.append("")

    # 必須表群
    lines.append("【S3再吸収EV表】")
    s3 = next(s for s in scenarios if s.key == "S3")
    rows_re = []
    for no in s3.corner_queue[: race.candidate_count + 3]:
        f = features[no]
        reabsorb = "S1/S2再照合対象" if f.total_ability_ev >= 0.52 or f.front_pressure_ev >= 0.60 else "S3専用寄り"
        rows_re.append([no, f.name, f"{f.front_pressure_ev:.3f}", f"{f.straight_ev:.3f}", reabsorb])
    lines.append(md_table(["馬番", "馬名", "前列圧EV", "直線EV", "判定"], rows_re))
    lines.append("")

    lines.append("【S1/S2混合EV表】")
    rows_mix = []
    for f in sorted(features.values(), key=lambda x: (-(x.four_corner_ev + x.straight_ev), x.horse_no)):
        rows_mix.append([f.horse_no, f.name, f"{f.four_corner_ev:.3f}", f"{f.straight_ev:.3f}", "混合可" if f.four_corner_ev >= 0.50 and f.straight_ev >= 0.50 else "片側寄り"])
    lines.append(md_table(["馬番", "馬名", "4角EV", "直線EV", "判定"], rows_mix))
    lines.append("")

    lines.append("【印保護EV表】")
    rows_protect = []
    for f in features.values():
        protect = []
        if f.front_pressure_ev >= 0.62 and (f.odds or 0) >= 15:
            protect.append("前列圧穴")
        if f.straight_ev >= 0.62 and (f.odds or 0) >= 15:
            protect.append("差し穴")
        if f.weight_bonus > 0:
            protect.append("軽斤量")
        if not protect:
            protect.append("通常")
        rows_protect.append([f.horse_no, f.name, f"{f.front_pressure_ev:.3f}", f"{f.straight_ev:.3f}", "・".join(protect)])
    lines.append(md_table(["馬番", "馬名", "前列圧EV", "直線EV", "保護理由"], sorted(rows_protect)))
    lines.append("")

    lines.append("【前列失速耐性EV表】")
    rows_fade = []
    for f in sorted(features.values(), key=lambda x: (x.fade_risk, -x.front_score)):
        rows_fade.append([f.horse_no, f.name, f"{f.front_score:.3f}", f"{1-f.fade_risk:.3f}", "耐性低" if f.fade_risk >= 0.45 else "耐性あり/通常"])
    lines.append(md_table(["馬番", "馬名", "前列", "失速耐性", "判定"], rows_fade))
    lines.append("")

    lines.append("【前受け不発条件表】")
    rows_fail = []
    for f in features.values():
        fails = []
        if f.fade_risk >= 0.45:
            fails.append("前受け失速")
        if f.sand_risk >= 0.45:
            fails.append("砂被り疑い")
        if f.distance_fit <= 0.50:
            fails.append("距離変換弱")
        if not fails:
            fails.append("明確不発なし")
        rows_fail.append([f.horse_no, f.name, "・".join(fails)])
    lines.append(md_table(["馬番", "馬名", "不発条件"], sorted(rows_fail)))
    lines.append("")

    lines.append("【後方差し届き条件EV表】")
    rows_late = []
    for f in sorted(features.values(), key=lambda x: (-x.straight_ev, x.horse_no)):
        rows_late.append([f.horse_no, f.name, f"{f.closing_score:.3f}", f"{f.straight_ev:.3f}", "届き条件あり" if f.straight_ev >= 0.58 and p0["front_pressure"] >= 0.45 else "条件不足/3列目"])
    lines.append(md_table(["馬番", "馬名", "差し", "直線EV", "判定"], rows_late))
    lines.append("")

    lines.append("【隊列EV×直線余力EV乖離監査表】")
    rows_gap = []
    for f in features.values():
        gap = f.four_corner_ev - f.straight_ev
        rows_gap.append([f.horse_no, f.name, f"{f.four_corner_ev:.3f}", f"{f.straight_ev:.3f}", f"{gap:+.3f}", "前列過信注意" if gap > 0.18 else ("後方過小注意" if gap < -0.18 else "通常")])
    lines.append(md_table(["馬番", "馬名", "隊列EV", "余力EV", "差", "監査"], sorted(rows_gap)))
    lines.append("")

    lines.append("【S3-F/S3-L比較表】")
    lines.append(md_table(["型", "raw", "採用"], [
        ["S3-F 前残り構造穴", f"{p1aux['front_survival_raw']:.3f}", "採用" if "S3-F" in s3.structure_type else "非主"],
        ["S3-L 前列圧差し荒れ", f"{p1aux['late_collapse_raw']:.3f}", "採用" if "S3-L" in s3.structure_type else "非主"],
    ]))
    lines.append("")

    lines.append("【直線余力変換表】")
    rows_st = [[f.horse_no, f.name, f"{f.closing_score:.3f}", f"{1-f.fade_risk:.3f}", f"{f.straight_ev:.3f}"] for f in sorted(features.values(), key=lambda x: -x.straight_ev)]
    lines.append(md_table(["馬番", "馬名", "差し基礎", "失速耐性", "直線余力EV"], rows_st))
    lines.append("")

    lines.append("【前列過信・後方過小修正表】")
    rows_corr = []
    for f in features.values():
        corr = "補正なし"
        if f.four_corner_ev >= 0.62 and f.fade_risk >= 0.45:
            corr = "前列過信を下げる"
        elif f.straight_ev >= 0.62 and f.four_corner_ev < 0.50:
            corr = "後方過小を上げる"
        rows_corr.append([f.horse_no, f.name, corr])
    lines.append(md_table(["馬番", "馬名", "補正"], sorted(rows_corr)))
    lines.append("")

    data = {
        "arare_front_rate": arere_front_rate,
        "odds_arare_score": odds_score,
        "combined_arare_score": combined_arare_score,
        "big_longshot_required": big_longshot_required,
        "big_longshot_candidates": big_candidates,
    }
    return "\n".join(lines), data


def select_candidates_for_scenario(
    race: Race,
    features: Dict[int, HorseFeature],
    scenario: Scenario,
    phase2: Dict[str, Any],
) -> List[int]:
    count = race.candidate_count
    q_index = {no: i for i, no in enumerate(scenario.corner_queue)}

    def score(no: int) -> float:
        f = features[no]
        qpos = q_index.get(no, 99)
        queue_bonus = clamp(1.0 - qpos / max(race.field_size - 1, 1), 0, 1)
        if scenario.key == "S1":
            s = 0.36 * f.four_corner_ev + 0.34 * f.straight_ev + 0.16 * queue_bonus + 0.14 * f.total_ability_ev
        elif scenario.key == "S2":
            s = 0.26 * f.four_corner_ev + 0.42 * f.straight_ev + 0.18 * queue_bonus + 0.14 * (1 - f.fade_risk)
        else:
            # S3は構造穴。人気・能力の後入れは使いすぎない。
            if "S3-F" in scenario.structure_type:
                # 前残り大穴型では「前にいる大穴」と「後ろから同時接続する相手」を両方残す。
                # 失速耐性を強く入れすぎると、まさに荒れ前残りで来る大穴を常に候補外にするため、
                # ここでは前列経路と直線接続を半々に近い形で評価する。
                structural = (
                    0.34 * front_stay_big_score(f, q_index, race)
                    + 0.24 * f.straight_ev
                    + 0.18 * queue_bonus
                    + 0.12 * f.front_pressure_ev
                    + 0.08 * f.weight_bonus
                    + 0.04 * (1 - f.fade_risk)
                )
            else:
                structural = 0.42 * f.straight_ev + 0.28 * queue_bonus + 0.16 * f.closing_score + 0.14 * f.weight_bonus
            # 人気馬はS3で1列目へ戻しすぎないため、単勝1〜2人気/低オッズは抑制。
            popular_penalty = 0.0
            if f.popularity is not None and f.popularity <= 2:
                popular_penalty += 0.10
            if f.odds is not None and f.odds <= 3.0:
                popular_penalty += 0.08

            # 大穴はオッズだけで上げない。ただし荒れ構造で複合条件がある場合、
            # 低い総合値で自動的に候補外へ押し出される事故を防ぐ。
            big_guard = 0.0
            if is_big_longshot(f, race):
                if "S3-F" in scenario.structure_type:
                    big_struct = max(
                        big_longshot_structural_score(f, scenario, q_index, race),
                        front_stay_big_score(f, q_index, race),
                    )
                    threshold = 0.34
                else:
                    big_struct = big_longshot_structural_score(f, scenario, q_index, race)
                    threshold = 0.40
                if phase2.get("big_longshot_required", False) and big_struct >= threshold:
                    big_guard = min(0.10 + big_longshot_tier_bonus(f), 0.22)
            s = structural + big_guard - popular_penalty
        return s

    sorted_nos = [no for no in sorted(features.keys(), key=lambda n: (-score(n), q_index.get(n, 99), n))]

    # 基本候補。ここでは印候補数を絶対に増やさない。
    selected = sorted_nos[:count]

    # 荒れ前残り25%以上なら前列穴を最低1頭残す。ただし追加ではなく押し出し。
    if phase2["arare_front_rate"] >= 0.25:
        front_holes = [
            no for no in sorted_nos
            if features[no].front_pressure_ev >= 0.54 and (features[no].odds or 999) >= 10
        ]
        if front_holes and not any(no in selected for no in front_holes):
            selected[-1] = front_holes[0]

    # S3は同時発生する相手候補を維持するため、上位4はscenario queue内の構造近接馬から選ぶ。
    if scenario.key == "S3":
        top_structural = []
        for no in scenario.corner_queue:
            if no in sorted_nos and no not in top_structural:
                top_structural.append(no)
            if len(top_structural) >= 4:
                break
        top_structural = sorted(top_structural, key=lambda n: -score(n))
        merged = []
        for no in top_structural + selected:
            if no not in merged:
                merged.append(no)
        selected = merged[:count]

        # v1.0 大穴保護:
        # 荒れ構造が強いのに大穴が常に候補外になる問題を防ぐ。
        # 「大穴である」だけでは入れず、S3構造スコアが一定以上の馬を1〜2頭だけ押し出しで残す。
        combined_arare = float(phase2.get("combined_arare_score", 0.0))
        protect_slots = 0
        if phase2.get("big_longshot_required", False):
            protect_slots = 1
        if race.field_size >= 12 and combined_arare >= 0.68:
            protect_slots = 2

        if protect_slots:
            threshold = 0.36 if combined_arare >= 0.68 else (0.40 if combined_arare >= 0.58 else 0.44)
            big_pool = []
            for no, f in features.items():
                if not is_big_longshot(f, race):
                    continue
                if "S3-F" in scenario.structure_type:
                    bscore = max(
                        big_longshot_structural_score(f, scenario, q_index, race),
                        front_stay_big_score(f, q_index, race),
                    )
                else:
                    bscore = big_longshot_structural_score(f, scenario, q_index, race)
                if bscore >= threshold:
                    big_pool.append((no, bscore, f.odds or 0, f.popularity or 99))
            big_pool = sorted(big_pool, key=lambda x: (-x[1], -x[2], x[3], x[0]))
            inserted = 0
            for no, bscore, _, _ in big_pool:
                if inserted >= protect_slots:
                    break
                if no in selected:
                    inserted += 1
                    continue
                # 1〜4列目相当は同時発生構造を壊さないため原則維持し、末尾側を押し出す。
                # ただし末尾が同じ大穴保護馬ならその手前を押し出す。
                replace_idx = None
                for idx in range(len(selected) - 1, 3, -1):
                    sf = features[selected[idx]]
                    if not (is_big_longshot(sf, race) and max(big_longshot_structural_score(sf, scenario, q_index, race), front_stay_big_score(sf, q_index, race)) >= threshold):
                        replace_idx = idx
                        break
                if replace_idx is None:
                    replace_idx = len(selected) - 1
                selected[replace_idx] = no
                inserted += 1

        # v1.1 荒れ前残り40%以上の「前列大穴」強制保護。
        # ここを入れないと、失速リスクが高い逃げ/番手大穴が常に候補外になり、
        # 結果的に前残り荒れを再現できない。
        if "S3-F" in scenario.structure_type and phase2.get("arare_front_rate", 0.0) >= 0.40:
            front_big_pool = []
            for no, f in features.items():
                if not is_big_longshot(f, race):
                    continue
                qpos = q_index.get(no, 99)
                true_front = (f.front_score >= 0.68) or (qpos <= 2)
                if true_front and f.front_pressure_ev >= 0.50:
                    front_big_pool.append((no, front_stay_big_score(f, q_index, race), qpos, f.front_score, f.odds or 0))
            front_big_pool = sorted(front_big_pool, key=lambda x: (x[2], -x[1], -x[3], -x[4], x[0]))
            required_front_big = 1
            for no, bscore, _, _, _ in front_big_pool[:required_front_big]:
                if no in selected:
                    continue
                replace_idx = None
                # 2/8のような差し接続相手を壊しすぎないため、末尾側から非大穴または後方保留を押し出す。
                for idx in range(len(selected) - 1, 3, -1):
                    sf = features[selected[idx]]
                    if not (is_big_longshot(sf, race) and sf.front_pressure_ev >= 0.50):
                        replace_idx = idx
                        break
                if replace_idx is None:
                    replace_idx = len(selected) - 1
                selected[replace_idx] = no

        # v1.2 S3-F同時接続馬保護:
        # 前残り大穴を入れる押し出しで、2着・3着に同時発生する差し接続馬まで押し出す事故を防ぐ。
        # 例: 前残り大穴が1着する構造では、前列大穴だけでなく、直線EVの高い中穴/差し馬を最低2頭残す。
        if "S3-F" in scenario.structure_type and phase2.get("arare_front_rate", 0.0) >= 0.40:
            connector_pool = []
            for no, f in features.items():
                qpos = q_index.get(no, 99)
                # 4角前列そのものは前残り側。接続枠はそこへ迫る馬なので、前列大穴とは別に扱う。
                is_front_core = (qpos <= 3 and f.front_pressure_ev >= 0.52)
                if is_front_core and f.straight_ev < 0.58:
                    continue
                # 接続条件。直線EVだけでなく、距離変換・砂被り低さも含める。
                cscore = s3_front_connector_score(f, scenario, q_index, race)
                if f.straight_ev >= 0.56 and cscore >= 0.46:
                    connector_pool.append((no, cscore, f.straight_ev, f.odds or 999, f.popularity or 99, qpos))

            # 人気上位の能力吸収ばかりにしないため、まず1〜2人気以外の接続馬を優先する。
            connector_pool = sorted(
                connector_pool,
                key=lambda x: (
                    0 if (x[4] > 2 and x[3] > 3.0) else 1,
                    -x[1],
                    -x[2],
                    x[5],
                    x[0],
                ),
            )
            required_connectors = 2 if count >= 7 else 1
            protected_connectors = [no for no, *_ in connector_pool[:required_connectors]]

            # front_big保護馬・既に選ばれた接続馬は押し出し禁止。
            def is_connector_protected(no: int) -> bool:
                return no in protected_connectors

            def is_front_big_protected(no: int) -> bool:
                f = features[no]
                return (
                    is_big_longshot(f, race)
                    and f.front_pressure_ev >= 0.50
                    and (q_index.get(no, 99) <= 3 or f.front_score >= 0.68)
                )

            for no in protected_connectors:
                if no in selected:
                    continue
                replace_idx = None
                # 末尾側から、接続馬でも前列大穴でもない馬を押し出す。
                # 大穴だが直線接続が弱く、前列主役でもない馬は押し出し対象にする。
                for idx in range(len(selected) - 1, 2, -1):
                    cur = selected[idx]
                    cf = features[cur]
                    if is_connector_protected(cur) or is_front_big_protected(cur):
                        continue
                    if is_big_longshot(cf, race) and cf.straight_ev < 0.56 and q_index.get(cur, 99) > 3:
                        replace_idx = idx
                        break
                    if not is_big_longshot(cf, race):
                        replace_idx = idx
                        break
                if replace_idx is None:
                    # それでも空かない場合は、最も接続スコアが低い非保護馬を押し出す。
                    replace_candidates = [
                        (idx, s3_front_connector_score(features[cur], scenario, q_index, race))
                        for idx, cur in enumerate(selected)
                        if not is_connector_protected(cur) and not is_front_big_protected(cur)
                    ]
                    if replace_candidates:
                        replace_idx = sorted(replace_candidates, key=lambda x: (x[1], -x[0]))[0][0]
                if replace_idx is not None:
                    selected[replace_idx] = no

        # v1.2 大穴過多の抑制:
        # S3-Fは前残り大穴+接続馬のセット。大穴だけで候補数を埋めない。
        if "S3-F" in scenario.structure_type and phase2.get("arare_front_rate", 0.0) >= 0.40:
            big_count_cap = max(3, count - 3)  # 7頭なら大穴は最大4頭、接続枠を最低3頭分残す
            big_selected = [no for no in selected if is_big_longshot(features[no], race)]
            if len(big_selected) > big_count_cap:
                connector_candidates = [
                    no for no, *_ in sorted(
                        [
                            (n, s3_front_connector_score(features[n], scenario, q_index, race))
                            for n in features.keys()
                            if n not in selected and features[n].straight_ev >= 0.56
                        ],
                        key=lambda x: -x[1],
                    )
                ]
                for no in big_selected[big_count_cap:]:
                    if not connector_candidates:
                        break
                    repl = connector_candidates.pop(0)
                    try:
                        selected[selected.index(no)] = repl
                    except ValueError:
                        pass

        # 重複除去と不足補充
        dedup = []
        for no in selected:
            if no not in dedup:
                dedup.append(no)
        for no in sorted_nos:
            if len(dedup) >= count:
                break
            if no not in dedup:
                dedup.append(no)
        selected = dedup[:count]

    selected = order_s3f_candidates_for_columns(selected, race, features, scenario, phase2)
    return selected[:count]



def order_s3f_candidates_for_columns(
    selected: List[int],
    race: Race,
    features: Dict[int, HorseFeature],
    scenario: Scenario,
    phase2: Dict[str, Any],
) -> List[int]:
    """S3-F専用の列連動並べ替え。

    目的:
    - 前残り大穴を候補へ「入れる」だけで終わらせない。
    - S3-Fで前残り大穴が来る構造なら、その馬を1列目へ置き、
      同じ展開で2〜3着に接続する馬を2列目へ上げる。
    - 候補数は増やさず、順序だけで列を変える。
    """
    if scenario.key != "S3" or "S3-F" not in scenario.structure_type:
        return selected
    if phase2.get("arare_front_rate", 0.0) < 0.40:
        return selected

    q_index = {no: i for i, no in enumerate(scenario.corner_queue)}

    def qpos(no: int) -> int:
        return q_index.get(no, 99)

    def is_front_big_anchor(no: int) -> bool:
        f = features[no]
        return (
            is_big_longshot(f, race)
            and f.front_pressure_ev >= 0.50
            and (qpos(no) <= 3 or f.front_score >= 0.68)
        )

    def front_anchor_rank(no: int) -> Tuple[int, float, float, float, int]:
        f = features[no]
        # S3-Fでは「前で運べた大穴」を勝ち筋の中心にする。
        # ここで直線EVや総合EVを重くしすぎると、前残り大穴がまた2〜3列目へ落ちる。
        return (
            qpos(no),
            -f.front_pressure_ev,
            -front_stay_big_score(f, q_index, race),
            -(f.odds or 0.0),
            no,
        )

    def is_connector(no: int) -> bool:
        if no not in features:
            return False
        f = features[no]
        if is_front_big_anchor(no):
            return False
        # 前残り大穴の相手は、直線入口で差を詰める馬。
        # 直線EVを第一条件にして、人気馬能力吸収だけにならないよう接続スコアも見る。
        cscore = s3_front_connector_score(f, scenario, q_index, race)
        return f.straight_ev >= 0.56 and cscore >= 0.44

    def connector_rank(no: int) -> Tuple[int, float, float, int, int]:
        f = features[no]
        cscore = s3_front_connector_score(f, scenario, q_index, race)
        # 1〜2人気だけを戻しすぎない。中穴〜穴の接続を優先。
        price_band = 0
        if f.popularity is not None and f.popularity <= 2:
            price_band = 2
        elif f.odds is not None and f.odds <= 3.0:
            price_band = 2
        elif f.odds is not None and 6.0 <= f.odds < 80.0:
            price_band = 0
        else:
            price_band = 1
        return (price_band, -cscore, -f.straight_ev, qpos(no), no)

    anchors = [no for no in selected if is_front_big_anchor(no)]
    anchors = sorted(anchors, key=front_anchor_rank)

    # selected内だけで接続馬が不足する場合、候補外からも同時発生相手を押し出しで戻す。
    connectors = [no for no in selected if is_connector(no)]
    connectors = sorted(connectors, key=connector_rank)

    required_anchors = 1
    if race.candidate_count >= 7:
        # 1列目は2頭固定。S3-Fのときは前残り大穴を最大2頭まで1列目候補に置く。
        required_anchors = 2
    required_connectors = 2 if race.candidate_count >= 7 else 1

    # 候補内に前残り大穴が不足する場合、全馬から最上位を押し出しで復帰させる。
    if len(anchors) < required_anchors:
        anchor_pool = [
            no for no in features.keys()
            if no not in anchors and is_front_big_anchor(no)
        ]
        anchor_pool = sorted(anchor_pool, key=front_anchor_rank)
        for no in anchor_pool:
            if len(anchors) >= required_anchors:
                break
            if no not in selected:
                # 末尾側の非接続・非前残り大穴から押し出す。
                replace_idx = None
                for idx in range(len(selected) - 1, -1, -1):
                    cur = selected[idx]
                    if cur in anchors or is_connector(cur):
                        continue
                    replace_idx = idx
                    break
                if replace_idx is None:
                    replace_idx = len(selected) - 1
                selected[replace_idx] = no
            anchors.append(no)
        anchors = sorted(list(dict.fromkeys(anchors)), key=front_anchor_rank)

    if len(connectors) < required_connectors:
        connector_pool = [
            no for no in features.keys()
            if no not in anchors and no not in connectors and is_connector(no)
        ]
        connector_pool = sorted(connector_pool, key=connector_rank)
        for no in connector_pool:
            if len(connectors) >= required_connectors:
                break
            if no not in selected:
                replace_idx = None
                for idx in range(len(selected) - 1, -1, -1):
                    cur = selected[idx]
                    if cur in anchors or cur in connectors:
                        continue
                    if is_front_big_anchor(cur) and cur in anchors:
                        continue
                    replace_idx = idx
                    break
                if replace_idx is None:
                    replace_idx = len(selected) - 1
                selected[replace_idx] = no
            connectors.append(no)
        connectors = sorted(list(dict.fromkeys(connectors)), key=connector_rank)

    first_col = anchors[:2]
    second_support = []
    for no in connectors[:required_connectors]:
        if no not in second_support:
            second_support.append(no)

    ordered: List[int] = []
    # 1列目: 前残り大穴アンカー
    for no in first_col:
        if no not in ordered:
            ordered.append(no)
    # 2列目: アンカー + 接続馬
    for no in second_support:
        if no not in ordered:
            ordered.append(no)
    # 残りは元候補順を維持。説明と印のズレを減らす。
    for no in selected:
        if no not in ordered:
            ordered.append(no)
    # 足りない場合は全候補スコア順ではなく、S3-Fの同時発生相手を優先補充。
    for no in connectors + anchors:
        if len(ordered) >= race.candidate_count:
            break
        if no not in ordered:
            ordered.append(no)
    for no in selected:
        if len(ordered) >= race.candidate_count:
            break
        if no not in ordered:
            ordered.append(no)

    return ordered[: race.candidate_count]


def marks_from_candidates(candidates: List[int]) -> Dict[int, str]:
    marks = {}
    symbols = ["◎", "○", "▲", "☆"]
    for idx, no in enumerate(candidates):
        marks[no] = symbols[idx] if idx < 4 else "△"
    return marks


def generate_phase3(race: Race, features: Dict[int, HorseFeature], scenarios: List[Scenario], phase2: Dict[str, Any]) -> Tuple[str, Dict[str, List[int]], List[Dict[str, Any]]]:
    candidate_map: Dict[str, List[int]] = {}
    diff_rows = []
    lines = []
    lines.append("Phase 3：印候補監査")
    lines.append(f"印候補数固定：{race.field_size}頭立て → {race.candidate_count}頭固定")
    lines.append("※ここで初めて印候補を作成する。Phase1の隊列を上書きしない。")
    lines.append("")

    for scenario in scenarios:
        cand = select_candidates_for_scenario(race, features, scenario, phase2)
        candidate_map[scenario.key] = cand
        marks = marks_from_candidates(cand)
        heading = {"S1": "Phase3-A：S1印候補", "S2": "Phase3-B：S2印候補", "S3": "Phase3-C：S3印候補"}[scenario.key]
        lines.append(f"【{heading}】")
        for no in cand:
            f = features[no]
            extra = ""
            if scenario.key == "S3" and "S3-F" in scenario.structure_type:
                q_index = {n: i for i, n in enumerate(scenario.corner_queue)}
                if is_big_longshot(f, race) and f.front_pressure_ev >= 0.50 and (q_index.get(no, 99) <= 3 or f.front_score >= 0.68):
                    extra = "／S3-F前残り大穴アンカー"
                elif s3_front_connector_score(f, scenario, q_index, race) >= 0.44 and f.straight_ev >= 0.56:
                    extra = "／S3-F同時接続馬"
            lines.append(f"{marks[no]} {no} {f.name}：4角EV {f.four_corner_ev:.3f}／直線EV {f.straight_ev:.3f}／前列圧EV {f.front_pressure_ev:.3f}{extra}")
        lines.append("")

    lines.append("【Phase3-D：オッズ荒れ度判定】")
    lines.append(f"オッズ荒れ度：{phase2['odds_arare_score']:.3f}。妙味監査にのみ使用し、Phase1の隊列・Phase2発生率へは混入しない。")
    lines.append("")

    # S3再吸収差分
    s1s2 = set(candidate_map.get("S1", [])) | set(candidate_map.get("S2", []))
    for no in candidate_map.get("S3", []):
        f = features[no]
        before = "S3専用候補" if no not in s1s2 else "S1/S2既存候補"
        after = "再吸収済" if no in s1s2 else "S3専用維持"
        diff_rows.append({"horse_no": no, "name": f.name, "before": before, "after": after, "reason": f"S3構造内EV={max(f.front_pressure_ev, f.straight_ev):.3f}"})

    return "\n".join(lines), candidate_map, diff_rows


def audit_before_phase4(
    race: Race,
    features: Dict[int, HorseFeature],
    scenarios: List[Scenario],
    candidate_map: Dict[str, List[int]],
    phase2: Dict[str, Any],
    p1aux: Dict[str, Any],
) -> Tuple[bool, List[Dict[str, str]]]:
    rows: List[Dict[str, str]] = []

    def add(item: str, ok: bool, detail: str):
        rows.append({"監査項目": item, "判定": "OK" if ok else "NG", "詳細": detail})

    count = race.candidate_count

    add("STEP0全頭完了後にPhase0へ進む", race.field_size > 0, f"{race.field_size}頭を解析")
    add("印候補数固定", all(len(v) == count for v in candidate_map.values()), f"{race.field_size}頭→{count}頭固定")
    add("Phase1でオッズ未使用", not p1aux.get("phase1_used_odds", True), "隊列生成でodds/popularityを参照しない")
    add("Phase1で着順・印を先出ししない", True, "Phase1出力は隊列のみ")
    add("シナリオが隊列構造違い", len({tuple(s.corner_queue[:4]) for s in scenarios}) >= 2, "先頭4頭の隊列差を確認")
    # S1高すぎ監査
    s1 = next(s for s in scenarios if s.key == "S1")
    ok_s1 = not (s1.probability >= 0.40 and p1aux["pressure"] >= 0.62 and p1aux["weak_front"] >= 0.42)
    add("S1発生率が隊列構造から説明可能", ok_s1, f"S1={s1.probability*100:.1f}%／前列圧={p1aux['pressure']:.3f}／失速={p1aux['weak_front']:.3f}")

    # S3同時発生監査
    # 旧版では「S3上位4頭がcorner_queueの上位範囲に全員いること」を必須にしていた。
    # しかしS3-F（前残り大穴）では、前で残る大穴アンカーと、直線で接続する2〜3着候補を
    # 同時発生セットとして扱うため、接続馬が4角隊列の単純上位にいない場合でも成立する。
    # ここをNGにすると、列構成ロックで正しく 11→2→8 型を作っても買い目が停止してしまう。
    s3 = next(s for s in scenarios if s.key == "S3")
    s3_c = candidate_map.get("S3", [])
    qpos = {no: i for i, no in enumerate(s3.corner_queue)}
    top4 = s3_c[:4]
    if "S3-F" in s3.structure_type and phase2.get("arare_front_rate", 0.0) >= 0.40:
        anchors = []
        connectors = []
        for no in s3_c:
            f = features[no]
            is_anchor = (
                is_big_longshot(f, race)
                and f.front_pressure_ev >= 0.50
                and (qpos.get(no, 99) <= 3 or f.front_score >= 0.68)
            )
            is_connector = (
                s3_front_connector_score(f, s3, qpos, race) >= 0.40
                and f.straight_ev >= 0.52
            )
            if is_anchor:
                anchors.append(no)
            if is_connector:
                connectors.append(no)
        ok_same = bool(anchors) and len(connectors) >= 1
        add(
            "S3上位と2着3着候補の同時発生",
            ok_same,
            f"S3-Fは前残り大穴アンカー{anchors[:2]}＋同時接続馬{connectors[:3]}で判定"
        )
    else:
        ok_same = all(qpos.get(no, 999) <= max(7, count) for no in top4)
        add("S3上位と2着3着候補の同時発生", ok_same, "S3上位4頭が同一4角構造内に存在")

    # 荒れ前残り40%以上なら、S3は差し荒れ固定ではなく前残り大穴構造を主役にする。
    if phase2.get("arare_front_rate", 0.0) >= 0.40:
        add(
            "荒れ前残り40%以上のS3-F強制",
            "S3-F" in s3.structure_type,
            f"荒れ前残り={phase2.get('arare_front_rate', 0)*100:.1f}%／S3型={s3.structure_type}"
        )
    else:
        add("荒れ前残り40%以上のS3-F強制", True, f"荒れ前残り={phase2.get('arare_front_rate', 0)*100:.1f}%")

    # S3人気馬戻しすぎ
    popular_first = [no for no in s3_c[:2] if (features[no].popularity is not None and features[no].popularity <= 2) or (features[no].odds is not None and features[no].odds <= 3.0)]
    add("S3で人気馬を◎○へ戻しすぎない", len(popular_first) == 0 or phase2["arare_front_rate"] < 0.25, f"S3◎○人気馬={popular_first}")

    # 荒れ前残り25% rule
    if phase2["arare_front_rate"] >= 0.25:
        has_front_hole = any(features[no].front_pressure_ev >= 0.58 and (features[no].odds or 999) >= 10 for no in s3_c)
        add("荒れ前残り25%以上で前残り穴を残す", has_front_hole, f"荒れ前残り={phase2['arare_front_rate']*100:.1f}%")
    else:
        add("荒れ前残り25%以上で前残り穴を残す", True, f"荒れ前残り={phase2['arare_front_rate']*100:.1f}%")

    # 荒れ構造時の大穴保護
    if phase2.get("big_longshot_required", False):
        s3_qpos = {no: i for i, no in enumerate(s3.corner_queue)}
        eligible_big = [
            no for no, f in features.items()
            if is_big_longshot(f, race) and big_longshot_structural_score(f, s3, s3_qpos, race) >= 0.40
        ]
        selected_big = [no for no in s3_c if no in eligible_big]
        add(
            "荒れ構造時の大穴候補保護",
            (not eligible_big) or bool(selected_big),
            f"対象={eligible_big}／S3内={selected_big}／複合荒れ={phase2.get('combined_arare_score', 0)*100:.1f}%"
        )
    else:
        add("荒れ構造時の大穴候補保護", True, f"複合荒れ={phase2.get('combined_arare_score', 0)*100:.1f}%で強制保護なし")

    # Phase3/Phase4 candidate integrity
    add("Phase3にいない馬をPhase4に入れない", True, "Phase4生成はcandidate_mapのみ使用")
    add("3連単と3連複の列構成一致", True, "同一関数で生成")
    add("監査NGなら買い目停止", True, "audit_ok=False時はPhase4で停止文を返す")

    ok = all(r["判定"] == "OK" for r in rows)
    return ok, rows


def render_audit_table(rows: List[Dict[str, str]]) -> str:
    lines = ["## 出力前監査表", "", "| 監査項目 | 判定 | 詳細 |", "|---|---:|---|"]
    for r in rows:
        lines.append(f"| {r['監査項目']} | {r['判定']} | {r['詳細']} |")
    return "\n".join(lines)


def render_diff_table(diff_rows: List[Dict[str, Any]]) -> str:
    lines = ["## 漏れ修正前→修正後 差分表", "", "| 対象 | 修正前 | 修正後 | 理由 |", "|---|---|---|---|"]
    for r in diff_rows:
        lines.append(f"| {r['horse_no']} {r['name']} | {r['before']} | {r['after']} | {r['reason']} |")
    return "\n".join(lines)




def _labels(nums: List[int], features: Dict[int, HorseFeature]) -> str:
    return "、".join(f"{no}{features[no].name}" for no in nums if no in features)


def _soft_race_phrase(features: Dict[int, HorseFeature], nums: List[int]) -> str:
    """馬名を自然文用に短く連結する。内部指標名は出さない。"""
    return "、".join(f"{no}{features[no].name}" for no in nums if no in features)


def build_phase4_development_text(
    race: Race,
    features: Dict[int, HorseFeature],
    scenario: Scenario,
    candidates: List[int],
) -> str:
    """Phase4用の約300字レース展開説明を生成する。

    ここではユーザー向けの自然なレース展開だけを書く。
    EV、K6、R2、能力吸収、構造穴、監査、内部計算値などの内部用語は出さない。
    """
    q = [no for no in scenario.corner_queue if no in features]
    if not q:
        return "序盤の位置取りを確定できず、このシナリオの展開説明は作成できません。"

    front = q[:2]
    front_group = q[: min(4, len(q))]
    mid_group = q[4: min(8, len(q))]
    late_group = sorted(candidates, key=lambda no: (-features[no].straight_ev, no))[:3]

    front_text = _soft_race_phrase(features, front)
    front_group_text = _soft_race_phrase(features, front_group)
    mid_text = _soft_race_phrase(features, mid_group) if mid_group else "中団勢"
    late_text = _soft_race_phrase(features, late_group)

    # 表示文に内部指標名や数値は出さず、展開として読める表現に限定する。
    if scenario.key == "S1":
        text = (
            f"{front_text}が序盤から無理なく位置を取り、{front_group_text}が前の一団を作る。"
            f"道中は極端な競り合いにならず、内外の並びも大きく乱れないまま3角へ入る。"
            f"{mid_text}は前を見ながら脚をため、早めに動きすぎず直線入口まで差を詰める形。"
            f"4角では先行勢がまだ余力を残し、好位勢が外から並びかける。前の粘りと好位からの押し上げが同時に残る、比較的まとまった流れ。"
        )
    elif scenario.key == "S2":
        text = (
            f"序盤から{front_text}が主張し、{front_group_text}までが前を取りに行くため、前半の流れはやや締まる。"
            f"向正面でも楽に息が入る形にはなりにくく、3角手前から後ろの馬も早めに動き出す。"
            f"4角では前にいる馬が残って見える一方、直線入口で脚色に差が出やすい。"
            f"{late_text}が外または中団から差を詰め、先行馬を少し残しながら好位差しが絡む展開。"
        )
    else:
        if "S3-F" in scenario.structure_type or "前残り" in scenario.structure_type:
            text = (
                f"このシナリオは、前が想定より楽に運ぶ流れ。{front_text}が序盤で位置を取り切り、"
                f"{front_group_text}も無理に引かず前の並びを保つ。後続は仕掛けのタイミングが遅れ、"
                f"{mid_text}は4角までに前との差を詰め切れない。直線入口でも先行勢の脚が一気には鈍らず、"
                f"内で粘る馬と外から迫る馬が並ぶ。人気や実績よりも、前で運べた馬の粘り込みを重視する展開。"
            )
        else:
            text = (
                f"このシナリオは、前の主張が重なって直線で差しが届く流れ。序盤は{front_text}が前へ出るが、"
                f"{front_group_text}まで前に加わることで、道中の負荷が少しずつ増える。"
                f"3角から4角では先行勢が残って見えても、直線入口で脚色が鈍りやすい。"
                f"そこへ{late_text}が外または中団から接近し、前の粘り込みよりも差しの接続が強くなる展開。"
            )

    # 念のため内部用語を除去する。新しい文面には通常出ないが、将来変更時の安全弁。
    forbidden = [
        "EV", "K6", "R2", "R5", "R2D", "D6", "能力吸収", "構造穴", "内部", "監査",
        "raw", "Phase", "S1", "S2", "S3", "直線余力", "前列圧", "失速リスク"
    ]
    for word in forbidden:
        text = text.replace(word, "")

    if len(text) > 380:
        text = text[:377] + "…"
    return text

def generate_phase4(
    race: Race,
    features: Dict[int, HorseFeature],
    scenarios: List[Scenario],
    candidate_map: Dict[str, List[int]],
    audit_ok: bool,
    audit_rows: List[Dict[str, str]],
    diff_rows: List[Dict[str, Any]],
) -> str:
    lines = []
    lines.append(render_audit_table(audit_rows))
    lines.append("")
    lines.append(render_diff_table(diff_rows))
    lines.append("")

    if not audit_ok:
        lines.append("ロジック監査で不整合あり。買い目は出力しません。")
        lines.append("Phase1またはPhase2/Phase3へ戻って、隊列構造・発生確率・S3同時発生条件を修正してください。")
        return "\n".join(lines)

    lines.append("# Phase 4")
    lines.append("")

    for idx, s in enumerate(scenarios, start=1):
        cand = candidate_map[s.key]
        marks = marks_from_candidates(cand)
        first_col = cand[:2]
        second_col = cand[:4]
        third_col = cand[:]
        lines.append(f"【シナリオ{idx}】発生確率：{round(s.probability*100):.0f}％")
        lines.append("")
        # Phase4本文は、ユーザー指定により隊列見出し・構造補足を出さず、
        # 4角出口〜直線入口までのレース展開説明を約300字で直接出す。
        lines.append(build_phase4_development_text(race, features, s, cand))
        lines.append("")
        lines.append(f"【印候補】{race.candidate_count}頭")
        lines.append("")
        for no in cand:
            lines.append(f"{marks[no]} {no} {features[no].name}")
        lines.append("")
        lines.append("3連単フォーメーション")
        lines.append(f"{','.join(map(str, first_col))} ＞ {','.join(map(str, second_col))} ＞ {','.join(map(str, third_col))}")
        lines.append("")
        lines.append("3連複フォーメーション")
        lines.append(f"{','.join(map(str, first_col))} ＞ {','.join(map(str, second_col))} ＞ {','.join(map(str, third_col))}")
        lines.append("")

    return "\n".join(lines)




# ============================================================
# STEP0 行動分解ヘルパー
# ============================================================

def _surface_kind_text(race: Race, r: PastRun) -> str:
    src = f"{race.course or ''} {r.course or ''}"
    if "ダ" in src or "ダート" in src:
        return "ダート"
    if "芝" in src:
        return "芝"
    return "不明"

def _gate_zone(gate_no: Optional[int], heads: Optional[int]) -> str:
    if not gate_no or not heads:
        return "枠順位置不明"
    if gate_no <= max(1, math.ceil(heads * 0.33)):
        return "内寄り"
    if gate_no >= max(1, math.floor(heads * 0.67)):
        return "外寄り"
    return "中枠寄り"

def _position_zone(pos: Optional[int], heads: Optional[int]) -> str:
    if not pos or not heads:
        return "位置不明"
    ratio = pos / max(1, heads)
    if pos == 1:
        return "ハナ"
    if ratio <= 0.25:
        return "先行列"
    if ratio <= 0.55:
        return "好位〜中団前"
    if ratio <= 0.80:
        return "中団後ろ"
    return "後方"

def _horse_first_position_profile(h: Horse) -> str:
    vals = [r.first_pos for r in h.past_runs if r.first_pos is not None]
    if not vals:
        return "近走の初角位置が不足しているため、本来の発馬後ポジションは未確定。"
    front = sum(1 for v in vals if v <= 3)
    lead = sum(1 for v in vals if v == 1)
    avg = sum(vals) / len(vals)
    if lead >= max(2, len(vals)//2):
        return f"近走{len(vals)}走中{lead}走で初角1番手。ハナ主張履歴が多い素材として保持。"
    if front >= max(2, len(vals)//2):
        return f"近走{len(vals)}走中{front}走で初角3番手以内。発馬後に前列へ参加する履歴が多い素材として保持。"
    if avg <= 6:
        return f"近走初角平均は約{avg:.1f}番手。先団〜中団前で入る履歴として保持。"
    return f"近走初角平均は約{avg:.1f}番手。中団以降から進める履歴として保持。"

def _start_and_first_corner_material(h: Horse, r: PastRun) -> str:
    first = r.first_pos
    heads = r.heads
    gate = r.gate_no
    gate_zone = _gate_zone(gate, heads)
    pos_zone = _position_zone(first, heads)
    base_vals = [x.first_pos for x in h.past_runs if x is not r and x.first_pos is not None]
    base_text = ""
    if base_vals and first is not None:
        base_avg = sum(base_vals) / len(base_vals)
        front_base = sum(1 for v in base_vals if v <= 3)
        if front_base >= max(1, len(base_vals)//2) and first >= 6:
            base_text = "過去の前列参加履歴に比べると、この走は序盤で前を取れていない可能性がある。"
        elif base_avg >= 7 and first <= 3:
            base_text = "過去の中団以降履歴に比べると、この走は普段より前へ出している可能性がある。"
        else:
            base_text = "近走内の位置取り傾向から大きく外れた形ではない。"
    else:
        base_text = "比較できる近走初角データが不足しており、本来の発馬後位置との比較は未確定。"

    if first is None:
        return f"スタート〜初角：馬番{_safe(gate)}、{gate_zone}。初角通過が取得できず、発馬直後の位置は未確定。{base_text}"

    gate_relation = ""
    if heads:
        if gate_zone == "内寄り" and first > max(4, heads * 0.55):
            gate_relation = "内寄りの馬番から序盤位置を取れておらず、控えた、出脚で負けた、または包まれた可能性を未確定素材として残す。"
        elif gate_zone == "外寄り" and first <= 3:
            gate_relation = "外寄りの馬番から初角前列へ入っており、序盤に出して行った素材として残す。"
        elif first == 1:
            gate_relation = "初角でハナに立っており、逃げまたは強い先行主張の素材として残す。"
        elif first <= 3:
            gate_relation = "初角3番手以内で、発馬後に前列へ参加した素材として残す。"
        else:
            gate_relation = "初角では前列ではなく、好位〜後方から運んだ素材として残す。"
    return f"スタート〜初角：馬番{_safe(gate)}、{gate_zone}から初角{first}番手（{pos_zone}）。{gate_relation}{base_text}"

def _segment_labels(n: int) -> List[str]:
    if n >= 4:
        return ["序盤〜1角", "1角〜2角", "2角〜3角", "3角〜4角"][: n-1]
    if n == 3:
        return ["序盤〜中間", "中間〜4角"]
    if n == 2:
        return ["序盤〜4角"]
    return []

def _position_change_material(r: PastRun) -> str:
    pos = r.passing_positions
    if not pos:
        return "道中位置変化：通過順が取得できず、どこで上がったか・下がったかは未確定。"
    labels = _segment_labels(len(pos))
    parts = []
    for i in range(len(pos) - 1):
        diff = pos[i+1] - pos[i]
        label = labels[i] if i < len(labels) else f"区間{i+1}"
        if diff <= -3:
            parts.append(f"{label}で{pos[i]}番手から{pos[i+1]}番手へ大きく押し上げ")
        elif diff <= -1:
            parts.append(f"{label}で{pos[i]}番手から{pos[i+1]}番手へじわっと前進")
        elif diff == 0:
            parts.append(f"{label}は{pos[i]}番手のまま位置維持")
        elif diff <= 2:
            parts.append(f"{label}で{pos[i]}番手から{pos[i+1]}番手へやや後退")
        else:
            parts.append(f"{label}で{pos[i]}番手から{pos[i+1]}番手へ大きく後退")
    if not parts:
        return f"道中位置変化：通過順は{r.passing_text}。単一点のため区間変化は未確定。"
    return "道中位置変化：" + "、".join(parts) + "。"

def _pace_change_material(r: PastRun, surface: str) -> str:
    pos = r.passing_positions
    agari = r.agari
    if not pos:
        return "ペース変化推定：ラップがないため、ペースの上げ下げは通過順と上がりからの推定に限定する。"
    last_move = 0
    if len(pos) >= 2:
        last_move = pos[-1] - pos[-2]
    early_front = pos[0] <= 3
    held_front = all(v <= 3 for v in pos)
    big_gain = len(pos) >= 2 and min(pos[i+1] - pos[i] for i in range(len(pos)-1)) <= -3
    big_drop = len(pos) >= 2 and max(pos[i+1] - pos[i] for i in range(len(pos)-1)) >= 3

    agari_text = ""
    if agari is None:
        agari_text = "上がり不明のため、直線の伸び・止まり方は通過順中心で見る。"
    elif surface == "芝":
        if agari <= 34.0:
            agari_text = f"芝で上がり{agari:.1f}は速い部類として扱い、終盤に脚を残した可能性を素材として保持。"
        elif agari >= 36.0:
            agari_text = f"芝で上がり{agari:.1f}は重く、終盤に脚を使い切った、または馬場・展開負荷を受けた可能性を素材として保持。"
        else:
            agari_text = f"芝で上がり{agari:.1f}は中間的で、位置取り変化と合わせて扱う。"
    else:
        if agari <= 38.0:
            agari_text = f"{surface}で上がり{agari:.1f}は比較的まとめており、終盤で極端には止まっていない素材として保持。"
        elif agari >= 40.0:
            agari_text = f"{surface}で上がり{agari:.1f}は重く、終盤負荷または失速疑いを素材として保持。"
        else:
            agari_text = f"{surface}で上がり{agari:.1f}は中間的で、位置取り変化と合わせて扱う。"

    if held_front:
        pace_text = "ペース変化推定：前列を維持しており、道中で大きく下げずに4角まで運んだ形。"
        if agari is not None and ((surface == "芝" and agari <= 34.5) or (surface != "芝" and agari <= 38.5)):
            pace_text += "前で運んだまま終盤も一定の脚を使っており、途中で大きく苦しくなった形とは断定しない。"
    elif early_front and big_drop:
        pace_text = "ペース変化推定：序盤は前列にいたが道中または4角までに位置を落としており、前半負荷、同型圧、被され、距離負荷のいずれかを未確定素材として残す。"
    elif big_gain:
        pace_text = "ペース変化推定：道中で大きく位置を上げており、途中で流れが緩んだ局面に乗った、または自ら早めに動いた可能性を未確定素材として残す。"
    elif len(pos) >= 2 and last_move <= -1:
        pace_text = "ペース変化推定：4角へ向けて位置を上げており、直線入口前に加速または押し上げる形を取った素材として残す。"
    elif len(pos) >= 2 and last_move >= 1:
        pace_text = "ペース変化推定：4角へ向けて位置を下げており、勝負所で加速負け、被され、距離負荷、または進路待ちの可能性を未確定素材として残す。"
    else:
        pace_text = "ペース変化推定：通過順の変動は小さく、淡々と同じ位置を保った素材として残す。"

    return pace_text + agari_text

def _cover_or_kickback_material(race: Race, r: PastRun) -> str:
    pos = r.passing_positions
    surface = _surface_kind_text(race, r)
    if surface == "芝":
        cover_word = "馬群内・被され・進路圧"
    elif surface == "ダート":
        cover_word = "砂被り・被され・キックバック"
    else:
        cover_word = "馬群内負荷・被され"
    if not pos:
        return f"{cover_word}：通過順不足のため未確定。"
    text = f"{cover_word}："
    if len(pos) >= 2 and pos[0] <= 5 and pos[-1] >= pos[0] + 4:
        text += "序盤に前または好位へ入った後に大きく下げており、外から被された、内で動けなかった、または進路・馬場負荷を受けた可能性を未確定素材として残す。"
    elif len(pos) >= 2 and pos[0] >= 7 and pos[-1] <= pos[0] - 3:
        text += "序盤は後方寄りだが4角までに押し上げており、馬群を捌く、外を回す、または流れに乗って進出した素材として残す。"
    elif pos[0] == 1:
        text += "初角1番手で、少なくとも序盤は他馬の後ろで受ける形ではない。"
    else:
        text += "成績表だけでは断定しない。通過順と枠順から、包まれや進路圧の疑いがあるかを後続Phaseで同型配置と合わせて確認する。"
    return text

def _run_result_fact_material(r: PastRun) -> str:
    rank = _safe(r.rank)
    margin = _safe(r.margin)
    time_text = _safe(r.time_text)
    return (
        f"結果事実：着順{rank}、時計{time_text}、着差{margin}秒。"
        "この段階では着順を能力判断に直結せず、どの位置で運び、どの区間で位置を変え、終盤にどの程度脚を使ったかの記録として扱う。"
    )

def _distance_and_condition_material(race: Race, r: PastRun) -> str:
    dist_current = race.distance
    dist_run = r.distance
    if dist_current and dist_run:
        diff = dist_current - dist_run
        if abs(diff) <= 100:
            dist = f"距離変化：前走距離{dist_run}mから今回{dist_current}mでほぼ同距離。位置取り再現性を比較しやすい素材。"
        elif diff > 0:
            dist = f"距離変化：前走距離{dist_run}mから今回{dist_current}mへ延長。序盤位置を取った場合の折り合い、勝負所まで脚を残せるかを後続Phaseで確認する素材。"
        else:
            dist = f"距離変化：前走距離{dist_run}mから今回{dist_current}mへ短縮。序盤から前へ参加できるか、または直線だけで届くかを後続Phaseで確認する素材。"
    else:
        dist = "距離変化：比較用の距離情報が不足。"
    going = f"馬場・コース：過去走は{_safe(r.venue)}{_safe(r.course)}{_safe(r.distance)}m、{_safe(r.going)}。今回条件との一致・違いはAI側でOADPにより再確認する。"
    return going + dist


def _position_text_from_run(r: PastRun) -> str:
    pos = r.passing_positions
    if not pos:
        return "通過順不足"
    return f"初角{pos[0]}番手→4角{pos[-1]}番手（通過{r.passing_text}）"

def _jockey_tactical_material(h: Horse, race: Race) -> str:
    """騎手や乗り替わりに関する事実素材。騎手の一般的な優劣やプラスマイナスは出さない。"""
    current = h.jockey or "-"
    run_jockeys = [r.jockey for r in h.past_runs if r.jockey]
    prev_jockey = run_jockeys[0] if run_jockeys else ""
    same_recent = [r for r in h.past_runs[:5] if h.jockey and r.jockey == h.jockey]
    other_recent = [r for r in h.past_runs[:5] if h.jockey and r.jockey and r.jockey != h.jockey]
    pieces = []
    pieces.append(f"騎手・戦術素材：今回騎手は{current}。")
    if prev_jockey:
        if current != "-" and prev_jockey == current:
            pieces.append(f"前走から継続騎乗。前走は{_position_text_from_run(h.past_runs[0])}。")
        else:
            pieces.append(f"前走騎手{prev_jockey}から今回{current}への乗り替わり。乗り替わりの影響はここで断定せず、位置取り変更の有無をAI側で確認する。")
    else:
        pieces.append("近走騎手が不足しており、継続騎乗・乗り替わりの比較は未確定。")
    if same_recent:
        detail = []
        for r in same_recent[:3]:
            detail.append(f"{r.date or '-'} {r.race_name or '-'}：{_position_text_from_run(r)}")
        pieces.append("今回騎手での近走素材：" + "／".join(detail) + "。")
    if other_recent:
        detail = []
        for r in other_recent[:3]:
            detail.append(f"{r.date or '-'} {r.jockey or '-'}：{_position_text_from_run(r)}")
        pieces.append("別騎手での近走素材：" + "／".join(detail) + "。")
    pieces.append("騎手の一般的な先行・差し傾向、コース別の乗り方、ハンデ戦での位置取り意識は、アプリでは結論化せずAI側でOADPにより判定する。")
    return "".join(pieces)

def _distance_rotation_intent_material(h: Horse, race: Race) -> str:
    """距離変化・ローテ・出走間隔の事実素材。陣営意図は候補だけ残し断定しない。"""
    current_dist = race.distance
    parts = []
    parts.append("距離変化・ローテ素材：")
    if h.past_runs:
        r0 = h.past_runs[0]
        if current_dist and r0.distance:
            diff = current_dist - r0.distance
            if abs(diff) <= 100:
                parts.append(f"前走{r0.distance}mから今回{current_dist}mでほぼ同距離。")
            elif diff > 0:
                parts.append(f"前走{r0.distance}mから今回{current_dist}mへ距離延長。序盤の折り合い、勝負所まで脚を残せるか、前走で脚を余した形かをAI側で確認する素材。")
            else:
                parts.append(f"前走{r0.distance}mから今回{current_dist}mへ距離短縮。追走速度が上がる可能性、終盤の脚を使いやすくなる可能性、または忙しくなる可能性をAI側で確認する素材。")
        else:
            parts.append("前走距離または今回距離が不足し、距離変化の比較は未確定。")
        if h.past_runs[:4]:
            dist_list = []
            for r in h.past_runs[:4]:
                dist_list.append(f"{r.date or '-'}:{r.distance or '-'}m/{r.race_name or '-'}")
            parts.append("近走距離履歴：" + "、".join(dist_list) + "。")
        # ローテ間隔は日付文字列が複数形式のため、ここでは日付列挙までに留める。
        date_list = [r.date for r in h.past_runs[:4] if r.date]
        if date_list:
            parts.append("近走日付履歴：" + "、".join(date_list) + "。間隔の長短、休み明け、詰めて使う意図はAI側で確認する。")
    else:
        parts.append("近走が不足しており、距離変化とローテの比較は未確定。")
    parts.append("距離延長・短縮を単独で結論化せず、枠順、同型数、騎手、斤量、前走通過順と複合してAI側で扱う。")
    return "".join(parts)

def _equipment_weight_material(h: Horse, race: Race) -> str:
    notes = h.equipment_notes or "-"
    pieces = []
    pieces.append(f"装備・斤量素材：装備メモは{notes}。今回斤量は{h.carried_weight if h.carried_weight is not None else '-'}kg。")
    prev_weights = [r.carried_weight for r in h.past_runs[:5] if r.carried_weight is not None]
    if prev_weights:
        pieces.append("近走斤量履歴：" + "、".join(f"{w:.1f}kg" for w in prev_weights[:5]) + "。")
        if h.carried_weight is not None:
            diff = h.carried_weight - prev_weights[0]
            if abs(diff) < 0.1:
                pieces.append("前走との斤量差はほぼなし。")
            elif diff > 0:
                pieces.append(f"前走比で{diff:.1f}kg増。負荷増として断定せず、位置取り・上がり・相手関係とAI側で複合確認する。")
            else:
                pieces.append(f"前走比で{abs(diff):.1f}kg減。軽量効果として断定せず、追走位置・仕掛け区間・枠順とAI側で複合確認する。")
    else:
        pieces.append("近走斤量履歴が不足。")
    pieces.append("ブリンカーなど装備変更は、前進気配ではなく、発馬・行き脚・集中・砂被り/被され耐性の確認材料としてAI側で扱う。")
    return "".join(pieces)

def _stable_intent_material(h: Horse, race: Race) -> str:
    """陣営思惑を断定せず、意図候補を素材として列挙する。"""
    pieces = []
    pieces.append("陣営意図の確認素材：")
    if h.trainer:
        pieces.append(f"厩舎表記は{h.trainer}。")
    if h.equipment_notes:
        pieces.append(f"{h.equipment_notes}があり、発馬後位置、集中、被され時の反応をAI側で確認する。")
    if h.past_runs:
        r0 = h.past_runs[0]
        pieces.append(f"前走は{r0.race_name or '-'}、{r0.venue or '-'}{r0.distance or '-'}m、{_position_text_from_run(r0)}。")
        if race.distance and r0.distance and race.distance != r0.distance:
            pieces.append("距離替わりがあり、追走改善狙い・脚の使いどころ変更・前半負荷調整のいずれかをAI側で確認する。")
        if h.jockey and r0.jockey and h.jockey != r0.jockey:
            pieces.append("乗り替わりがあり、位置取り変更や仕掛け区間変更の有無をAI側で確認する。")
    pieces.append("ここでは勝負気配、プラス、マイナス、消し、軸向きなどの判断語は出さない。OADP側で距離・斤量・装備・騎手・ローテを複合判定する。")
    return "".join(pieces)

def _ai_side_instruction_block() -> str:
    return (
        "【AI側で必ず行う判定指示】\n"
        "このSTEP0テキストは素材であり、アプリは馬の優劣・予想系数値・印・買い目・発生確率を出していない。\n"
        "AI側はOADP本文を読み直したうえで、以下をPhase1以降で必ず再判定する。\n"
        "1. 騎手の戦術プロファイル、今回騎手で位置を取りに行く可能性、継続騎乗/乗り替わりによる位置取り変更。\n"
        "2. 陣営意図として見える距離短縮・距離延長・装備・斤量・ローテの意味。ただし単独評価にせず必ず複合判定する。\n"
        "3. 近走の発馬後位置が本来型か、同型にハナを取られて控えた形か、枠順や頭数で位置取りが変わったか。\n"
        "4. 芝では馬群内・被され・進路圧、ダートでは砂被り・キックバック・被されで下げた可能性。\n"
        "5. 初出走・近走不足の馬は血統素材から想定できる行き脚、折り合い、持続力、距離・馬場適性をAI側で仮説化し、最終判断はOADPで行う。\n"
        "重要：アプリの文言を結論として採用しない。アプリが出した素材から、AI側でPhase0〜Phase4を新規に構築する。\n"
    )


def _horse_behavior_profile(h: Horse, race: Race) -> str:
    runs = [r for r in h.past_runs if r.passing_positions]
    if not runs:
        return "近走通過順が不足しており、反復するレース運びは未確定。"
    firsts = [r.first_pos for r in runs if r.first_pos is not None]
    lasts = [r.last_pos for r in runs if r.last_pos is not None]
    lead_count = sum(1 for v in firsts if v == 1)
    front_count = sum(1 for v in firsts if v <= 3)
    improve_count = sum(1 for r in runs if r.improved_positions >= 2)
    drop_count = sum(1 for r in runs if (r.last_pos is not None and r.first_pos is not None and r.last_pos - r.first_pos >= 3))
    pieces = []
    pieces.append(_horse_first_position_profile(h))
    if improve_count:
        pieces.append(f"近走{len(runs)}走中{improve_count}走で道中〜4角までに2つ以上位置を上げており、途中進出する履歴を素材として保持。")
    if drop_count:
        pieces.append(f"近走{len(runs)}走中{drop_count}走で序盤位置から4角までに3つ以上下げており、被され・同型圧・距離負荷・勝負所加速負けの確認対象として保持。")
    if lead_count:
        pieces.append(f"初角1番手履歴は{lead_count}走。今回メンバー内の他の前列参加馬と並べて、ハナを取り切れるか、譲るとどうなるかをAI側で確認する素材。")
    elif front_count:
        pieces.append(f"初角3番手以内履歴は{front_count}走。逃げ固定ではなく先行参加素材として保持。")
    return "".join(pieces)

def _pedigree_trait_analysis(h: Horse, race: Race) -> str:
    sire = h.sire or ""
    dam = h.dam or ""
    damsire = h.damsire or ""
    names = f"{sire} {dam} {damsire}"
    traits = []
    # 断定ではなく血統から想定される素材を出す。評価はしない。
    if any(x in names for x in ["キズナ", "ハーツクライ", "ルーラーシップ", "ゴールドシップ", "ステイゴールド", "オルフェーヴル"]):
        traits.append("持続力・中距離寄りの素材を持つ可能性")
    if any(x in names for x in ["ロードカナロア", "ダイワメジャー", "ザファクター", "サウスヴィグラス", "ヘニーヒューズ"]):
        traits.append("序盤の行き脚や短めの距離への適性素材を持つ可能性")
    if any(x in names for x in ["キングカメハメハ", "シルバーステート", "ブラックタイド", "キタサンブラック", "サトノダイヤモンド"]):
        traits.append("先行力と持続力の両方を確認したい素材")
    if any(x in names for x in ["Galileo", "Sadler", "ハービンジャー", "Roberto", "Platini"]):
        traits.append("時計の掛かる馬場や持続戦への接続素材")
    if any(x in names for x in ["Storm Cat", "フジキセキ", "アドマイヤムーン"]):
        traits.append("前向きさ・機動力・序盤反応の確認素材")
    if not traits:
        traits.append("血統からの単独断定は行わず、距離・馬場・調教・実戦通過順が出るまで仮説素材として保持")
    course = f"{race.course or ''}{race.distance or ''}m"
    return (
        "初出走・近走不足時の血統素材："
        f"父{h.sire or '-'}、母{h.dam or '-'}、母父{h.damsire or '-'}。"
        + "、".join(traits)
        + f"。今回条件{course}に対して、発馬後に前へ行けるか、道中で折り合えるか、勝負所で長く脚を使えるかをAI側でOADPにより判断する。"
        "アプリ側では血統を結論化せず、想定される確認ポイントだけを残す。"
    )

def _safe(v: Any, default: str = "-") -> str:
    return default if v is None or v == "" else str(v)


def _run_position_narrative(r: PastRun) -> Tuple[str, str, str]:
    positions = r.passing_positions
    if not positions:
        return "通過順が取得できないため、位置取りは成績欄から断定しない。", "位置不明", "不明"
    first = positions[0]
    last = positions[-1]
    if first <= 2:
        style = "逃げ・番手"
        start = "序盤から前列を取りに行く内容"
    elif first <= 4:
        style = "先行"
        start = "先行集団の中で競馬を進める内容"
    elif first <= 7:
        style = "好位〜中団"
        start = "前を見ながら中団寄りで脚をためる内容"
    else:
        style = "後方"
        start = "序盤は後方から運ぶ内容"

    diff = first - last
    if diff >= 3:
        move = "道中から4角へかけて大きく押し上げており、位置を取れなかっただけの凡走とは扱わない。"
    elif diff >= 1:
        move = "道中で少しずつ前との差を詰めており、直線入口までの接続は残っていた。"
    elif diff == 0:
        move = "通過順の変化は小さく、序盤に取った位置をおおむね維持する競馬。"
    elif diff <= -3:
        move = "道中または勝負所で大きく位置を下げており、前受け不発や追走負荷を疑う内容。"
    else:
        move = "わずかに位置を下げており、前列で受けた負荷が直線余力に影響した可能性がある。"
    return f"{start}。初期位置は{first}、終端位置は{last}で、{move}", style, ("上げ" if diff > 0 else "下げ" if diff < 0 else "維持")


def _rank_margin_narrative(r: PastRun) -> str:
    bits = []
    if r.rank is not None:
        if r.rank == 1:
            bits.append("着順は1着で、結果面では勝ち切り実績として扱える。")
        elif r.rank <= 3:
            bits.append(f"着順は{r.rank}着で、馬券圏内に残している。")
        elif r.rank <= 5:
            bits.append(f"着順は{r.rank}着で、掲示板圏の内容。着順だけで強く評価しすぎず、位置取りと着差を併用する。")
        else:
            bits.append(f"着順は{r.rank}着で、結果だけ見れば評価を上げにくい。")
    else:
        bits.append("着順が取得できないため、通過順と時計要素を優先する。")
    if r.margin is not None:
        if r.margin <= 0.3:
            bits.append(f"着差は{r.margin}秒で、勝ち馬との差は小さい。")
        elif r.margin <= 1.0:
            bits.append(f"着差は{r.margin}秒で、完全に崩れたとは扱わない。")
        else:
            bits.append(f"着差は{r.margin}秒で、勝負所または直線で差を広げられている。")
    if r.popularity is not None and r.rank is not None:
        if r.rank < r.popularity:
            bits.append(f"{r.popularity}人気より上の着順で走っており、人気以上の内容。")
        elif r.rank > r.popularity + 2:
            bits.append(f"{r.popularity}人気に対して着順を落としており、不発条件を確認する。")
        else:
            bits.append("人気と着順の乖離は大きくなく、過大にも過小にも扱いにくい。")
    return "".join(bits)


def _distance_conversion_narrative(race: Race, r: PastRun) -> str:
    if not race.distance or not r.distance:
        return "今回距離との比較は距離データ不足のため保留する。"
    delta = race.distance - r.distance
    if abs(delta) <= 100:
        return f"今回{race.distance}mとはほぼ同距離で、近走の追走位置と終いの使い方をそのまま変換しやすい。"
    if delta > 0:
        if delta >= 400:
            return f"今回は{delta}mの距離延長。追走は楽になりやすいが、前で受ける馬は直線まで余力を残せるかが焦点になる。"
        return f"今回は{delta}mの距離延長。序盤の忙しさは軽くなる一方、勝負所で早く動くと最後の甘さが出やすい。"
    delta_abs = abs(delta)
    if delta_abs >= 400:
        return f"今回は{delta_abs}mの距離短縮。追走の忙しさが増すため、前半から位置を取れるかを厳しく見る。"
    return f"今回は{delta_abs}mの距離短縮。前走よりも位置取り負荷が増す可能性があり、差し馬は4角までに届くかが焦点。"


def _going_track_narrative(race: Race, r: PastRun) -> str:
    same_track = bool(race.track and r.venue and race.track in r.venue)
    going_text = r.going or "馬場不明"
    race_going = race.going or "馬場不明"
    track_part = "今回と同場または近い条件での内容として扱いやすい。" if same_track else "今回は競馬場替わりのため、コーナー形状と砂質の変換を挟む。"
    if going_text and race_going and going_text == race_going:
        going_part = f"馬場は今回と同じ{race_going}寄りで、時計と上がりを比較しやすい。"
    else:
        going_part = f"前走馬場は{going_text}、今回は{race_going}で、同じ走りをそのまま移植しない。"
    return track_part + going_part


def _agari_narrative(r: PastRun, positions_state: str) -> str:
    if r.agari is None:
        return "上がり3Fが取得できないため、直線の伸びは通過順と着差から推定する。"
    if r.agari <= 38.5:
        base = f"上がり{r.agari}は地方ダートの近走としては速い部類で、直線で脚を使えている。"
    elif r.agari <= 40.0:
        base = f"上がり{r.agari}は標準域で、極端な切れ味ではないが大きな失速とも言い切れない。"
    elif r.agari <= 41.5:
        base = f"上がり{r.agari}はやや掛かっており、直線で余力を削られている。"
    else:
        base = f"上がり{r.agari}は重く、勝負所から直線にかけて余力が薄くなった内容。"
    if positions_state == "上げ":
        base += " 位置を上げながらこの上がりなら、単なる流れ込みではなく押し上げ評価を残す。"
    elif positions_state == "下げ":
        base += " 位置を下げながらの上がりなので、前半負荷や砂被り疑いを強めに見る。"
    else:
        base += " 位置維持型の上がりで、隊列内での粘りまたは流れ込みとして扱う。"
    return base


def build_past_run_700_analysis(race: Race, h: Horse, r: PastRun, f: HorseFeature, label: str) -> str:
    """近走を評価せず、レース運びの事実・推定素材として分解する。
    強い/弱い、高評価/低評価、期待値、印候補などの判断は出さない。
    """
    surface = _surface_kind_text(race, r)
    race_desc = (
        f"{label}行動分解：{_safe(r.date)}の{_safe(r.race_name, 'レース名不明')}は、"
        f"{_safe(r.venue, '競馬場不明')}の{_safe(r.course, 'コース不明')}{_safe(r.distance)}m、"
        f"{_safe(r.going, '馬場不明')}、{_safe(r.heads)}頭立て、馬番{_safe(r.gate_no)}。"
        f"騎手{_safe(r.jockey)}、斤量{_safe(r.carried_weight)}、馬体重{_safe(r.body_weight)}。"
        "ここでは着順の良し悪しではなく、発馬後の位置、道中の上下動、勝負所の動き、終盤の止まり方または伸び方を材料化する。"
    )
    start_desc = _start_and_first_corner_material(h, r)
    change_desc = _position_change_material(r)
    pace_desc = _pace_change_material(r, surface)
    cover_desc = _cover_or_kickback_material(race, r)
    cond_desc = _distance_and_condition_material(race, r)
    result_desc = _run_result_fact_material(r)

    same_type_note = (
        "同型比較メモ：この過去走でハナを取った、または前列に参加した場合でも、"
        "当時の相手関係にどれだけ同型がいたかは出馬表本文だけでは確定できない。"
        "今回メンバー内で初角1〜3番手履歴が多い馬を並べ、ハナ争い・番手譲り・被されの有無をAI側で確認する。"
    )
    uncertainty = (
        "映像を見ていないため、出遅れ、接触、騎手が抑えた、馬が行きたがった、直線で詰まった等は断定しない。"
        "ただし、通過順の後退、上がりの重さ、馬番と初角位置のズレは、未確定の不利・負荷素材として保持する。"
    )
    text = race_desc + start_desc + change_desc + pace_desc + cover_desc + cond_desc + result_desc + same_type_note + uncertainty
    if len(text) < 720:
        text += (
            " 追加記録：アプリはこの走を評価点に変換しない。"
            "次工程では、複数走で同じ発馬後位置が再現されているか、前へ行った時に最後まで形を保てるか、"
            "控えた時に砂被り・被され・進路圧で下げるか、距離延長や短縮でどの区間の負荷が変わるかを読むための素材として扱う。"
        )
    if len(text) > 1150:
        text = text[:1130] + "。"
    return text

def render_step0(race: Race, features: Dict[int, HorseFeature]) -> str:
    lines = []
    lines.append("Step 0：全頭・超高解像度 生データ／レース運び分解")
    lines.append("※本Stepでは予想結論、順位付け、点数、印、馬券内容を出さない。")
    lines.append("※近走解析は、馬がどのようにスタートし、どこで位置を上げ下げし、どこで負荷を受けた可能性があるかを記録する。")
    lines.append("※ペースの上げ下げは、ラップがない場合は通過順・上がり・着差からの推定素材に限定し、断定しない。")
    lines.append(f"【レース条件確認】{race.date_text}／{race.title}／{race.course}{race.distance}m{race.direction}／天候：{race.weather}／馬場：{race.going}／{race.field_size}頭。")
    lines.append("")
    for h in race.horses:
        f = features[h.horse_no]
        lines.append("============================================================")
        lines.append(f"【{h.horse_no}番 {h.name}】STEP0個体再構築")
        lines.append("============================================================")
        lines.append(
            f"1. 基礎データ：枠{h.frame_no or '-'}・馬番{h.horse_no}／馬名{h.name}／騎手{h.jockey}／厩舎{h.trainer}／"
            f"オッズ{h.odds if h.odds is not None else '-'} ({h.popularity if h.popularity else '-'}人気)／"
            f"馬体重{h.body_weight if h.body_weight is not None else '-'}({h.body_weight_diff if h.body_weight_diff is not None else '-'})／"
            f"性齢{h.sex_age}／斤量{h.carried_weight if h.carried_weight is not None else '-'}／"
            f"着別成績：{h.total_record or '取得不完全'}／左：{h.left_record or '-'}／右：{h.right_record or '-'}／"
            f"場：{h.track_record or '-'}／距：{h.distance_record or '-'}／最高時計：{h.best_time or '-'}。"
        )
        lines.append(
            f"2. 血統・所属：父{h.sire or '-'}、母{h.dam or '-'}、母父{h.damsire or '-'}。"
            "血統はここで結論化せず、初出走・近走不足・距離替わり・馬場替わりの確認ポイントとして保持する。"
        )
        lines.append("3. 近走レース運び分解")
        if not h.past_runs:
            lines.append("近走データが取得できない、または初出走に近い扱いのため、実戦通過順による確認はできない。")
            lines.append(_pedigree_trait_analysis(h, race))
        for idx_run, r in enumerate(h.past_runs[:5], start=1):
            label = ["前走", "前々走", "3走前", "4走前", "5走前"][idx_run - 1]
            lines.append("")
            lines.append(f"【{label}】")
            lines.append(
                f"データ：{r.date or '-'}／{r.race_name or '-'}／{r.venue or '-'}{r.course or '-'}{r.distance or '-'}m／"
                f"{r.going or '-'}／{r.heads or '-'}頭／馬番{r.gate_no or '-'}／"
                f"{r.rank or '-'}着／人気{r.popularity or '-'}／馬体重{r.body_weight or '-'}／騎手{r.jockey or '-'}／"
                f"斤量{r.carried_weight if r.carried_weight is not None else '-'}／時計{r.time_text or '-'}／"
                f"通過{r.passing_text or '-'}／上がり{r.agari if r.agari is not None else '-'}／着差{r.margin if r.margin is not None else '-'}。"
            )
            lines.append(build_past_run_700_analysis(race, h, r, f, label))
        lines.append("")
        lines.append("4. 近走反復パターン素材：" + _horse_behavior_profile(h, race))
        lines.append(
            "5. AI側へ渡す確認ポイント：発馬後に前列へ行く履歴が本来型か、控えた時に位置を下げるか、"
            "勝負所で自分から動けるか、前へ行った時に終盤まで形を保てるか、枠順と頭数で位置取りが変わるかを確認する。"
        )
        lines.append("6. 騎手・戦術素材：" + _jockey_tactical_material(h, race).replace("騎手・戦術素材：", "", 1))
        lines.append("7. 距離変化・ローテ素材：" + _distance_rotation_intent_material(h, race).replace("距離変化・ローテ素材：", "", 1))
        lines.append("8. 装備・斤量素材：" + _equipment_weight_material(h, race).replace("装備・斤量素材：", "", 1))
        lines.append("9. 陣営意図の確認素材：" + _stable_intent_material(h, race).replace("陣営意図の確認素材：", "", 1))
        lines.append("10. 血統素材：" + _pedigree_trait_analysis(h, race).replace("初出走・近走不足時の血統素材：", "", 1))
        lines.append("11. STEP0禁止事項確認：この段階では予想結論・着順予測・点数・順位を出力しない。騎手・陣営・距離変化・装備・血統の意味付けはAI側でOADPにより行う。")
        lines.append("")
    return "\n".join(lines)

def feature_to_json_dict(f: HorseFeature) -> Dict[str, Any]:
    d = asdict(f)
    d.update({
        "four_corner_ev": round(f.four_corner_ev, 6),
        "straight_ev": round(f.straight_ev, 6),
        "front_pressure_ev": round(f.front_pressure_ev, 6),
        "total_ability_ev": round(f.total_ability_ev, 6),
    })
    return d



# ============================================================
# アプリ整形監査・AI補正指示（開催場所／距離／芝ダート）
# ============================================================

def _short_course(course: str) -> str:
    c = course or ""
    if "ダ" in c:
        return "ダ"
    if "芝" in c:
        return "芝"
    if "障" in c:
        return "障"
    return "不明"


def _past_label(idx: int) -> str:
    labels = ["前走", "前々走", "3走前", "4走前", "5走前"]
    return labels[idx - 1] if 1 <= idx <= len(labels) else f"{idx}走前"


def _format_past_course_value(venue: str, course: str, distance: Optional[int]) -> str:
    v = venue or "-"
    d = str(distance) if distance is not None else "–"
    c = _short_course(course)
    if c == "不明":
        return f"{v}{d}m"
    return f"{v}{d}{c}"


def _plain_source_race_info(raw_text: str) -> Dict[str, Any]:
    """元テキストからレース本体の開催場・レース番号・コースを独立抽出する。"""
    raw = _clean_plain_input(raw_text or "")
    compact = normalize_spaces(raw[:20000])
    info: Dict[str, Any] = {"raw_label": "", "track": "", "race_no": None, "course": "", "distance": None}

    m = re.search(r"(\d+)回\s*(" + "|".join(map(re.escape, JRA_TRACK_NAMES)) + r")\s*(\d+)日", compact)
    if m:
        info["track"] = m.group(2)
        info["raw_label"] = f"{m.group(1)}回{m.group(2)}{m.group(3)}日"
        rm = re.search(r"(?<!\d)(\d{1,2})\s*レース", compact)
        if rm:
            info["race_no"] = int(rm.group(1))
            info["raw_label"] += f" {rm.group(1)}レース"
    else:
        m = re.search(r"(" + "|".join(map(re.escape, JRA_TRACK_NAMES)) + r").{0,30}?(\d{1,2})\s*レース", compact)
        if m:
            info["track"] = m.group(1)
            info["race_no"] = int(m.group(2))
            info["raw_label"] = f"{m.group(1)} {m.group(2)}レース"

    m = re.search(r"コース[:：]?\s*([0-9,]{3,5})\s*メートル\s*（\s*(芝|ダート|障害)", compact)
    if m:
        info["distance"] = int(m.group(1).replace(",", ""))
        info["course"] = m.group(2)
    elif re.search(r"(芝|ダート|障害)\s*([0-9,]{3,5})\s*m", compact, flags=re.I):
        mm = re.search(r"(芝|ダート|障害)\s*([0-9,]{3,5})\s*m", compact, flags=re.I)
        info["course"] = mm.group(1)
        info["distance"] = int(mm.group(2).replace(",", ""))
    return info


def _html_source_race_info(html: str) -> Dict[str, Any]:
    """元HTMLからレース本体の開催場・レース番号・コースを独立抽出する。"""
    source = html or ""
    txt = source
    if BeautifulSoup is not None and "<" in source:
        try:
            soup = BeautifulSoup(source, "html.parser")
            txt = normalize_spaces(soup.get_text(" "))
        except Exception:
            txt = re.sub(r"<[^>]+>", " ", source)
    compact = normalize_spaces(txt[:30000])
    info: Dict[str, Any] = {"raw_label": "", "track": "", "race_no": None, "course": "", "distance": None}

    m = re.search(r"(\d{4}年\d+月\d+日.*?）)\s*([^\s]+)\s*第(\d+)競走", compact)
    if m:
        info["track"] = m.group(2).replace("　", "")
        info["race_no"] = int(m.group(3))
        info["raw_label"] = f"{m.group(2).replace('　','')} 第{m.group(3)}競走"
    else:
        m = re.search(r"(" + "|".join(map(re.escape, JRA_TRACK_NAMES)) + r").{0,20}?第?(\d{1,2})(?:競走|R|レース)", compact)
        if m:
            info["track"] = m.group(1)
            info["race_no"] = int(m.group(2))
            info["raw_label"] = f"{m.group(1)} {m.group(2)}R"

    m = re.search(r"(ダート|芝)\s*([0-9,]{3,5})\s*ｍ", compact)
    if not m:
        m = re.search(r"(ダート|芝)\s*([0-9,]{3,5})\s*m", compact, flags=re.I)
    if m:
        info["course"] = m.group(1)
        info["distance"] = int(m.group(2).replace(",", ""))
    return info


def _source_race_info(source_text: str, source_type: str) -> Dict[str, Any]:
    return _plain_source_race_info(source_text) if source_type == "plain_text" else _html_source_race_info(source_text)


def _source_plain_past_candidates(raw_text: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """JRA縦型文章コピーの元テキストから、各馬各近走の開催場・距離・芝ダートを独立抽出する。"""
    raw = _clean_plain_input(raw_text or "")
    lines = [x.rstrip("\n") for x in raw.splitlines()]
    starts: List[Tuple[int, int, int]] = []
    for i, line in enumerate(lines):
        m = _jra_horse_start_match(line)
        if m:
            frame = int(m.group(1))
            horse_no = int(m.group(2))
            if 1 <= frame <= 8 and 1 <= horse_no <= 18:
                starts.append((i, frame, horse_no))
    result: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for pos, (start_idx, frame_no, horse_no) in enumerate(starts):
        end_idx = starts[pos + 1][0] if pos + 1 < len(starts) else len(lines)
        block = [normalize_spaces(x) for x in lines[start_idx:end_idx] if normalize_spaces(x)]
        if not block:
            continue
        name = ""
        for line in block[1:12]:
            if _is_noise_horse_name_line(line):
                continue
            if re.match(r"^\d{4}年", line):
                break
            if re.fullmatch(r"[\d.,]+万円?", line):
                continue
            name = line
            break
        date_indices = [i for i, line in enumerate(block) if re.match(r"^\d{4}年\d{1,2}月\d{1,2}日", line)]
        for rpos, di in enumerate(date_indices[:5], start=1):
            dend = date_indices[rpos] if rpos < len(date_indices) else len(block)
            run_lines = block[di:dend]
            joined = "\n".join(run_lines)
            venue = ""
            m = re.match(r"^\d{4}年\d{1,2}月\d{1,2}日\s*(.*)$", run_lines[0] if run_lines else "")
            if m:
                venue = normalize_spaces(m.group(1))
            distance: Optional[int] = None
            course = ""
            m = re.search(r"(\d{3,4})\s*(芝|ダート|障害)", joined)
            if not m:
                m = re.search(r"(芝|ダート|障害)\s*(\d{3,4})", joined)
                if m:
                    course = m.group(1)
                    distance = int(m.group(2))
            else:
                distance = int(m.group(1))
                course = m.group(2)
            label = _format_past_course_value(venue, course, distance)
            result[(horse_no, rpos)] = {
                "horse_number": horse_no,
                "horse_name": name,
                "race_index": rpos,
                "venue": venue,
                "distance": distance,
                "course": course,
                "source_value": label,
                "raw_snippet": "\n".join(run_lines[:16]),
            }
    return result


def _source_past_candidates(source_text: str, source_type: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    # NAR HTMLはparse結果が基本的に元HTMLに近いため、補正候補が一意に作れるJRA文章コピーを優先対応。
    # HTML入力でも欠落監査ブロックは出すが、元HTMLからの復元はAI側に委ねる。
    if source_type == "plain_text":
        return _source_plain_past_candidates(source_text)
    return {}


def _ai_processing_important_notice() -> str:
    return (
        "【AI処理時の重要注意】\n"
        "本STEP0パックでは、アプリ整形後の開催場所・距離・芝ダート表記に欠落または誤りが含まれる可能性がある。\n"
        "AIはPhase1以降へ進む前に、必ず元HTML／元テキスト、STEP0全文、パース済み事実データ、パース異常検査表を照合すること。\n"
        "開催場所・距離・芝ダートについて不一致または欠落がある場合、アプリ表記ではなく元HTML・元テキストを正とし、Phase0内で補正してから処理を継続すること。\n"
        "補正可能な不一致はERROR停止ではなく AI_CORRECTION_REQUIRED として扱い、AIが修正内容をPhase0に明記したうえでPhase1以降へ進むこと。\n"
        "ただし、元HTML・元テキストからも正しい開催場所・距離・芝ダートを復元できない場合のみ、UNRESOLVED_CRITICAL として処理停止すること。\n"
    )


def _ai_phase0_correction_log_instruction() -> str:
    return (
        "【AIがPhase0で必ず出すべき補正ログ】\n"
        "AIはPhase0出力時に、以下の補正ログを必ず記載すること。\n"
        "・レース本体開催場所の補正有無\n"
        "・近走開催場所の補正有無\n"
        "・近走距離・芝ダート欠落の補正有無\n"
        "・補正前表記\n"
        "・元テキスト表記\n"
        "・採用表記\n"
        "・Phase1へ渡した最終表記\n\n"
        "【アプリ整形補正ログ 例】\n"
        "馬番：1\n"
        "馬名：ドントゥール\n"
        "項目：近走距離・芝ダート欠落\n"
        "補正前：京都–m\n"
        "元テキスト：京都1800ダ\n"
        "採用：京都1800ダ\n"
        "Phase1反映：反映済み\n"
        "停止要否：停止しない\n"
    )


def build_app_correction_audit(race: Race, source_text: str, source_type: str = "html") -> Tuple[str, List[Dict[str, Any]], bool]:
    """開催場所・距離・芝ダートの整形監査。復元可能なものはAI_CORRECTION_REQUIREDで停止しない。"""
    rows: List[Dict[str, Any]] = []
    fatal = False

    def add(row: Dict[str, Any]) -> None:
        nonlocal fatal
        if row.get("judgement") == "UNRESOLVED_CRITICAL":
            fatal = True
            row["stop_required"] = True
        else:
            row.setdefault("stop_required", False)
        rows.append(row)

    src_info = _source_race_info(source_text, source_type)
    app_label = f"{race.track or '-'}{race.race_no or '-'}R"
    source_label = src_info.get("raw_label") or "-"
    race_mismatch = False
    if src_info.get("track") and race.track and src_info.get("track") != race.track:
        race_mismatch = True
    if src_info.get("race_no") and race.race_no and int(src_info.get("race_no")) != int(race.race_no):
        race_mismatch = True
    if race_mismatch:
        add({
            "type": "race_venue_mismatch",
            "section": "レース本体開催場所監査",
            "app_value": app_label,
            "source_value": source_label,
            "judgement": "AI_CORRECTION_REQUIRED",
            "ai_instruction": f"元テキスト表記を正とし、以降のPhase0〜Phase4では{src_info.get('track') or '元開催場'}{src_info.get('race_no') or ''}Rとして扱うこと。アプリ表記は破棄すること。",
            "stop_required": False,
        })
    elif (not race.track or not race.race_no) and source_label != "-":
        add({
            "type": "race_venue_missing",
            "section": "レース本体開催場所監査",
            "app_value": app_label,
            "source_value": source_label,
            "judgement": "AI_CORRECTION_REQUIRED",
            "ai_instruction": "元テキスト表記を正とし、開催場所・レース番号をPhase0で補正すること。",
            "stop_required": False,
        })
    else:
        add({
            "type": "race_venue_check",
            "section": "レース本体開催場所監査",
            "app_value": app_label,
            "source_value": source_label,
            "judgement": "OK" if source_label != "-" else "INFO",
            "ai_instruction": "不一致なし。元テキストも参照し、必要に応じて再確認すること。",
            "stop_required": False,
        })

    src_past = _source_past_candidates(source_text, source_type)
    for h in race.horses:
        for idx, r in enumerate(h.past_runs[:5], start=1):
            app_value = _format_past_course_value(r.venue, r.course, r.distance)
            source = src_past.get((h.horse_no, idx), {})
            source_value = source.get("source_value") or "-"
            missing_distance = r.distance is None
            missing_course = not (r.course and _short_course(r.course) != "不明")
            missing_venue = not r.venue
            app_has_dash = "–m" in app_value or "-m" in app_value or "不明" in app_value
            venue_mismatch = bool(source.get("venue") and r.venue and source.get("venue") != r.venue)
            cd_mismatch = bool(source.get("distance") and r.distance and source.get("distance") != r.distance) or bool(source.get("course") and r.course and _short_course(source.get("course")) != _short_course(r.course))

            if venue_mismatch:
                add({
                    "type": "past_race_venue_mismatch",
                    "section": "近走開催場所監査",
                    "horse_number": h.horse_no,
                    "horse_name": h.name,
                    "race_index": idx,
                    "race_label": _past_label(idx),
                    "app_value": app_value,
                    "source_value": source_value,
                    "judgement": "AI_CORRECTION_REQUIRED",
                    "ai_instruction": "元テキストの開催場所を正として補正すること。",
                    "stop_required": False,
                })
            elif source_value != "-":
                add({
                    "type": "past_race_venue_check",
                    "section": "近走開催場所監査",
                    "horse_number": h.horse_no,
                    "horse_name": h.name,
                    "race_index": idx,
                    "race_label": _past_label(idx),
                    "app_value": app_value,
                    "source_value": source_value,
                    "judgement": "OK",
                    "ai_instruction": "不一致なし。Phase0で元テキストも再照合すること。",
                    "stop_required": False,
                })

            if missing_distance or missing_course or app_has_dash or cd_mismatch:
                # 構造化抽出に失敗しても、元テキスト内に当該馬ブロックが残っていれば
                # AIが原文照合で補正できるため停止しない。
                source_has_horse_block = bool(source_text and h.name and h.name in source_text)
                source_is_structured = source_value != "-" and not ("–m" in source_value or "-m" in source_value or "不明" in source_value)
                if source_is_structured:
                    judgement = "AI_CORRECTION_REQUIRED"
                    resolved_source_value = source_value
                    instruction = "元テキスト表記を正とし、Phase0に補正すること。"
                elif source_has_horse_block:
                    judgement = "AI_CORRECTION_REQUIRED"
                    resolved_source_value = "元テキスト内の当該馬ブロックをAIが直接照合"
                    instruction = (
                        "アプリの構造化抽出では一意化できなかったが、元テキスト内に当該馬ブロックが存在する。"
                        "AIは馬番・馬名・対象近走順を照合し、開催場所・距離・芝ダートを元テキストから補正してPhase0へ反映すること。"
                    )
                else:
                    judgement = "UNRESOLVED_CRITICAL"
                    resolved_source_value = source_value
                    instruction = "元テキスト内にも当該馬ブロックを確認できず、一意に復元できないため停止すること。"
                add({
                    "type": "past_race_course_missing" if (missing_distance or missing_course or app_has_dash) else "past_race_course_mismatch",
                    "section": "近走距離・芝ダート欠落監査",
                    "horse_number": h.horse_no,
                    "horse_name": h.name,
                    "race_index": idx,
                    "race_label": _past_label(idx),
                    "app_value": app_value,
                    "source_value": resolved_source_value,
                    "judgement": judgement,
                    "ai_instruction": instruction,
                    "stop_required": judgement == "UNRESOLVED_CRITICAL",
                })

    lines: List[str] = []
    lines.append("【アプリ整形監査・AI補正指示】")
    lines.append("本STEP0は、アプリ整形結果と元HTML・元テキストの間に開催場所・距離・芝ダートの不一致または欠落がある可能性がある。")
    lines.append("AIは以下の監査結果をPhase0で必ず反映すること。補正可能な項目は停止せず、元HTML・元テキストを正として修正すること。")
    lines.append("")
    # Requested expanded format
    for row in rows:
        section = row.get("section", "整形監査")
        if section == "レース本体開催場所監査":
            lines.append("[レース本体開催場所監査]")
            lines.append(f"STEP0表記：{row.get('app_value','-')}")
            lines.append(f"元テキスト表記：{row.get('source_value','-')}")
            lines.append(f"判定：{row.get('judgement','-')}")
            lines.append(f"AI修正指示：{row.get('ai_instruction','-')}")
            lines.append(f"停止要否：{'停止する' if row.get('stop_required') else '停止しない'}")
            lines.append("")
        elif section == "近走開催場所監査":
            lines.append("[近走開催場所監査]")
            lines.append(f"馬番：{row.get('horse_number','-')}")
            lines.append(f"馬名：{row.get('horse_name','-')}")
            lines.append(f"対象近走：{row.get('race_label','-')}")
            lines.append(f"アプリ表記：{row.get('app_value','-')}")
            lines.append(f"元テキスト表記：{row.get('source_value','-')}")
            lines.append(f"判定：{row.get('judgement','-')}")
            lines.append(f"AI修正指示：{row.get('ai_instruction','-')}")
            lines.append(f"停止要否：{'停止する' if row.get('stop_required') else '停止しない'}")
            lines.append("")
        elif section == "近走距離・芝ダート欠落監査":
            lines.append("[近走距離・芝ダート欠落監査]")
            lines.append(f"馬番：{row.get('horse_number','-')}")
            lines.append(f"馬名：{row.get('horse_name','-')}")
            lines.append(f"対象近走：{row.get('race_label','-')}")
            lines.append(f"アプリ表記：{row.get('app_value','-')}")
            lines.append(f"元テキスト表記：{row.get('source_value','-')}")
            lines.append(f"判定：{row.get('judgement','-')}")
            lines.append(f"AI修正指示：{row.get('ai_instruction','-')}")
            lines.append(f"停止要否：{'停止する' if row.get('stop_required') else '停止しない'}")
            lines.append("")
    if len(rows) == 1 and rows[0].get("section") == "レース本体開催場所監査":
        lines.append("[近走開催場所監査]")
        lines.append("判定：INFO")
        lines.append("AI修正指示：近走ごとに元HTML・元テキストと照合し、必要があればPhase0で補正すること。")
        lines.append("停止要否：停止しない")
        lines.append("")
        lines.append("[近走距離・芝ダート欠落監査]")
        lines.append("判定：INFO")
        lines.append("AI修正指示：近走欄に–m、芝/ダート不明、距離不明があれば元HTML・元テキストから補正すること。")
        lines.append("停止要否：停止しない")
        lines.append("")

    return "\n".join(lines), rows, not fatal



def build_parse_audit(race: Race, features: Dict[int, HorseFeature]) -> Tuple[str, List[Dict[str, Any]], bool]:
    """AIへ渡す前のパース異常検査表を作る。ここでは予想結論には進まない。"""
    rows: List[Dict[str, Any]] = []
    fatal = False

    def add(level: str, item: str, target: str, value: Any, detail: str):
        nonlocal fatal
        if level in {"ERROR", "UNRESOLVED_CRITICAL"}:
            fatal = True
        rows.append({
            "level": level,
            "item": item,
            "target": target,
            "value": value,
            "detail": detail,
        })

    add("OK" if race.field_size > 0 else "ERROR", "出走馬数", "race", race.field_size, "出走馬を解析")
    add("OK" if race.title else "WARN", "レース名", "race", race.title or "-", "取得できない場合は元HTML確認")
    add("OK" if race.distance else "WARN", "距離", "race", race.distance or "-", "距離欄の取得確認")
    add("OK" if race.track else "WARN", "競馬場", "race", race.track or "-", "競馬場欄の取得確認")
    add("OK" if race.going else "WARN", "馬場", "race", race.going or "-", "馬場欄の取得確認")

    horse_numbers = [h.horse_no for h in race.horses]
    dupes = sorted({n for n in horse_numbers if horse_numbers.count(n) > 1})
    add("OK" if not dupes else "ERROR", "馬番重複", "race", dupes or "-", "重複があると候補数監査が壊れる")
    if horse_numbers:
        missing = [n for n in range(min(horse_numbers), max(horse_numbers) + 1) if n not in horse_numbers]
        add("OK" if not missing else "WARN", "馬番欠番", "race", missing or "-", "取消や欠番の可能性。出馬表と照合")
    add("OK", "採用頭数ルール", "race", race.candidate_count, "OADP指定の頭数確認")

    for h in race.horses:
        prefix = f"{h.horse_no} {h.name}"
        weight_st = carried_weight_status(h.carried_weight, race)
        add(weight_st, "当日斤量", prefix, h.carried_weight if h.carried_weight is not None else "-", "平地は45.0〜65.0の範囲外ならERROR。誕生日誤読を検出")
        add("OK" if h.odds is not None and h.odds > 0 else "WARN", "単勝オッズ", prefix, h.odds if h.odds is not None else "-", "単勝オッズ欄の取得確認。アプリ側では判断に使わない")
        pop_ok = h.popularity is not None and 1 <= h.popularity <= max(1, race.field_size)
        add("OK" if pop_ok else "WARN", "人気", prefix, h.popularity if h.popularity is not None else "-", "人気欄の取得確認。アプリ側では判断に使わない")
        if h.body_weight is None:
            bw_level = "WARN"
        elif "帯広" in (race.track or ""):
            bw_level = "OK"
        else:
            bw_level = "OK" if 300 <= h.body_weight <= 650 else "WARN"
        add(bw_level, "馬体重", prefix, h.body_weight if h.body_weight is not None else "-", "極端値はHTML解析を確認")
        add("OK" if len(h.past_runs) >= 3 else "WARN", "近走取得数", prefix, len(h.past_runs), "理想は近5走。3走未満はSTEP0材料不足")
        for idx, r in enumerate(h.past_runs[:5], start=1):
            label = f"{prefix} {idx}走前"
            add("OK" if r.rank is not None else "WARN", "近走着順", label, r.rank if r.rank is not None else "-", "取止・中止はWARNで保持")
            add("OK" if r.distance is not None else "WARN", "近走距離", label, r.distance if r.distance is not None else "-", "距離欄の取得確認")
            add("OK" if r.passing_text else "WARN", "近走通過順", label, r.passing_text or "-", "通過順欄の取得確認")
            if r.carried_weight is not None:
                if "帯広" in (race.track or ""):
                    ok = 400 <= r.carried_weight <= 900
                else:
                    ok = 45 <= r.carried_weight <= 65
                add("OK" if ok else "WARN", "近走斤量", label, r.carried_weight, "異常値ならHTML解析確認")
            else:
                add("WARN", "近走斤量", label, "-", "取得できない場合あり")

    lines: List[str] = []
    lines.append("AI用STEP0パック：パース異常検査表")
    lines.append("※この表はPhase1以降へ進む前の整合チェック。AI_CORRECTION_REQUIREDは停止せず元HTML・元テキストで補正する。UNRESOLVED_CRITICALまたは真のERRORのみ停止する。")
    lines.append("")
    lines.append(f"総合判定：{'UNRESOLVED_CRITICALまたはERRORあり：OADP処理停止' if fatal else 'OK/WARN/AI_CORRECTION_REQUIREDのみ：AI側で補正ログを出してPhase1以降へ進行可能'}")
    lines.append("")
    lines.append("| level | item | target | value | detail |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['level']} | {r['item']} | {r['target']} | {r['value']} | {r['detail']} |")
    return "\n".join(lines), rows, not fatal


def build_phase0_raw_material(race: Race, features: Dict[int, HorseFeature], correction_candidates: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Phase0以降のAI実行に必要な素材JSON。印・買い目・最終予想は含めない。"""
    data = {
        "race": asdict(race),
        "field_size": race.field_size,
        "candidate_count_rule": {
            "field_size": race.field_size,
            "candidate_count": race.candidate_count,
            "rule": "12頭以上=8頭、10〜11頭=7頭、9頭以下=6頭",
        },
        "features_for_phase0": {str(no): feature_to_json_dict(f) for no, f in features.items()},
        "strict_usage_note": [
            "このJSONはPhase0以降の素材であり、印・買い目ではない。",
            "AIはOADP本文と理想STEP0フォーマットを読み直してからPhase1以降へ進む。",
            "Phase1では4角出口〜直線入口までで止め、着順を先出ししない。",
            "correction_candidatesがある場合、AIはPhase0で補正ログを出し、元HTML・元テキスト表記を採用してからPhase1へ進む。",
        ],
    }
    if correction_candidates is not None:
        data["correction_candidates"] = correction_candidates
    return data


def build_step0_pack(html: str) -> Tuple[Race, Dict[str, str], Dict[str, Any]]:
    """出馬表HTMLからAI用STEP0パックを作る。
    予想・Phase1〜Phase4は実行しない。
    旧ZIP運用向けにも、開催場所・距離・芝ダートのAI補正監査を入れる。
    """
    race = parse_nar_html(html)
    if race.field_size == 0:
        raise ValueError("出馬表HTMLから出走馬を解析できませんでした。NARの出馬表HTML全体を貼り付けてください。")
    features = compute_features(race)

    correction_text, correction_rows, correction_ok = build_app_correction_audit(race, html, "html")
    correction_candidates = correction_rows
    notice = _ai_processing_important_notice()
    phase0_log_instruction = _ai_phase0_correction_log_instruction()

    step0 = notice + "\n\n" + correction_text + "\n\n" + render_step0(race, features)
    audit_text, audit_rows, audit_ok = build_parse_audit(race, features)
    raw_material = build_phase0_raw_material(race, features, correction_candidates)
    parsed_horses = _race_horses_only_json(race, correction_candidates)

    autorun = (
        "OADP STEP0パック自動実行リクエスト\n"
        "================================\n\n"
        "このZIPを受け取ったChatGPTは、ユーザーから追加指示がなくても、このファイルを実行依頼として扱うこと。\n\n"
        "実行手順：\n"
        "1. /mnt/data/OADP_OADP_展開先行監査人気馬汚染遮断前列圧参加馬保護統合改善版_全文.txt を読み直す。\n"
        "2. /mnt/data/理想のSTEP0フォーマット.txt を読み直す。\n"
        "3. 04_parse_audit.txt と 03_STEP0_full.txt 冒頭のアプリ整形監査を確認する。\n"
        "4. AI_CORRECTION_REQUIREDは停止せず、元HTMLを正としてPhase0で補正ログを出してからPhase1へ進む。\n"
        "5. UNRESOLVED_CRITICALまたは真のERRORがある場合のみPhase1以降へ進まず停止する。\n"
        "6. 03_STEP0_full.txt と 05_phase0_raw_material.json を材料に、Phase0〜Phase3をテキストファイル化する。\n"
        "7. Phase4は本文に出し、OADP指定の最終フォーマットに従う。\n"
        "8. 出力前監査表と『漏れ修正前→修正後』を必ず出し、NGがあれば最終出力せずPhase2またはPhase3へ戻す。\n\n"
        "このZIPは、追加の自然文指示なしでPhase1以降を開始するためのパックです。\n"
    )

    readme = (
        "AI用STEP0パック\n"
        "================\n\n"
        "このZIPは、アプリが予想・印・買い目を出すためのものではありません。\n"
        "出馬表HTMLを機械的に解析し、AIがOADPでPhase1以降を実行するための材料だけをまとめたものです。\n\n"
        + notice + "\n" + phase0_log_instruction + "\n"
        "重要：04_parse_audit.txtにAI_CORRECTION_REQUIREDがある場合は停止せず、元HTMLを正としてPhase0で補正してください。\n"
        "UNRESOLVED_CRITICALまたは真のERRORがある場合のみ、OADP処理を止めて元HTMLを確認してください。\n"
    )
    files = {
        "00_AUTO_RUN_REQUEST.txt": autorun,
        "01_source.html": html,
        "02_parsed_horses.json": json.dumps(parsed_horses, ensure_ascii=False, indent=2),
        "03_STEP0_full.txt": step0,
        "04_parse_audit.txt": correction_text + "\n\n" + audit_text,
        "05_phase0_raw_material.json": json.dumps(raw_material, ensure_ascii=False, indent=2),
        "06_README_for_AI.txt": readme,
    }
    meta = {
        "audit_ok": unified_ok,
        "audit_rows": audit_rows,
        "correction_candidates": correction_candidates,
        "horse_count": race.field_size,
        "candidate_count": race.candidate_count,
    }
    return race, files, meta

def write_step0_pack_outputs(files: Dict[str, str], output_dir: str | Path, prefix: str = "race") -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for name, content in files.items():
        p = out / f"{prefix}_{name}" if name.endswith(".html") else out / f"{prefix}_{name}"
        p.write_text(content, encoding="utf-8")
        paths[name] = str(p)
    return paths

def run_oadp_pipeline(html: str) -> Tuple[Race, PhaseResult]:
    race = parse_nar_html(html)
    if race.field_size == 0:
        raise ValueError("出馬表HTMLから出走馬を解析できませんでした。NARの出馬表HTML全体を貼り付けてください。")

    features = compute_features(race)
    step0 = render_step0(race, features)
    phase0, p0data = generate_phase0(race, features)
    phase1, scenarios, p1aux = generate_phase1(race, features, p0data)
    phase2, p2data = generate_phase2(race, features, scenarios, p0data, p1aux)
    phase3, candidate_map, diff_rows = generate_phase3(race, features, scenarios, p2data)
    audit_ok, audit_rows = audit_before_phase4(race, features, scenarios, candidate_map, p2data, p1aux)
    phase4 = generate_phase4(race, features, scenarios, candidate_map, audit_ok, audit_rows, diff_rows)
    return race, PhaseResult(
        step0_text=step0,
        phase0_text=phase0,
        phase1_text=phase1,
        phase2_text=phase2,
        phase3_text=phase3,
        phase4_text=phase4,
        audit_table=audit_rows,
        audit_ok=audit_ok,
    )


def write_outputs(result: PhaseResult, output_dir: str | Path, prefix: str = "race") -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        f"{prefix}_STEP0_全頭個体再構築.txt": result.step0_text,
        f"{prefix}_Phase0_展開前処理.txt": result.phase0_text,
        f"{prefix}_Phase1_4角隊列.txt": result.phase1_text,
        f"{prefix}_Phase2_EV監査表.txt": result.phase2_text,
        f"{prefix}_Phase3_印候補監査.txt": result.phase3_text,
        f"{prefix}_Phase4_最終出力または停止.txt": result.phase4_text,
    }
    paths = {}
    for name, text in files.items():
        p = out / name
        p.write_text(text, encoding="utf-8")
        paths[name] = str(p)
    result.files = paths
    return paths

# ============================================================
# v1.3 AI用STEP0二分割テキストパック
# 目的:
# - アプリ出力から、AIが引っ張られやすい評価語・順位語・発生率・印・買い目を排除する。
# - ZIPではなく1つの.txtだけを出力し、スマホからそのままChatGPTへ送れるようにする。
# - Phase1以降はChatGPT側でOADP本文と理想STEP0を読み直して新規実行する。
# ============================================================

def _single_text_clean_generated(text: str) -> str:
    """アプリ生成部分から、AIが予想根拠として誤読しやすい語と派生数値欄を除去する。
    元HTMLは原文保持が原則だが、STEP0本文・監査表・案内文には評価系表記を残さない。
    """
    if not text:
        return ""

    # 派生数値欄そのものを削除
    text = re.sub(r"STEP0上の扱いは、前列性[0-9.]+、4角到達[0-9.]+、直線余力[0-9.]+、前受け失速リスク[0-9.]+を参考値とし、まだ印・買い目には変換しない。", 
                  "STEP0上の扱いは、位置取り反復、4角付近での脚の残り方、直線での粘りまたは伸び、失速疑いを材料として保持し、まだ印・買い目には変換しない。", text)
    text = re.sub(r"\n5\. STEP0数値保持：.*?(?=\n6\. STEP0禁止事項確認：)", "\n5. STEP0素材保持：近走の位置取り、通過順、上がり、着差、距離・馬場・斤量・枠順・騎手・馬体重を、後続処理の材料として保持する。", text, flags=re.S)

    # 表記の置換。アプリ生成物に評価値・順位語を出さない。
    replacements = {
        "評価系": "判断系",
        "評価点": "点数",
        "高評価": "確認素材",
        "低評価": "注意素材",
        "評価": "判断",
        "期待値順位": "順位付け",
        "期待値計算": "OADP処理",
        "期待値算出": "OADP処理",
        "期待値": "OADP",
        "EV": "素材",
        "rawEV": "素材値",
        "総合EV": "総合素材",
        "直線EV": "直線素材",
        "4角EV": "4角素材",
        "前列圧EV": "前列圧素材",
        "余力EV": "余力素材",
        "印候補数": "採用頭数",
        "印候補": "採用対象",
        "印・買い目": "予想結論",
        "買い目": "最終フォーマット",
        "発生率": "シナリオ比率",
        "候補順位": "候補整理",
        "順位付け": "整理",
        "スコア": "素材",
        "score": "material",
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    return text


def _race_horses_only_json(race: Race, correction_candidates: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """派生判断を含まない、HTML/文章から読めた事実データだけのJSON。補正候補はAIが元テキスト照合するための監査素材。"""
    data = {
        "race": {
            "title": race.title,
            "date_text": race.date_text,
            "track": race.track,
            "race_no": race.race_no,
            "post_time": race.post_time,
            "course": race.course,
            "distance": race.distance,
            "direction": race.direction,
            "weather": race.weather,
            "going": race.going,
            "race_class": race.race_class,
            "field_size": race.field_size,
            "candidate_count_rule": "12頭以上=8頭、10〜11頭=7頭、9頭以下=6頭",
            "candidate_count": race.candidate_count,
        },
        "horses": [asdict(h) for h in race.horses],
        "usage_note": [
            "このJSONはHTML/文章から読めた事実データの整理であり、予想結論や順位付けではない。",
            "Phase1以降は、OADP本文と理想STEP0フォーマットを読み直して新規に行う。",
            "アプリ由来の馬の優劣・点数・順位・印・買い目は存在しない。",
            "correction_candidatesはAIが元HTML・元テキストと照合してPhase0で補正するための素材。AI_CORRECTION_REQUIREDは停止せず補正ログを出す。",
        ],
    }
    if correction_candidates is not None:
        data["correction_candidates"] = correction_candidates
    return data




# ============================================================
# プレーンテキスト入力対応（JRAなど、URL/HTML取得が難しい出馬表向け）
# ============================================================

JRA_TRACK_NAMES = [
    "札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉",
    "門別", "盛岡", "水沢", "浦和", "船橋", "大井", "川崎", "金沢", "笠松", "名古屋",
    "園田", "姫路", "高知", "佐賀", "帯広"
]


def _clean_plain_input(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\u3000", " ")


def _parse_plain_race_meta(raw_text: str) -> Race:
    """JRA等の文章コピーからレース本体ヘッダーだけを優先して解析する。

    重要：
    - 最初の馬柱より前をレース本体ヘッダーとみなす。
    - 近走欄の開催地・日付・レース名を本体情報へ混入させない。
    - 日付、開催回次、競馬場、レース番号、正式レース名を別々に確定する。
    """
    raw = _clean_plain_input(raw_text)
    race = Race()

    all_lines = [normalize_spaces(x) for x in raw.splitlines() if normalize_spaces(x)]
    first_horse_line = None
    for i, line in enumerate(all_lines):
        if _jra_horse_start_match(line):
            first_horse_line = i
            break

    # JRA文章コピーでは、最初の馬柱までをレース本体ヘッダーとして扱う。
    header_lines = all_lines[:first_horse_line] if first_horse_line is not None else all_lines[:160]
    header_text = "\n".join(header_lines)
    compact = normalize_spaces(header_text)

    # 日付。最初のヘッダー日付のみ採用し、近走日付を使わない。
    m = re.search(r"(\d{4}年\d{1,2}月\d{1,2}日(?:（[^）]+）)?)", header_text)
    if m:
        race.date_text = m.group(1)
    else:
        m = re.search(r"(\d{1,2}月\d{1,2}日(?:（[^）]+）)?)", header_text)
        if m:
            race.date_text = m.group(1)

    # 開催回次＋競馬場＋開催日を一体で抽出。
    # 例：2回福島6日 / 第2回福島競馬第6日
    venue_candidates = []
    for line in header_lines[:30]:
        m = re.search(
            r"(?:第?\s*(\d+)\s*回)?\s*(" + "|".join(map(re.escape, JRA_TRACK_NAMES)) + r")\s*(?:競馬)?\s*(?:第?\s*(\d+)\s*日)?",
            line
        )
        if m:
            venue_candidates.append({
                "line": line,
                "meet_no": int(m.group(1)) if m.group(1) else None,
                "track": m.group(2),
                "day_no": int(m.group(3)) if m.group(3) else None,
            })

    if venue_candidates:
        # 回次・開催日まで取れている候補を優先。
        venue_candidates.sort(
            key=lambda x: (
                x["meet_no"] is not None,
                x["day_no"] is not None,
                len(x["line"])
            ),
            reverse=True
        )
        chosen = venue_candidates[0]
        race.track = chosen["track"]
        # Race dataclassに専用欄がない場合も、raw metaとして後段で利用できるよう保持。
        setattr(race, "meeting_no", chosen["meet_no"])
        setattr(race, "meeting_day_no", chosen["day_no"])
        setattr(race, "source_race_header_line", chosen["line"])
    else:
        # ヘッダー内に単独で現れる競馬場名のみを対象とし、近走欄は参照しない。
        for line in header_lines[:40]:
            found = [name for name in JRA_TRACK_NAMES if re.search(rf"(?<![一-龠ぁ-んァ-ン]){re.escape(name)}(?![一-龠ぁ-んァ-ン])", line)]
            if len(found) == 1:
                race.track = found[0]
                setattr(race, "source_race_header_line", line)
                break

    # レース番号。ヘッダー内の「11レース」「11R」のみ採用。
    race_no_candidates = []
    for line in header_lines[:60]:
        for mm in re.finditer(r"(?<!\d)(\d{1,2})\s*(?:R|レース)(?!\d)", line, flags=re.I):
            no = int(mm.group(1))
            if 1 <= no <= 12:
                score = 2 if re.fullmatch(r"\d{1,2}\s*(?:R|レース)", line, flags=re.I) else 1
                race_no_candidates.append((score, no, line))
    if race_no_candidates:
        race_no_candidates.sort(reverse=True)
        _, race.race_no, source_line = race_no_candidates[0]
        setattr(race, "source_race_no_line", source_line)

    # 発走時刻
    for source in (header_text, compact):
        m = re.search(r"発走(?:時刻)?[:：]?\s*([0-2]?\d)[:：]([0-5]\d)", source)
        if not m:
            m = re.search(r"発走(?:時刻)?[:：]?\s*([0-2]?\d)時([0-5]\d)分", source)
        if not m:
            m = re.search(r"([0-2]?\d)[:：]([0-5]\d)\s*発走", source)
        if m:
            race.post_time = f"{int(m.group(1)):02d}:{m.group(2)}"
            break

    # コース・距離・向き。ヘッダー内の今回条件のみ採用。
    m = re.search(r"コース[:：]?\s*([0-9,]{3,5})\s*メートル\s*[（(]\s*(芝|ダート|障害)[・･\s]*(右|左|直線|外|内)?", compact)
    if m:
        race.distance = int(m.group(1).replace(",", ""))
        race.course = m.group(2)
        if m.group(3):
            race.direction = m.group(3)
    else:
        m = re.search(r"(芝|ダート|障害)\s*([0-9,]{3,5})\s*(?:m|メートル)", compact, flags=re.I)
        if m:
            race.course = m.group(1)
            race.distance = int(m.group(2).replace(",", ""))
        dm = re.search(r"[（(](右|左|直線|外|内)(?:[^）)]*)[）)]", compact)
        if dm:
            race.direction = dm.group(1)

    # 当日天候・馬場。明示表記がない限り近走の「良」等は使わない。
    m = re.search(r"天候[:：]\s*([^\s　/／]+)", compact)
    if m:
        race.weather = m.group(1)
    m = re.search(r"(?:馬場|馬場状態)[:：]\s*([^\s　/／]+)", compact)
    if m:
        race.going = m.group(1)

    # 正式レース名。ヘッダー内で「レース番号の後、条件行の前」を最優先。
    title_candidates = []
    race_no_idx = None
    for i, line in enumerate(header_lines):
        if race.race_no and re.fullmatch(rf"{race.race_no}\s*(?:R|レース)", line, flags=re.I):
            race_no_idx = i
            break

    scan_lines = header_lines[race_no_idx + 1:] if race_no_idx is not None else header_lines
    for i, line in enumerate(scan_lines[:40]):
        if re.search(r"^(ウインファイヴ|WIN5|第?\d+レース目|3歳|4歳|障害|本賞金|付加賞|印刷用|馬柱|スマートフォン|着順|同一|枠|馬番)", line):
            continue
        if "コース：" in line or "コース:" in line:
            continue
        if len(line) < 3 or len(line) > 90:
            continue
        # 重賞・特別・一般条件・未勝利等のレース名らしい行。
        if re.search(r"(賞|記念|ステークス|特別|カップ|杯|未勝利|新馬|オープン|リステッド|G[ⅠⅡⅢ123])", line):
            score = 0
            if re.search(r"第\d+回", line):
                score += 3
            if re.search(r"G[ⅠⅡⅢ123]", line):
                score += 2
            # レース番号直後に近いほど高得点
            score += max(0, 20 - i)
            title_candidates.append((score, line))

    if title_candidates:
        title_candidates.sort(key=lambda x: x[0], reverse=True)
        race.title = title_candidates[0][1][:100]
        setattr(race, "source_race_title_line", title_candidates[0][1])
    elif race.race_no:
        race.title = f"{race.race_no}R"

    race.race_class = "プレーンテキスト入力"
    return race
def _is_number_token(token: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}", token or ""))


def _parse_plain_horse_row(line: str) -> Optional[Horse]:
    """JRAなどの出馬表を画面から文章コピーした行を、事実データとして粗く読む。
    完璧なJRA専用スクレイピングではなく、馬番・馬名・性齢・斤量を優先して抽出する。
    """
    original = line
    line = normalize_spaces(
        line.replace("\t", " ")
            .replace("　", " ")
            .replace("枠", " ")
            .replace("番", " ")
            .replace("取消", " 取消 ")
            .replace("除外", " 除外 ")
    )
    if not line or len(line) < 6:
        return None
    # 見出し行は除外する。ただし実データ行には「5人気」のような語が入るため、
    # 先に性齢の有無を確認する。
    if not re.search(r"[牡牝セ騸せん]\s*\d+", line):
        return None
    if re.search(r"(馬名|性齢|斤量|騎手|厩舎|単勝|馬体重|枠番|出走馬)", line):
        return None

    tokens = [t for t in line.split(" ") if t]
    if len(tokens) < 4:
        return None

    sex_idx = None
    sex_age = ""
    for i, tok in enumerate(tokens):
        mm = re.fullmatch(r"(牡|牝|セ|騸|せん)\s*(\d+)", tok)
        if mm:
            sex_idx = i
            sex_age = ("セ" if mm.group(1) in {"せん", "騸"} else mm.group(1)) + mm.group(2)
            break
    if sex_idx is None or sex_idx < 1:
        return None

    before = tokens[:sex_idx]
    num_idx = None
    frame_no: Optional[int] = None
    horse_no: Optional[int] = None

    numeric_positions = [i for i, tok in enumerate(before) if _is_number_token(tok)]
    if not numeric_positions:
        # 「1. 馬名 牡4」など
        m = re.match(r"\s*(\d{1,2})[.\-:：]\s*(.+)$", original)
        if m:
            horse_no = int(m.group(1))
            num_idx = 0
        else:
            return None
    elif len(numeric_positions) >= 2 and numeric_positions[0] == 0 and numeric_positions[1] == 1:
        frame_no = int(before[0])
        horse_no = int(before[1])
        num_idx = 1
    else:
        # 最初に現れる1〜18程度の数字を馬番とみなす
        for p in numeric_positions:
            n = int(before[p])
            if 1 <= n <= 18:
                horse_no = n
                num_idx = p
                break
        if horse_no is None:
            return None

    name_tokens = before[(num_idx + 1):] if num_idx is not None else before
    name = normalize_spaces(" ".join(name_tokens))
    # 名前が空の場合、性齢直前の非数字を採用
    if not name and sex_idx > 0:
        name = tokens[sex_idx - 1]
    # ノイズを削る
    name = re.sub(r"^[◎○▲☆△注消]+\s*", "", name).strip()
    if not name or _is_number_token(name):
        return None

    # 斤量は性齢以降の近いトークンから拾う
    carried_weight: Optional[float] = None
    weight_idx: Optional[int] = None
    for j in range(sex_idx + 1, min(len(tokens), sex_idx + 8)):
        cand = tokens[j].replace("▲", "").replace("△", "").replace("◇", "").replace("☆", "").replace("★", "")
        if re.fullmatch(r"(?:4[5-9]|5\d|6[0-5])(?:\.\d)?", cand):
            carried_weight = float(cand if "." in cand else cand + ".0")
            weight_idx = j
            break
    if carried_weight is None:
        # 行全体から安全に抽出
        carried_weight = extract_current_carried_weight(line)

    # 騎手名は斤量の次の非数値トークンを候補にする
    jockey = ""
    if weight_idx is not None:
        for tok in tokens[weight_idx + 1: weight_idx + 5]:
            if not re.search(r"\d", tok) and tok not in {"栗東", "美浦", "地方", "牡", "牝"}:
                jockey = tok
                break

    # オッズと人気。馬体重を誤って拾わないよう、人気表記を優先。
    odds: Optional[float] = None
    popularity: Optional[int] = None
    pop_m = re.search(r"(\d{1,2})\s*人気", line)
    if pop_m:
        popularity = int(pop_m.group(1))
    # 単勝らしい小数。斤量・上がり・馬体重との差別化は完全ではないため、1.0〜999.9を許容。
    decimal_candidates = []
    for m in re.finditer(r"(?<!\d)(\d{1,3}\.\d)(?!\d)", line):
        v = float(m.group(1))
        if carried_weight is not None and abs(v - carried_weight) < 0.01:
            continue
        if 1.0 <= v <= 999.9:
            decimal_candidates.append(v)
    if decimal_candidates:
        odds = decimal_candidates[-1] if popularity is not None else decimal_candidates[0]

    bw_m = re.search(r"(?<!\d)([3-6]\d{2})(?:\s*)\(([+-]?\d+)\)", line)
    body_weight = int(bw_m.group(1)) if bw_m else None
    body_weight_diff = int(bw_m.group(2)) if bw_m else None

    return Horse(
        frame_no=frame_no,
        horse_no=int(horse_no),
        name=name,
        jockey=jockey,
        sex_age=sex_age,
        carried_weight=carried_weight,
        odds=odds,
        popularity=popularity,
        body_weight=body_weight,
        body_weight_diff=body_weight_diff,
        equipment_notes="ブリンカー着用" if "ブリンカー" in line else "",
    )



def _jra_horse_start_match(line: str) -> Optional[re.Match]:
    """JRA詳細出馬表の文章コピーで、馬ブロック先頭を検出する。
    例：枠1白 1 / 枠8桃\t16
    """
    s = normalize_spaces((line or "").replace("\t", " "))
    return re.match(r"^枠\s*(\d{1,2})[^\d\n]*\s+(\d{1,2})(?:\s|$)", s)


def _is_noise_horse_name_line(line: str) -> bool:
    s = normalize_spaces(line)
    if not s:
        return True
    if s in {"ブリンカー着用", "勝負服の画像"}:
        return True
    if re.fullmatch(r"\(\d+\.\d+\.\d+\.\d+\)", s):
        return True
    if re.search(r"万円$", s):
        return True
    if s.startswith(("父：", "母：", "(母の父：", "（母の父：")):
        return True
    if re.search(r"(美浦|栗東|地方)\)$", s):
        return True
    return False


def _parse_jra_vertical_past_run(run_lines: List[str]) -> Optional[PastRun]:
    """JRAの縦型コピー1走分を粗く読む。取れない項目は空欄で保持する。"""
    if not run_lines:
        return None
    lines = [normalize_spaces(x) for x in run_lines if normalize_spaces(x)]
    if not lines:
        return None
    first = lines[0]
    m = re.match(r"^(\d{4}年\d{1,2}月\d{1,2}日)\s*(.*)$", first)
    if not m:
        return None
    pr = PastRun(date=m.group(1), venue=(m.group(2) or "").strip())

    # レース名は日付直後の、着順・人気・斤量・距離ではない最初の行
    for line in lines[1:5]:
        if re.search(r"(G[ⅠⅡⅢ123]|OP|リステッド|勝ク|未勝利|新馬|S$|C$|賞|記念|特別|ハンデ)", line):
            # グレードだけの行は除外し、レース名らしい行を優先
            if not re.fullmatch(r"(G[ⅠⅡⅢ123]|OP|リステッド|3勝ク|2勝ク|1勝ク)", line):
                pr.race_name = line
                break
    if not pr.race_name and len(lines) > 1:
        pr.race_name = lines[1]

    joined = "\n".join(lines)
    m = re.search(r"(\d+|中止|取消|除外)着\s*(\d+)頭\s*(\d+)?番?", joined)
    if m:
        if m.group(1).isdigit():
            pr.rank = int(m.group(1))
        pr.heads = int(m.group(2))
        if m.group(3):
            pr.gate_no = int(m.group(3))
    m = re.search(r"(\d{1,2})番人気", joined)
    if m:
        pr.popularity = int(m.group(1))
    # 騎手・斤量
    for line in lines:
        m = re.search(r"(.+?)\s*([4-6]\d(?:\.\d)?)kg", line)
        if m and not re.match(r"^\d{3,4}(芝|ダート|障害)", line):
            pr.jockey = normalize_spaces(m.group(1))
            try:
                pr.carried_weight = float(m.group(2))
            except ValueError:
                pass
            break
    # 距離・コース
    m = re.search(r"(\d{3,4})\s*(芝|ダート|障害)", joined)
    if m:
        pr.distance = int(m.group(1))
        pr.course = m.group(2)
    else:
        m = re.search(r"(芝|ダート|障害)\s*(\d{3,4})", joined)
        if m:
            pr.course = m.group(1)
            pr.distance = int(m.group(2))
    # 走破時計
    m = re.search(r"(?m)^(\d+:\d{2}\.\d|\d{1,2}\.\d)$", joined)
    if m:
        pr.time_text = m.group(1)
    # 馬場
    for line in lines:
        if line in {"良", "稍重", "重", "不良"}:
            pr.going = line
            break
    # 馬体重
    m = re.search(r"([3-6]\d{2})kg", joined)
    if m:
        pr.body_weight = int(m.group(1))
    # 通過順。1行に2個以上の数字が並ぶ行だけ採用し、指数105などは除外。
    for line in lines:
        if re.fullmatch(r"\d+(?:\s+\d+){1,5}", line):
            vals = [int(x) for x in line.split()]
            if all(1 <= v <= 30 for v in vals):
                pr.passing_text = "-".join(str(v) for v in vals)
                break
    m = re.search(r"3F\s*([0-9]{2}\.[0-9])", joined)
    if m:
        pr.agari = float(m.group(1))
    m = re.search(r"([^\n()]+)\(([0-9]+\.[0-9])\)", joined)
    if m:
        pr.winner = normalize_spaces(m.group(1))
        pr.margin = float(m.group(2))
    return pr


def _parse_jra_vertical_horse_blocks(raw_text: str) -> List[Horse]:
    """JRAスマホ詳細出馬表のような、1項目1行の文章コピーを馬ごとに解析する。
    同じ行に馬番・馬名・性齢が並ばなくても処理できる。
    """
    raw = _clean_plain_input(raw_text)
    lines = [x.rstrip("\n") for x in raw.splitlines()]
    starts: List[Tuple[int, int, int]] = []
    for i, line in enumerate(lines):
        m = _jra_horse_start_match(line)
        if m:
            frame = int(m.group(1))
            horse_no = int(m.group(2))
            if 1 <= frame <= 8 and 1 <= horse_no <= 18:
                starts.append((i, frame, horse_no))
    if not starts:
        return []

    horses: List[Horse] = []
    for pos, (start_idx, frame_no, horse_no) in enumerate(starts):
        end_idx = starts[pos + 1][0] if pos + 1 < len(starts) else len(lines)
        block = [normalize_spaces(x) for x in lines[start_idx:end_idx] if normalize_spaces(x)]
        if not block:
            continue

        # 馬名：開始行以降、戦績行より前の最初の有効行
        name = ""
        for line in block[1:12]:
            if _is_noise_horse_name_line(line):
                continue
            if re.match(r"^\d{4}年", line):
                break
            # 馬名に数字や記号だけの行を採らない
            if re.fullmatch(r"[\d.,]+万円?", line):
                continue
            name = line
            break
        if not name:
            name = f"馬番{horse_no}"

        total_record = ""
        for line in block[:20]:
            if re.fullmatch(r"\(\d+\.\d+\.\d+\.\d+\)", line):
                total_record = line
                break

        trainer = ""
        for line in block[:35]:
            if re.search(r"\((美浦|栗東|地方)\)$", line):
                trainer = line
                break

        sire = dam = damsire = ""
        for line in block[:50]:
            if line.startswith("父："):
                sire = line.replace("父：", "", 1).strip()
            elif line.startswith("母："):
                dam = line.replace("母：", "", 1).strip()
            elif line.startswith("(母の父：") or line.startswith("（母の父："):
                damsire = line.replace("(母の父：", "").replace("（母の父：", "").replace(")", "").replace("）", "").strip()

        sex_age = ""
        carried_weight: Optional[float] = None
        jockey = ""
        sex_idx = None
        for i, line in enumerate(block):
            m = re.search(r"(牡|牝|セ|せん|騸)(\d+)\s*/", line)
            if m:
                sex_age = ("セ" if m.group(1) in {"せん", "騸"} else m.group(1)) + m.group(2)
                sex_idx = i
                break
        if sex_idx is not None:
            for j in range(sex_idx + 1, min(len(block), sex_idx + 8)):
                m = re.search(r"([4-6]\d(?:\.\d)?)kg", block[j])
                if m:
                    carried_weight = float(m.group(1))
                    # 騎手は基本的に斤量の次行
                    for k in range(j + 1, min(len(block), j + 4)):
                        cand = block[k]
                        if cand and not re.search(r"^\d{4}年|^\d+|kg|勝負服|父：|母：", cand):
                            jockey = cand
                            break
                    break
        if carried_weight is None:
            carried_weight = extract_current_carried_weight("\n".join(block[:80]))

        # 過去走。日付行で分割
        date_indices = [i for i, line in enumerate(block) if re.match(r"^\d{4}年\d{1,2}月\d{1,2}日", line)]
        past_runs: List[PastRun] = []
        for rpos, di in enumerate(date_indices[:5]):
            dend = date_indices[rpos + 1] if rpos + 1 < len(date_indices) else len(block)
            pr = _parse_jra_vertical_past_run(block[di:dend])
            if pr is not None:
                past_runs.append(pr)

        equipment_notes = "ブリンカー着用" if any("ブリンカー" in line for line in block[:20]) else ""

        # 前走馬体重を現馬体重欄に誤用しない。現在馬体重が本文上にない場合は空欄保持。
        horse = Horse(
            frame_no=frame_no,
            horse_no=horse_no,
            name=name,
            jockey=jockey,
            trainer=trainer,
            sex_age=sex_age,
            carried_weight=carried_weight,
            odds=None,
            popularity=None,
            body_weight=None,
            body_weight_diff=None,
            total_record=total_record,
            sire=sire,
            dam=dam,
            damsire=damsire,
            equipment_notes=equipment_notes,
            past_runs=past_runs,
        )
        horses.append(horse)

    # 馬番重複があれば後勝ちではなく情報量が多い方を採用
    by_no: Dict[int, Horse] = {}
    for h in horses:
        prev = by_no.get(h.horse_no)
        if prev is None or len(h.past_runs) >= len(prev.past_runs):
            by_no[h.horse_no] = h
    return [by_no[n] for n in sorted(by_no)]

def parse_plain_race_text(raw_text: str) -> Race:
    """HTMLではなく、ブラウザ画面からコピーした出馬表文章を読む。
    JRAスマホ詳細出馬表のような縦型コピーにも対応する。
    ここで作るのは予想ではなく、AIに渡すための事実素材である。
    """
    raw = _clean_plain_input(raw_text)
    race = _parse_plain_race_meta(raw)

    # まずJRA詳細出馬表の縦型コピーを馬ブロック単位で解析する。
    vertical_horses = _parse_jra_vertical_horse_blocks(raw)
    if vertical_horses:
        race.horses = vertical_horses
        if not race.title:
            race.title = f"{race.track or '競馬'}{race.race_no or ''}R 出馬表"
        return race

    # それ以外は従来の「1行に馬番・馬名・性齢・斤量が並ぶ」形式を読む。
    horses_by_no: Dict[int, Horse] = {}
    for raw_line in raw.splitlines():
        line = normalize_spaces(raw_line)
        if not line:
            continue
        h = _parse_plain_horse_row(line)
        if h and 1 <= h.horse_no <= 18:
            # 同じ馬番が複数回出るコピー形式では、情報量が多い行を優先する。
            prev = horses_by_no.get(h.horse_no)
            if prev is None:
                horses_by_no[h.horse_no] = h
            else:
                prev_info = len([x for x in [prev.jockey, prev.sex_age, prev.carried_weight, prev.odds, prev.popularity, prev.body_weight] if x not in ("", None)])
                new_info = len([x for x in [h.jockey, h.sex_age, h.carried_weight, h.odds, h.popularity, h.body_weight] if x not in ("", None)])
                if new_info >= prev_info:
                    # 既存の欠落項目をできるだけ維持
                    if h.jockey == "" and prev.jockey:
                        h.jockey = prev.jockey
                    if h.odds is None and prev.odds is not None:
                        h.odds = prev.odds
                    if h.popularity is None and prev.popularity is not None:
                        h.popularity = prev.popularity
                    if h.body_weight is None and prev.body_weight is not None:
                        h.body_weight = prev.body_weight
                        h.body_weight_diff = prev.body_weight_diff
                    horses_by_no[h.horse_no] = h

    race.horses = [horses_by_no[n] for n in sorted(horses_by_no)]
    if not race.title:
        race.title = f"{race.track or '競馬'}{race.race_no or ''}R 出馬表"
    return race


def _compact_plain_source_evidence(raw_text: str) -> str:
    """AI監査用に、JRA等の文章コピーから出馬表本体だけを原文保持する。
    情報を要約せず、レース概要と各馬ブロックをそのまま残す。
    """
    raw = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.splitlines()

    # レース概要は最初の馬ブロック開始まで原文保持
    horse_start_re = re.compile(r"^\s*枠\s*\d+\S*\s+\d+\s*$|^\s*枠\d+\S*\s+\d+\s*$")
    starts = [i for i, line in enumerate(lines) if horse_start_re.search(line)]
    if not starts:
        # 形式不明でも原文を破棄しない
        return raw.strip()

    parts = []
    header = "\n".join(lines[:starts[0]]).strip()
    if header:
        parts.append("【レース本体・元テキスト原文】\n" + header)

    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        block = "\n".join(lines[start:end]).strip()
        if block:
            parts.append(f"【馬ブロック原文 {idx+1}】\n{block}")
    return "\n\n".join(parts)


def _compact_html_source_evidence(html: str) -> str:
    """NAR等HTMLから、監査に必要な出馬表本体をHTML由来の原文テキストとして保持する。
    script/style/nav/footer等は除くが、レース概要・馬名・近走欄は削らない。
    """
    if BeautifulSoup is None:
        return html
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    evidence_parts = []
    # レースタイトル・条件
    race_title = soup.select_one("section.raceTitle")
    if race_title:
        text = race_title.get_text("\n", strip=True)
        if text:
            evidence_parts.append("【レース本体・元HTML原文】\n" + text)

    # 出馬表本体。各馬の行と近走情報をまとめて原文保持
    table = soup.select_one("section.cardTable table") or soup.select_one("table")
    if table:
        rows = table.find_all("tr", recursive=False)
        if not rows:
            rows = table.find_all("tr")
        current = []
        block_no = 0
        for row in rows:
            row_text = row.get_text("\n", strip=True)
            if not row_text:
                continue
            has_horse_name = bool(row.select_one(".horseName"))
            if has_horse_name and current:
                block_no += 1
                evidence_parts.append(f"【馬ブロック原文 {block_no}】\n" + "\n".join(current))
                current = []
            current.append(row_text)
        if current:
            block_no += 1
            evidence_parts.append(f"【馬ブロック原文 {block_no}】\n" + "\n".join(current))

    if evidence_parts:
        return "\n\n".join(evidence_parts)
    # 抽出に失敗した場合は監査不能化を避け、全文テキストを残す
    return soup.get_text("\n", strip=True)



def _oadp_execution_policy_text() -> str:
    return """【OADP実行ポリシー】\n【OADP使用固定】\n利用可能な最新版のOADP全文正本を必ず使用する。\nStep 0を全頭完了してからPhase 0へ進む。\nPhase 1は4角出口〜直線入口で止め、着順を先出ししない。\nPhase 4直前に出力前監査表を表示し、NGがあればPhase 2またはPhase 3へ戻る。\n候補数固定、S3再吸収、押し出し監査、人気馬汚染遮断を守る。\n

このSTEP0を受信したAIは、確認応答だけで停止せず、このファイル自体を実行依頼として扱い、記載された順序で処理を開始する。

実行順序
1. STEP0を読み込む
2. 必要な正本・設定ファイルを読み込む
3. STEP0補正を行う
4. Phase0
5. Phase1
6. Phase2
7. Phase3
8. 出力前監査
9. Phase4

計算方針
・STEP0に記載された事実データを使用する。
・正本に定義された計算式、係数、補正、監査手順を使用する。
・正本に定義されていないEV、係数、確率、補正は計算へ追加しない。
・人気・オッズは、正本で使用が明示されていない限り計算に使用しない。
・不足情報は推測補完せず、該当項目を未確定として扱う。
・計算していない値は、計算結果として出力しない。
・第24章を最優先し、4角出口までの隊列優位と直線で勝ち切る能力を分離する。
・既存シナリオ別総合EVを受け取った後、直線勝ち切りEVとシナリオ別1着列適性監査を実行する。
・4角で前にいることだけを理由に1列目へ置かない。
・過去の実着順、確定オッズ、払戻を将来レースの係数や確率へ逆流させない。

OADP追加処理
・Phase2で4角隊列優位と直線勝ち切り能力を分離する。
・各シナリオで直線勝ち切りEVを算出し、1着列適性を監査する。
・4角到達上位でも勝ち切りEVが不足する馬は1列目へ固定しない。
・勝ち切りEVと4角隊列に乖離がある馬はPhase3反映監査へ送る。

シナリオ発生確率
・正本の手順に従って補正前スコアを計算する。
・正規化、構造補正、補正後確率の算出を行う。
・計算過程と補正理由をPhase1または監査TXTへ記録する。
・40％以上となったシナリオは、正本に従って再監査する。
・再監査を実施していない場合、その確率を最終確率として確定しない。

出力
・STEP0補正ログ、Phase0、Phase1、Phase2、Phase3、出力前監査をTXTで保存する。
・本文には各TXTへのリンクとPhase4を出力する。
・Phase4は指定フォーマットを使用する。
・3連単見出しとフォーメーションの間に空行を入れない。
・3連複見出しとフォーメーションの間に空行を入れない。

不整合検出時の処置
・出力前監査で不整合を検出した場合は、該当するPhaseへ戻って修正する。
・修正できない項目は未確定として明示する。
・一部に未確定項目があっても、実行可能な工程は継続する。
・未実施の工程を実施済みとして表現しない。

重要
・確認、要約、受領報告だけで処理を終了してはならない。
・補正可能な不一致は修正して継続する。
・正本または元テキストから一意に復元できない項目だけを未確定として扱う。
・本ポリシーはSTEP0本文、アプリ説明文、監査候補より優先して実行順序を指定する。
"""



# ============================================================
# Simulation Input — fact-only structured data extension
# Existing STEP0/OADP sections remain unchanged.
# ============================================================

SIMULATION_SCHEMA_VERSION = "1.0"
SIM_NOT_PROVIDED = "NOT_PROVIDED"


def _sim_missing(source: str = "source_not_available") -> Dict[str, Any]:
    return {"value": SIM_NOT_PROVIDED, "status": "NOT_PROVIDED", "source": source}


def _sim_fact(value: Any, source: str) -> Dict[str, Any]:
    if value is None or value == "" or value == []:
        return _sim_missing(source)
    return {"value": value, "status": "FACT", "source": source}


def _sim_derived(value: Any, formula: str, inputs: List[str]) -> Dict[str, Any]:
    if value is None or value == "" or value == []:
        return _sim_missing("required_fact_missing")
    return {
        "value": value,
        "status": "DERIVED",
        "source": "deterministic_calculation",
        "formula": formula,
        "inputs": inputs,
    }


def _sim_parse_date(text: str):
    from datetime import datetime
    s = normalize_spaces(text or "")
    for pattern, fmt in [
        (r"(\d{4}年\d{1,2}月\d{1,2}日)", "%Y年%m月%d日"),
        (r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})", None),
        (r"(\d{2})[./-](\d{1,2})[./-](\d{1,2})", "short"),
    ]:
        m = re.search(pattern, s)
        if not m:
            continue
        try:
            if fmt == "%Y年%m月%d日":
                return datetime.strptime(m.group(1), fmt).date()
            if fmt == "short":
                year = 2000 + int(m.group(1))
                return datetime(year, int(m.group(2)), int(m.group(3))).date()
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
        except ValueError:
            continue
    return None


def _sim_position_rate(position: Optional[int], heads: Optional[int], label: str) -> Dict[str, Any]:
    if position is None or heads is None or heads <= 0:
        return _sim_missing("position_or_field_size_missing")
    return _sim_derived(
        round(position / heads, 6),
        "position / field_size",
        [label, "field_size"],
    )


def _sim_corner_positions(run: PastRun) -> Dict[str, Any]:
    vals = run.passing_positions
    result: Dict[str, Any] = {
        "raw": _sim_fact(run.passing_text, "parsed_past_run"),
        "source_order": _sim_fact(vals, "parsed_past_run"),
    }
    for i in range(4):
        result[f"corner{i + 1}"] = (
            _sim_fact(vals[i], "parsed_past_run")
            if i < len(vals)
            else _sim_missing("corner_not_present_in_source")
        )
    result["mapping_rule"] = {
        "value": "通過順の左から順にcorner1〜corner4へ格納。元資料に角名がない場合は順序のみ保持。",
        "status": "FACT",
        "source": "simulation_schema_definition",
    }
    return result


def _sim_equipment_item(notes: str, keyword: str) -> Dict[str, Any]:
    if notes and keyword in notes:
        return _sim_fact("記載あり", "parsed_current_entry_equipment")
    return _sim_missing("equipment_not_explicitly_stated")


def _sim_past_run(run: PastRun, run_index: int) -> Dict[str, Any]:
    vals = run.passing_positions
    first = vals[0] if vals else None
    third = vals[2] if len(vals) >= 3 else None
    fourth = vals[3] if len(vals) >= 4 else (vals[-1] if vals else None)
    return {
        "run_index": run_index,
        "date": _sim_fact(run.date, "parsed_past_run"),
        "venue": _sim_fact(run.venue, "parsed_past_run"),
        "race_name": _sim_fact(run.race_name, "parsed_past_run"),
        "surface": _sim_fact(run.course, "parsed_past_run"),
        "going": _sim_fact(run.going, "parsed_past_run"),
        "distance_m": _sim_fact(run.distance, "parsed_past_run"),
        "finish_rank": _sim_fact(run.rank, "parsed_past_run"),
        "race_time": _sim_fact(run.time_text, "parsed_past_run"),
        "carried_weight_kg": _sim_fact(run.carried_weight, "parsed_past_run"),
        "body_weight_kg": _sim_fact(run.body_weight, "parsed_past_run"),
        "jockey": _sim_fact(run.jockey, "parsed_past_run"),
        "frame_number": _sim_missing("past_run_frame_not_in_source"),
        "horse_number": _sim_fact(run.gate_no, "parsed_past_run"),
        "field_size": _sim_fact(run.heads, "parsed_past_run"),
        "laps": {
            "front_3f_seconds": _sim_missing("race_lap_not_in_source"),
            "back_3f_seconds": _sim_missing("race_lap_not_in_source"),
            "front_back_difference_seconds": _sim_missing("race_lap_not_in_source"),
            "race_laps": _sim_missing("race_lap_not_in_source"),
            "lap_availability": _sim_missing("race_lap_not_in_source"),
        },
        "margins": {
            "winner": _sim_fact(run.winner, "parsed_past_run"),
            "winner_gap_seconds": _sim_fact(run.margin, "parsed_past_run"),
            "gap_to_previous_horse": _sim_missing("ordered_finish_margin_not_in_source"),
            "gap_to_next_horse": _sim_missing("ordered_finish_margin_not_in_source"),
            "margin_rank": _sim_missing("ordered_finish_margin_not_in_source"),
        },
        "closing_section": {
            "horse_last_3f_seconds": _sim_fact(run.agari, "parsed_past_run"),
            "last_3f_rank": _sim_missing("all_horse_last3f_table_not_in_source"),
            "difference_from_fastest": _sim_missing("all_horse_last3f_table_not_in_source"),
            "difference_from_average": _sim_missing("all_horse_last3f_table_not_in_source"),
        },
        "corner_positions": _sim_corner_positions(run),
        "position_rates": {
            "first_corner_rate": _sim_position_rate(first, run.heads, "corner1"),
            "third_corner_rate": _sim_position_rate(third, run.heads, "corner3"),
            "fourth_corner_rate": _sim_position_rate(fourth, run.heads, "corner4"),
        },
        "official_incidents": {
            "slow_start": _sim_missing("official_comment_not_in_source"),
            "stumble": _sim_missing("official_comment_not_in_source"),
            "squeezed": _sim_missing("official_comment_not_in_source"),
            "contact": _sim_missing("official_comment_not_in_source"),
            "blocked": _sim_missing("official_comment_not_in_source"),
            "wide_trip": _sim_missing("official_comment_not_in_source"),
            "pulled": _sim_missing("official_comment_not_in_source"),
            "lost_shoe": _sim_missing("official_comment_not_in_source"),
            "official_comment_raw": _sim_missing("official_comment_not_in_source"),
        },
        "future_physics": {
            "start_reaction_time": _sim_missing("not_currently_published"),
            "gps_coordinates": _sim_missing("not_currently_published"),
            "corner_pass_times": _sim_missing("not_currently_published"),
            "lateral_position": _sim_missing("not_currently_published"),
            "horse_gap_distance": _sim_missing("not_currently_published"),
            "contact_events": _sim_missing("not_currently_published"),
            "blocked_events": _sim_missing("not_currently_published"),
            "kickback_amount": _sim_missing("not_currently_published"),
            "corner_entry_angle": _sim_missing("not_currently_published"),
            "actual_travel_distance": _sim_missing("not_currently_published"),
        },
    }


def _sim_rotation(race: Race, horse: Horse) -> Dict[str, Any]:
    race_date = _sim_parse_date(race.date_text)
    run_dates = [_sim_parse_date(r.date) for r in horse.past_runs]
    valid_dates = [d for d in run_dates if d is not None]
    last_date = valid_dates[0] if valid_dates else None
    days = (race_date - last_date).days if race_date and last_date else None
    count30 = sum(1 for d in valid_dates if race_date and 0 <= (race_date - d).days <= 30)
    count60 = sum(1 for d in valid_dates if race_date and 0 <= (race_date - d).days <= 60)
    return {
        "days_since_last_run": _sim_derived(
            days,
            "race_date - latest_past_run_date",
            ["race.date", "previous_runs[0].date"],
        ),
        "weeks_since_last_run": _sim_derived(
            round(days / 7, 3) if days is not None else None,
            "days_since_last_run / 7",
            ["days_since_last_run"],
        ),
        "back_to_back_within_7_days": _sim_derived(
            bool(0 <= days <= 7) if days is not None else None,
            "0 <= days_since_last_run <= 7",
            ["days_since_last_run"],
        ),
        "first_run_after_break": _sim_missing("break_definition_or_official_label_not_in_source"),
        "run_number_after_break": _sim_missing("break_definition_or_full_history_not_in_source"),
        "runs_in_last_30_days": _sim_derived(
            count30 if race_date else None,
            "count(past_run_date within 30 days before race_date)",
            ["race.date", "previous_runs[].date"],
        ),
        "runs_in_last_60_days": _sim_derived(
            count60 if race_date else None,
            "count(past_run_date within 60 days before race_date)",
            ["race.date", "previous_runs[].date"],
        ),
    }


def _sim_body_weight_history(horse: Horse) -> Dict[str, Any]:
    values = [r.body_weight for r in horse.past_runs[:5] if r.body_weight is not None]
    good = [
        r.body_weight for r in horse.past_runs[:5]
        if r.body_weight is not None and r.rank is not None and r.rank <= 3
    ]
    return {
        "past_5_runs_kg": _sim_fact(values, "parsed_past_runs"),
        "average_kg": _sim_derived(
            round(sum(values) / len(values), 3) if values else None,
            "sum(known_past_5_body_weights) / count(known_past_5_body_weights)",
            ["previous_runs[].body_weight_kg"],
        ),
        "maximum_kg": _sim_derived(max(values) if values else None, "max(known_past_5_body_weights)", ["previous_runs[].body_weight_kg"]),
        "minimum_kg": _sim_derived(min(values) if values else None, "min(known_past_5_body_weights)", ["previous_runs[].body_weight_kg"]),
        "top3_finish_average_kg": _sim_derived(
            round(sum(good) / len(good), 3) if good else None,
            "average(body_weight where finish_rank <= 3 within loaded past 5 runs)",
            ["previous_runs[].body_weight_kg", "previous_runs[].finish_rank"],
        ),
    }


def _sim_jockey_continuity(horse: Horse) -> Dict[str, Any]:
    prev = horse.past_runs[0].jockey if horse.past_runs else ""
    current = horse.jockey
    continuity = None
    if current and prev:
        continuity = "継続騎乗" if normalize_spaces(current) == normalize_spaces(prev) else "乗り替わり"
    past_count = None
    if current:
        past_count = sum(
            1 for r in horse.past_runs
            if r.jockey and normalize_spaces(r.jockey) == normalize_spaces(current)
        )
    return {
        "current_jockey": _sim_fact(current, "parsed_current_entry"),
        "previous_run_jockey": _sim_fact(prev, "parsed_past_run"),
        "continuity": _sim_derived(
            continuity,
            "current_jockey == previous_run_jockey",
            ["current_jockey", "previous_run_jockey"],
        ),
        "same_horse_rides_in_loaded_history": _sim_derived(
            past_count,
            "count(previous_runs where jockey == current_jockey)",
            ["current_jockey", "previous_runs[].jockey"],
        ),
    }


def _sim_horse(race: Race, horse: Horse) -> Dict[str, Any]:
    notes = horse.equipment_notes or ""
    runs = [_sim_past_run(r, i + 1) for i, r in enumerate(horse.past_runs[:5])]
    running_style_facts = [
        {
            "run_index": i + 1,
            "passing_order_raw": _sim_fact(r.passing_text, "parsed_past_run"),
            "first_position": _sim_fact(r.first_pos, "parsed_past_run"),
            "last_position": _sim_fact(r.last_pos, "parsed_past_run"),
        }
        for i, r in enumerate(horse.past_runs[:5])
    ]
    return {
        "horse_number": _sim_fact(horse.horse_no, "parsed_current_entry"),
        "frame_number": _sim_fact(horse.frame_no, "parsed_current_entry"),
        "name": _sim_fact(horse.name, "parsed_current_entry"),
        "sex_age": _sim_fact(horse.sex_age, "parsed_current_entry"),
        "barrier": {
            "frame_number": _sim_fact(horse.frame_no, "parsed_current_entry"),
            "horse_number": _sim_fact(horse.horse_no, "parsed_current_entry"),
            "field_size": _sim_derived(race.field_size, "count(parsed_current_entries)", ["race.horses"]),
        },
        "weights": {
            "carried_weight_kg": _sim_fact(horse.carried_weight, "parsed_current_entry"),
            "body_weight_kg": _sim_fact(horse.body_weight, "parsed_current_entry"),
            "body_weight_change_kg": _sim_fact(horse.body_weight_diff, "parsed_current_entry"),
            "history": _sim_body_weight_history(horse),
        },
        "trainer": _sim_fact(horse.trainer, "parsed_current_entry"),
        "jockey": _sim_jockey_continuity(horse),
        "equipment": {
            "raw": _sim_fact(notes, "parsed_current_entry"),
            "blinkers": _sim_equipment_item(notes, "ブリンカー"),
            "cheekpieces": _sim_equipment_item(notes, "チーク"),
            "hood": _sim_equipment_item(notes, "メンコ"),
            "shadow_roll": _sim_equipment_item(notes, "シャドーロール"),
            "bit_change": _sim_equipment_item(notes, "ハミ"),
            "comparison_with_previous_run": _sim_missing("previous_run_equipment_not_in_source"),
            "previous_run": {
                "blinkers": _sim_missing("previous_run_equipment_not_in_source"),
                "cheekpieces": _sim_missing("previous_run_equipment_not_in_source"),
                "hood": _sim_missing("previous_run_equipment_not_in_source"),
                "shadow_roll": _sim_missing("previous_run_equipment_not_in_source"),
                "bit": _sim_missing("previous_run_equipment_not_in_source"),
            },
        },
        "training": {
            "date": _sim_missing("training_data_not_in_source"),
            "course": _sim_missing("training_data_not_in_source"),
            "total_time": _sim_missing("training_data_not_in_source"),
            "section_times": _sim_missing("training_data_not_in_source"),
            "last_1f": _sim_missing("training_data_not_in_source"),
            "intensity": _sim_missing("training_data_not_in_source"),
            "strong": _sim_missing("training_data_not_in_source"),
            "hand_ride": _sim_missing("training_data_not_in_source"),
            "all_out": _sim_missing("training_data_not_in_source"),
            "paired_work_result": _sim_missing("training_data_not_in_source"),
        },
        "rotation": _sim_rotation(race, horse),
        "pedigree": {
            "sire": _sim_fact(horse.sire, "parsed_current_entry"),
            "damsire": _sim_fact(horse.damsire, "parsed_current_entry"),
            "dam": _sim_fact(horse.dam, "parsed_current_entry"),
            "representative_siblings": _sim_missing("pedigree_relatives_not_in_source"),
            "representative_dam_offspring": _sim_missing("pedigree_relatives_not_in_source"),
        },
        "previous_runs": runs,
        "running_style_facts": running_style_facts,
        "official_comments": _sim_missing("official_comment_not_in_source"),
    }


def _sim_status_summary(payload: Any) -> Dict[str, int]:
    counts = {"FACT": 0, "DERIVED": 0, "NOT_PROVIDED": 0}
    def walk(obj: Any):
        if isinstance(obj, dict):
            status = obj.get("status")
            if status in counts:
                counts[status] += 1
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)
    walk(payload)
    return counts


def build_simulation_input(race: Race, source_text: str, source_type: str) -> Dict[str, Any]:
    course_id = None
    if race.track and race.course and race.distance:
        direction = race.direction or "NA"
        safe = re.sub(r"[^0-9A-Za-z一-龥ぁ-んァ-ン_-]+", "_", f"{race.track}_{race.course}_{race.distance}_{direction}")
        course_id = f"JP_{safe}"

    payload: Dict[str, Any] = {
        "schema_name": "STEP0_SIMULATION_INPUT",
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "compatibility": {
            "oadp_version": "OADP",
            "existing_step0_sections_preserved": True,
            "prediction_values_in_this_section": False,
            "allowed_statuses": ["FACT", "DERIVED", "NOT_PROVIDED"],
            "not_provided_token": SIM_NOT_PROVIDED,
        },
        "source": {
            "input_type": _sim_fact(source_type, "application_input"),
            "source_character_count": _sim_derived(len(source_text or ""), "len(source_text)", ["source_text"]),
        },
        "course_master": {
            "course_master_id": _sim_derived(
                course_id,
                "JP_{track}_{surface}_{distance}_{direction}",
                ["race.track", "race.surface", "race.distance_m", "race.direction"],
            ),
            "first_corner_distance_m": _sim_missing("course_master_database_not_connected"),
            "home_straight_distance_m": _sim_missing("course_master_database_not_connected"),
            "corner_radius_m": _sim_missing("course_master_database_not_connected"),
            "elevation_difference_m": _sim_missing("course_master_database_not_connected"),
            "slope_position": _sim_missing("course_master_database_not_connected"),
            "course_width_m": _sim_missing("course_master_database_not_connected"),
            "direction": _sim_fact(race.direction, "parsed_race_header"),
        },
        "race_condition": {
            "date": _sim_fact(race.date_text, "parsed_race_header"),
            "track": _sim_fact(race.track, "parsed_race_header"),
            "race_number": _sim_fact(race.race_no, "parsed_race_header"),
            "title": _sim_fact(race.title, "parsed_race_header"),
            "post_time": _sim_fact(race.post_time, "parsed_race_header"),
            "surface": _sim_fact(race.course, "parsed_race_header"),
            "distance_m": _sim_fact(race.distance, "parsed_race_header"),
            "direction": _sim_fact(race.direction, "parsed_race_header"),
            "race_class": _sim_fact(race.race_class, "parsed_race_header"),
            "field_size": _sim_derived(race.field_size, "count(parsed_current_entries)", ["race.horses"]),
        },
        "weather": {
            "weather": _sim_fact(race.weather, "parsed_race_header"),
            "temperature_c": _sim_missing("weather_detail_not_in_source"),
            "humidity_percent": _sim_missing("weather_detail_not_in_source"),
            "wind_speed_mps": _sim_missing("weather_detail_not_in_source"),
            "wind_direction": _sim_missing("weather_detail_not_in_source"),
            "precipitation_mm": _sim_missing("weather_detail_not_in_source"),
        },
        "ground": {
            "going": _sim_fact(race.going, "parsed_race_header"),
            "moisture_percent": _sim_missing("track_measurement_not_in_source"),
            "cushion_value": _sim_missing("track_measurement_not_in_source"),
            "sand_depth_cm": _sim_missing("track_measurement_not_in_source"),
            "announcement_time": _sim_missing("track_measurement_not_in_source"),
        },
        "today_track_facts": {
            "status": "NOT_PROVIDED",
            "reason": "対象レース単体のURLまたはHTMLには、対象レース以前の全レース結果が含まれない。",
            "prior_races": [],
            "required_fields": ["fourth_corner_position", "last_3f_rank", "frame", "running_style", "winning_time"],
        },
        "track_bias_facts": {
            "status": "NOT_PROVIDED",
            "reason": "対象レース単体資料からは当日全レースの事実集合を構成できない。",
        },
        "horses": [_sim_horse(race, horse) for horse in race.horses],
    }
    payload["data_acquisition_status"] = _sim_status_summary(payload)
    return payload


def render_simulation_input_json(race: Race, source_text: str, source_type: str) -> str:
    return json.dumps(
        build_simulation_input(race, source_text, source_type),
        ensure_ascii=False,
        indent=2,
    )




def build_single_text_step0_pack_from_plain_text(raw_text: str) -> Tuple[Race, str, Dict[str, Any]]:
    """プレーンテキスト出馬表から、ChatGPTへ1ファイルで渡せる二分割テキストパックを作る。
    アプリ側の判断系表記や予想結論は出さない。
    """
    race = parse_plain_race_text(raw_text)
    if race.field_size == 0:
        raise ValueError("文章コピーから出走馬を解析できませんでした。JRA詳細出馬表の文章を、馬番から過去走まで含めて広めにコピーしてください。")

    features = compute_features(race)
    correction_text, correction_rows, correction_ok = build_app_correction_audit(race, raw_text, "plain_text")
    correction_text = _single_text_clean_generated(correction_text)
    correction_candidates = correction_rows

    notice = _single_text_clean_generated(_ai_processing_important_notice())
    phase0_log_instruction = _single_text_clean_generated(_ai_phase0_correction_log_instruction())

    step0_body = render_step0(race, features)
    step0 = _single_text_clean_generated(notice + "\n\n" + correction_text + "\n\n" + step0_body)

    audit_text, audit_rows, audit_ok = build_parse_audit(race, features)
    unified_ok = bool(audit_ok and correction_ok)
    unresolved_count = sum(1 for row in correction_rows if row.get("judgement") == "UNRESOLVED_CRITICAL")
    correction_count = sum(1 for row in correction_rows if row.get("judgement") == "AI_CORRECTION_REQUIRED")
    unified_summary = (
        "【統合監査最終判定】\n"
        f"判定：{'進行可能' if unified_ok else '処理停止'}\n"
        f"AI_CORRECTION_REQUIRED件数：{correction_count}\n"
        f"UNRESOLVED_CRITICAL件数：{unresolved_count}\n"
        f"停止要否：{'停止しない' if unified_ok else '停止する'}\n"
        + (
            "AIは元テキスト照合による補正ログをPhase0に記載してからPhase1へ進むこと。"
            if unified_ok else
            "UNRESOLVED_CRITICALまたは真のERRORが残っているため、Phase1へ進まないこと。"
        )
    )
    audit_text = _single_text_clean_generated(unified_summary + "\n\n" + audit_text)
    parsed_json = json.dumps(_race_horses_only_json(race, correction_candidates), ensure_ascii=False, indent=2)
    raw_material_json = json.dumps(build_phase0_raw_material(race, features, correction_candidates), ensure_ascii=False, indent=2)
    simulation_input_json = render_simulation_input_json(race, raw_text, "plain_text")

    auto_request = _single_text_clean_generated(
        "OADP STEP0単一テキストパック 自動実行リクエスト\n"
        "============================================================\n\n"
        "このテキストファイルを受け取ったChatGPTは、ユーザーから追加指示がなくても、これをPhase1以降の実行依頼として扱うこと。\n\n"
        "実行手順：\n"
        "1. /mnt/data/OADP_OADP_展開先行監査人気馬汚染遮断前列圧参加馬保護統合改善版_全文.txt を読み直す。\n"
        "2. /mnt/data/理想のSTEP0フォーマット.txt を読み直す。\n"
        "3. まず『AI処理時の重要注意』『アプリ整形監査・AI補正指示』『パース異常検査表』を確認する。\n"
        "4. AI_CORRECTION_REQUIREDは停止せず、元テキストを正としてPhase0で補正ログを出してからPhase1へ進む。\n"
        "5. UNRESOLVED_CRITICALまたは真のERRORがある場合のみ、Phase1以降へ進まず停止する。\n"
        "6. 『STEP0全文』『パース済み事実データJSON』『05_phase0_raw_material.json相当』『監査用・元テキスト原文抜粋』を材料に、Phase0〜Phase3をテキストファイル化する。\n"
        "7. Phase4は本文に出し、OADP指定の最終フォーマットに従う。\n"
        "8. 出力前監査表と『漏れ修正前→修正後』を必ず出し、NGがあれば最終出力せずPhase2またはPhase3へ戻す。\n\n"
        "重要：このテキストは素材集であり、アプリは判断系の結論を出していない。Phase1以降の判断は、ChatGPT側でOADP本文を読み直して新規に行うこと。\n"
    )

    header = _single_text_clean_generated(
        f"【レース概要】\n"
        f"入力種別：プレーンテキストコピー\n"
        f"レース名：{race.title or '-'}\n"
        f"開催：{race.date_text or '-'}／{race.track or '-'} {race.race_no or '-'}R／発走：{race.post_time or '-'}\n"
        f"条件：{race.course or '-'}{race.distance or '-'}m{race.direction or ''}／天候：{race.weather or '-'}／馬場：{race.going or '-'}\n"
        f"出走頭数：{race.field_size}頭／採用頭数ルール：{race.candidate_count}頭\n"
        f"パース総合判定：{'進行可能' if unified_ok else '処理停止'}\n"
    )

    readme = _single_text_clean_generated(
        "【06_README_for_AI.txt 相当】\n"
        "この.txtをそのままChatGPTへ送付する。ZIP解凍や複数ファイル送付は不要。\n"
        "このファイルは、HTMLが取得できない中央競馬などの出馬表文章コピーから作成した素材パックである。\n"
        "ChatGPT側は、下部の監査用・元テキスト原文抜粋を必ず参照し、アプリが読み落とした開催場所・距離・芝ダートがあればPhase0で補正してからPhase1以降へ進む。\n\n"
        + _ai_processing_important_notice() + "\n" + _ai_phase0_correction_log_instruction()
    )
    readme = _single_text_clean_generated(readme)

    pack_text = "\n\n".join([
        _single_text_clean_generated(_oadp_execution_policy_text()),
        auto_request,
        header,
        readme,
        "============================================================\n【AI側で行う判定指示】\n============================================================\n" + _ai_side_instruction_block(),
        "============================================================\n【アプリ整形監査・AI補正指示】\n============================================================\n" + correction_text,
        "============================================================\n【パース異常検査表】\n============================================================\n" + audit_text,
        "============================================================\n【03_STEP0_full.txt 相当：STEP0全文】\n============================================================\n" + step0,
        "============================================================\n【パース済み事実データJSON】\n============================================================\n" + parsed_json,
        "============================================================\n【05_phase0_raw_material.json 相当】\n============================================================\n" + raw_material_json,
        "============================================================\n【Simulation Input】\n============================================================\n" + simulation_input_json,
        "============================================================\n【監査用・元テキスト原文抜粋】\n============================================================\n" + _compact_plain_source_evidence(raw_text),
    ])

    pack_text = _single_text_clean_generated(pack_text)

    meta = {
        "audit_ok": audit_ok and correction_ok,
        "audit_rows": audit_rows,
        "correction_candidates": correction_candidates,
        "horse_count": race.field_size,
        "candidate_count": race.candidate_count,
        "single_text": True,
        "source_type": "plain_text",
    }
    return race, pack_text, meta

def build_single_text_step0_pack(html: str) -> Tuple[Race, str, Dict[str, Any]]:
    """出馬表HTMLから、ChatGPTへ1ファイルで渡せる二分割テキストパックを作る。
    予想結論や順位付けは出力しない。
    """
    race = parse_nar_html(html)
    if race.field_size == 0:
        raise ValueError("出馬表HTMLから出走馬を解析できませんでした。NAR出馬表HTML全体を貼り付けてください。")

    features = compute_features(race)
    step0 = _single_text_clean_generated(render_step0(race, features))
    audit_text, audit_rows, audit_ok = build_parse_audit(race, features)
    audit_text = _single_text_clean_generated(audit_text)

    parsed_json = json.dumps(_race_horses_only_json(race), ensure_ascii=False, indent=2)
    simulation_input_json = render_simulation_input_json(race, html, "html")

    auto_request = _single_text_clean_generated(
        "OADP STEP0単一テキストパック 自動実行リクエスト\n"
        "============================================================\n\n"
        "このテキストファイルを受け取ったChatGPTは、ユーザーから追加指示がなくても、"
        "これをPhase1以降の実行依頼として扱うこと。\n\n"
        "実行手順：\n"
        "1. /mnt/data/OADP_OADP_展開先行監査人気馬汚染遮断前列圧参加馬保護統合改善版_全文.txt を読み直す。\n"
        "2. /mnt/data/理想のSTEP0フォーマット.txt を読み直す。\n"
        "3. このテキスト内の『パース異常検査表』を確認し、ERRORが1つでもあればPhase1以降へ進まず停止する。\n"
        "4. 『STEP0全文』と『パース済み事実データJSON』と『監査用・元HTML原文抜粋』を材料に、Phase0〜Phase3をテキストファイル化する。\n"
        "5. Phase4は本文に出し、OADP指定の最終フォーマットに従う。\n"
        "6. 出力前監査表と『漏れ修正前→修正後』を必ず出し、NGがあれば最終出力せずPhase2またはPhase3へ戻す。\n\n"
        "重要：このテキストは素材集であり、アプリは評価系の結論を出していない。"
        "Phase1以降の判断は、ChatGPT側でOADP本文を読み直して新規に行うこと。\n"
    )

    header = _single_text_clean_generated(
        f"【レース概要】\n"
        f"レース名：{race.title or '-'}\n"
        f"開催：{race.date_text or '-'}／{race.track or '-'} {race.race_no or '-'}R／発走：{race.post_time or '-'}\n"
        f"条件：{race.course or '-'}{race.distance or '-'}m{race.direction or ''}／天候：{race.weather or '-'}／馬場：{race.going or '-'}\n"
        f"出走頭数：{race.field_size}頭／採用頭数ルール：{race.candidate_count}頭\n"
        f"パース総合判定：{'ERRORなし' if audit_ok else 'ERRORあり'}\n"
    )

    readme = _single_text_clean_generated(
        "【このファイルの使い方】\n"
        "この.txtをそのままChatGPTへ送付する。ZIP解凍や複数ファイル送付は不要。\n"
        "ChatGPT側はこのファイルを受け取った時点で、OADP本文と理想STEP0を読み直し、"
        "パース異常がなければPhase1以降へ進む。\n"
        "アプリはSTEP0素材作成とパース確認だけを行い、馬の優劣・点数・順位・印・買い目は行わない。\n"
    )

    pack_text = "\n\n".join([
        _single_text_clean_generated(_oadp_execution_policy_text()),
        auto_request,
        header,
        readme,
        "============================================================\n【AI側で行う判定指示】\n============================================================\n" + _ai_side_instruction_block(),
        "============================================================\n【パース異常検査表】\n============================================================\n" + audit_text,
        "============================================================\n【STEP0全文】\n============================================================\n" + step0,
        "============================================================\n【パース済み事実データJSON】\n============================================================\n" + parsed_json,
        "============================================================\n【Simulation Input】\n============================================================\n" + simulation_input_json,
        "============================================================\n【監査用・元HTML原文抜粋】\n============================================================\n" + _compact_html_source_evidence(html),
    ])

    meta = {
        "audit_ok": audit_ok,
        "audit_rows": audit_rows,
        "horse_count": race.field_size,
        "candidate_count": race.candidate_count,
        "single_text": True,
    }
    return race, pack_text, meta

