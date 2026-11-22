import math
import os
from datetime import datetime, timezone, timedelta
import requests
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import brentq
from scipy.stats import poisson, nbinom

st.set_page_config(page_title="RUSPRG Auto v4.6 Pro", layout="wide")

TR_TZ = timezone(timedelta(hours=3))

LEAGUES = {
    "Türkiye Süper Lig": "soccer_turkey_super_league",
    "İngiltere Premier Lig": "soccer_epl",
    "İspanya La Liga": "soccer_spain_la_liga",
    "İtalya Serie A": "soccer_italy_serie_a",
    "Almanya Bundesliga": "soccer_germany_bundesliga",
    "Fransa Ligue 1": "soccer_france_ligue_one",
    "UEFA Şampiyonlar Ligi": "soccer_uefa_champs_league",
    "UEFA Avrupa Ligi": "soccer_uefa_europa_league",
    "Hollanda Eredivisie": "soccer_netherlands_eredivisie",
    "Portekiz Primeira Liga": "soccer_portugal_primeira_liga",
}


# ==============================================================================
# 1. İSTATİSTİKSEL MOTOR (P, Q, R, U, W, X, V ve CANLI v4.6)
# ==============================================================================
def safe_fair_odd(prob):
    if prob is None or prob <= 0.001:
        return "-"
    return round(1.0 / prob, 2)


def devig_all(odds):
    inv = np.array([1.0 / o for o in odds], dtype=float)
    s = inv.sum()
    p_prop = inv / s
    if s <= 1.0001:
        return p_prop, max(0.0, s - 1.0)

    k = brentq(lambda x: np.sum(inv ** x) - 1.0, 0.1, 10.0)
    p_pow = inv ** k

    def shin_p(z):
        return (np.sqrt(z**2 + 4 * (1 - z) * (inv**2) / s) - z) / (2 * (1 - z))

    z = brentq(lambda z_: np.sum(shin_p(z_)) - 1.0, 0.0, 0.999)
    p_shin = shin_p(z)
    return (p_prop + p_pow + p_shin) / 3.0, s - 1.0


def invert_odds_to_xg(p_h, p_a, p_over25=None):
    if p_over25 is not None and 0.05 < p_over25 < 0.95:
        mu = brentq(lambda m: 1.0 - math.exp(-m) * (1 + m + 0.5 * m**2) - p_over25, 0.4, 6.0)
    else:
        mu = 2.65

    target_ratio = p_h / max(p_h + p_a, 1e-6)

    def win_ratio_diff(lh):
        la = mu - lh
        mat = np.outer(poisson.pmf(range(9), lh), poisson.pmf(range(9), la))
        ph = np.tril(mat, -1).sum()
        pa = np.triu(mat, 1).sum()
        return (ph / max(ph + pa, 1e-6)) - target_ratio

    try:
        lam_h = brentq(win_ratio_diff, 0.15, mu - 0.15)
    except ValueError:
        lam_h = mu * target_ratio
    return lam_h, mu - lam_h


def dixon_coles_matrix(lam_h, lam_a, rho=-0.10, max_g=8):
    mat = np.zeros((max_g + 1, max_g + 1))
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
            if i == 0 and j == 0:
                tau = 1 - lam_h * lam_a * rho
            elif i == 0 and j == 1:
                tau = 1 + lam_h * rho
            elif i == 1 and j == 0:
                tau = 1 + lam_a * rho
            elif i == 1 and j == 1:
                tau = 1 - rho
            else:
                tau = 1.0
            mat[i, j] = max(0.0, p * tau)
    return mat / mat.sum()


def bivariate_poisson_matrix(mu_h, mu_a, l3=0.12, max_g=8):
    l3 = min(l3, mu_h * 0.45, mu_a * 0.45)
    l1, l2 = max(mu_h - l3, 1e-6), max(mu_a - l3, 1e-6)
    mat = np.zeros((max_g + 1, max_g + 1))
    exp_t = math.exp(-(l1 + l2 + l3))
    for x in range(max_g + 1):
        for y in range(max_g + 1):
            s = sum(
                math.comb(x, k) * math.comb(y, k) * math.factorial(k) * ((l3 / (l1 * l2)) ** k)
                for k in range(min(x, y) + 1)
            )
            mat[x, y] = exp_t * (l1**x / math.factorial(x)) * (l2**y / math.factorial(y)) * s
    return mat / mat.sum()


def model_r_negbin(expected_total, phi=0.08):
    r = 1.0 / max(phi, 1e-6)
    p = r / (r + expected_total)
    probs = [nbinom.pmf(g, r, p) for g in range(7)]
    o15 = 1.0 - sum(probs[:2])
    o25 = 1.0 - sum(probs[:3])
    o35 = 1.0 - sum(probs[:4])
    return {
        "O1.5": o15, "U1.5": 1.0 - o15,
        "O2.5": o25, "U2.5": 1.0 - o25,
        "O3.5": o35, "U3.5": 1.0 - o35,
    }


def model_u_elo_from_fair(p_h, p_a):
    exp_h = np.clip(p_h / max(p_h + p_a, 1e-6), 0.05, 0.95)
    diff = -400.0 * math.log10((1.0 / exp_h) - 1.0)
    p_draw = 0.28 * math.exp(-((diff / 300.0) ** 2))
    return {"1": (1 - p_draw) * exp_h, "X": p_draw, "2": (1 - p_draw) * (1 - exp_h)}


def extract_markets(mat):
    h, d, a = np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()
    o15 = sum(mat[i, j] for i in range(mat.shape[0]) for j in range(mat.shape[1]) if i + j > 1.5)
    o25 = sum(mat[i, j] for i in range(mat.shape[0]) for j in range(mat.shape[1]) if i + j > 2.5)
    o35 = sum(mat[i, j] for i in range(mat.shape[0]) for j in range(mat.shape[1]) if i + j > 3.5)
    btts = mat[1:, 1:].sum()
    return {
        "1": h, "X": d, "2": a,
        "1X": h + d, "X2": a + d,
        "O1.5": o15, "U1.5": 1 - o15,
        "O2.5": o25, "U2.5": 1 - o25,
        "O3.5": o35, "U3.5": 1 - o35,
        "KG Var": btts, "KG Yok": 1 - btts,
    }


def analyze_multibook_match(match_obj, bankroll=1000.0, rho=-0.10, l3=0.12, phi=0.08):
    books = match_obj["books"]
    if not (books["1"] and books["X"] and books["2"]):
        return None

    def ref_price(mkt):
        if "Pinnacle" in books[mkt]:
            return books[mkt]["Pinnacle"]
        return float(np.median(list(books[mkt].values())))

    ref_1x2 = [ref_price("1"), ref_price("X"), ref_price("2")]
    fair_1x2, margin = devig_all(ref_1x2)

    p_o25 = None
    if books["O2.5"] and books["U2.5"]:
        p_o25 = devig_all([ref_price("O2.5"), ref_price("U2.5")])[0][0]

    lam_h, lam_a = invert_odds_to_xg(fair_1x2[0], fair_1x2[2], p_o25)

    mp = extract_markets(dixon_coles_matrix(lam_h, lam_a, rho))
    mw = extract_markets(bivariate_poisson_matrix(lam_h, lam_a, l3))
    mr = model_r_negbin(lam_h + lam_a, phi)
    mu = model_u_elo_from_fair(fair_1x2[0], fair_1x2[2])

    p1 = (mp["1"] * 1.0 + mw["1"] * 0.9 + fair_1x2[0] * 1.0 + mu["1"] * 0.9) / 3.8
    px = (mp["X"] * 1.0 + mw["X"] * 0.9 + fair_1x2[1] * 1.0 + mu["X"] * 0.9) / 3.8
    p2 = (mp["2"] * 1.0 + mw["2"] * 0.9 + fair_1x2[2] * 1.0 + mu["2"] * 0.9) / 3.8

    cons = {
        "1": p1,
        "X": px,
        "2": p2,
        "1X": p1 + px,
        "X2": p2 + px,
        "O1.5": (mp["O1.5"] * 1.0 + mw["O1.5"] * 0.9 + mr["O1.5"] * 1.0) / 2.9,
        "O2.5": (mp["O2.5"] * 1.0 + mw["O2.5"] * 0.9 + mr["O2.5"] * 1.0) / 2.9,
        "U2.5": (mp["U2.5"] * 1.0 + mw["U2.5"] * 0.9 + mr["U2.5"] * 1.0) / 2.9,
        "O3.5": (mp["O3.5"] * 1.0 + mw["O3.5"] * 0.9 + mr["O3.5"] * 1.0) / 2.9,
        "KG Var": (mp["KG Var"] * 1.0 + mw["KG Var"] * 0.9) / 1.9,
        "KG Yok": (mp["KG Yok"] * 1.0 + mw["KG Yok"] * 0.9) / 1.9,
    }

    market_rows = []
    value_rows = []

    for mkt, prob in cons.items():
        fair_odd = safe_fair_odd(prob)
        mkt_books = books.get(mkt, {})
        if mkt_books:
            best_book = max(mkt_books, key=mkt_books.get)
            best_price = mkt_books[best_book]
            med_price = float(np.median(list(mkt_books.values())))
            edge = prob * best_price - 1.0
            stale = "!" if best_price >= med_price * 1.08 else ""

            is_value = (edge >= 0.02 and best_price >= 1.40 and prob >= 0.20)
            stake = round(0.25 * (edge / (best_price - 1.0)) * bankroll, 1) if is_value else 0.0

            if is_value and stale:
                verdict = "⚠️ KONTROL (!)"
            elif edge >= 0.05 and is_value:
                verdict = "🟢 GÜÇLÜ DEĞER"
            elif is_value:
                verdict = "🟡 DEĞERLİ"
            else:
                verdict = "⛔ PAS"

            row = {
                "Karar": verdict,
                "Maç": f"{match_obj['home']} - {match_obj['away']}",
                "Saat (TR)": match_obj["time"],
                "Pazar": mkt,
                "Olasılık (%)": round(prob * 100, 1),
                "Adil Oran": fair_odd,
                "En İyi Oran": best_price,
                "Büro": best_book,
                "Edge (%)": round(edge * 100, 2),
                "1/4 Kelly": stake if stake > 0 else 0.0,
                "Medyan": round(med_price, 2),
                "Büro Sayısı": len(mkt_books),
            }
            market_rows.append(row)
            if is_value:
                value_rows.append(row)
        else:
            market_rows.append({
                "Karar": "ℹ️ ADİL ORAN",
                "Maç": f"{match_obj['home']} - {match_obj['away']}",
                "Saat (TR)": match_obj["time"],
                "Pazar": mkt,
                "Olasılık (%)": round(prob * 100, 1),
                "Adil Oran": fair_odd,
                "En İyi Oran": None,
                "Büro": "-",
                "Edge (%)": None,
                "1/4 Kelly": 0.0,
                "Medyan": None,
                "Büro Sayısı": 0,
            })

    return {
        "lam_h": lam_h,
        "lam_a": lam_a,
        "margin": margin,
        "markets": market_rows,
        "values": value_rows,
    }


def live_full_auto(lam_h, lam_a, minute, home_goals, away_goals, use_pace=True):
    m_clamped = max(0.0, min(90.0, float(minute)))
    elapsed_frac = m_clamped / 90.0
    rem = max(0.0, 90.0 - m_clamped)
    rem_frac = rem / 90.0
    hg = int(home_goals)
    ag = int(away_goals)
    cur_total = hg + ag

    if use_pace and elapsed_frac > 0:
        k_prior = 1.5
        lam_h_live = float(np.clip((k_prior * lam_h + hg) / (k_prior + elapsed_frac), 0.25, 4.0))
        lam_a_live = float(np.clip((k_prior * lam_a + ag) / (k_prior + elapsed_frac), 0.25, 4.0))
    else:
        lam_h_live, lam_a_live = lam_h, lam_a

    lh, la = lam_h_live * rem_frac, lam_a_live * rem_frac
    tot_rem = lh + la

    short_rows = []
    p10_val = 0.0
    for w in (5, 10, 15):
        w_eff = min(w, rem)
        p_w = 1.0 - math.exp(-(lam_h_live + lam_a_live) * (w_eff / 90.0)) if w_eff > 0 else 0.0
        if w == 10:
            p10_val = p_w
        short_rows.append({
            "Süre": f"Sonraki {w} dk Gol",
            "Olasılık (%)": round(p_w * 100, 1),
            "Adil Oran": safe_fair_odd(p_w),
        })

    p_no_more = math.exp(-tot_rem)
    p_at_least_1 = 1.0 - p_no_more
    p_at_least_2 = 1.0 - math.exp(-tot_rem) * (1.0 + tot_rem)
    p_home_scores = 1.0 - math.exp(-lh)
    p_away_scores = 1.0 - math.exp(-la)

    if tot_rem > 0:
        p_next_h = p_at_least_1 * (lh / tot_rem)
        p_next_a = p_at_least_1 * (la / tot_rem)
    else:
        p_next_h = p_next_a = 0.0

    next_rows = [
        {"İhtimal": "Sıradaki Gol: Ev Sahibi", "Olasılık (%)": round(p_next_h * 100, 1), "Adil Oran": safe_fair_odd(p_next_h)},
        {"İhtimal": "Sıradaki Gol: Deplasman", "Olasılık (%)": round(p_next_a * 100, 1), "Adil Oran": safe_fair_odd(p_next_a)},
        {"İhtimal": "Daha Fazla Gol Olmaz", "Olasılık (%)": round(p_no_more * 100, 1), "Adil Oran": safe_fair_odd(p_no_more)},
    ]

    line_1 = cur_total + 0.5
    line_2 = cur_total + 1.5
    market_dict = {
        f"Toplam {line_1} Üst (En az 1 gol daha)": p_at_least_1,
        f"Toplam {line_1} Alt (Daha gol olmaz)": p_no_more,
        f"Toplam {line_2} Üst (En az 2 gol daha)": p_at_least_2,
        "Ev Sahibi 1 Gol Daha Atar": p_home_scores,
        "Deplasman 1 Gol Daha Atar": p_away_scores,
        "Sıradaki Gol: Ev Sahibi": p_next_h,
        "Sıradaki Gol: Deplasman": p_next_a,
    }

    market_rows = [
        {
            "Canlı Pazar": name,
            "Olasılık (%)": round(prob * 100, 1),
            "Adil Oran (Büro bundan büyükse AL)": safe_fair_odd(prob),
        }
        for name, prob in market_dict.items()
    ]

    ph_pct = int(round(p_home_scores * 100))
    pa_pct = int(round(p_away_scores * 100))
    if p10_val >= 0.45:
        signal = "🟢 GOL YAKIN"
    elif p10_val >= 0.35:
        signal = "🟡 TEMPO YÜKSEK"
    elif p_home_scores >= 0.55 and p_away_scores >= 0.55:
        signal = f"🟡 İKİ TAKIM (E:%{ph_pct} D:%{pa_pct})"
    elif p_home_scores >= 0.55:
        signal = f"🟡 EV GOLÜ (%{ph_pct})"
    elif p_away_scores >= 0.55:
        signal = f"🟡 DEP GOLÜ (%{pa_pct})"
    else:
        signal = "⚪ NORMAL"

    return {
        "rem_min": int(rem),
        "lam_h_live": lam_h_live,
        "lam_a_live": lam_a_live,
        "p10": round(p10_val * 100, 1),
        "p_more1": round(p_at_least_1 * 100, 1),
        "fair_more1": safe_fair_odd(p_at_least_1),
        "p_home_g": round(p_home_scores * 100, 1),
        "p_away_g": round(p_away_scores * 100, 1),
        "signal": signal,
        "short_df": pd.DataFrame(short_rows),
        "next_df": pd.DataFrame(next_rows),
        "markets_df": pd.DataFrame(market_rows),
        "market_probs": market_dict,
    }


# ==============================================================================
# 2. VERİ ÇEKİCİLER (The Odds API v4 + SStats Canlı & Otomatik Kapanış Oranı)
# ==============================================================================
@st.cache_data(ttl=120)
def fetch_odds_api(api_key, sport_key):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": api_key,
        "regions": "eu,uk",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
    }
    resp = requests.get(url, params=params, timeout=15)
    remaining = resp.headers.get("x-requests-remaining", "?")
    if resp.status_code != 200:
        raise RuntimeError(f"API Hatası ({resp.status_code}): {resp.text[:160]}")

    events = resp.json()
    parsed = []
    for ev in events:
        home = ev.get("home_team", "Ev")
        away = ev.get("away_team", "Dep")
        raw_time = ev.get("commence_time", "")
        try:
            dt_tr = datetime.fromisoformat(raw_time.replace("Z", "+00:00")).astimezone(TR_TZ)
            commence_str = dt_tr.strftime("%d.%m %H:%M")
            match_date = dt_tr.date()
        except Exception:
            commence_str = raw_time[:16].replace("T", " ")
            match_date = None

        books = {"1": {}, "X": {}, "2": {}, "O2.5": {}, "U2.5": {}}

        for bm in ev.get("bookmakers", []):
            bm_name = bm.get("title", bm.get("key", "Büro"))
            for mk in bm.get("markets", []):
                mkey = mk.get("key")
                if mkey == "h2h":
                    for o in mk.get("outcomes", []):
                        nm, pr = o.get("name"), float(o.get("price", 0))
                        if pr <= 1.0:
                            continue
                        if nm == home:
                            books["1"][bm_name] = pr
                        elif nm == away:
                            books["2"][bm_name] = pr
                        elif nm == "Draw":
                            books["X"][bm_name] = pr
                elif mkey == "totals":
                    for o in mk.get("outcomes", []):
                        if o.get("point") == 2.5:
                            nm, pr = o.get("name"), float(o.get("price", 0))
                            if pr <= 1.0:
                                continue
                            if nm == "Over":
                                books["O2.5"][bm_name] = pr
                            elif nm == "Under":
                                books["U2.5"][bm_name] = pr

        parsed.append({
            "home": home, "away": away,
            "time": commence_str, "date": match_date,
            "books": books,
        })
    return parsed, remaining


def _parse_sstats_odds_blocks(blocks):
    """SStats /Odds/{gameId} veya /Games/{gameId} JSON yapısından 1X2 ve 2.5 Alt/Üst çıkarır."""
    vals = {"1": [], "X": [], "2": [], "O2.5": [], "U2.5": []}

    def scan_markets(mk_list):
        if not isinstance(mk_list, list):
            return
        for mk in mk_list:
            if not isinstance(mk, dict):
                continue
            if "odds" in mk and isinstance(mk["odds"], list) and mk["odds"] and isinstance(mk["odds"][0], dict):
                if "marketName" in mk["odds"][0] or "odds" in mk["odds"][0]:
                    scan_markets(mk["odds"])
                    continue
            if "markets" in mk and isinstance(mk["markets"], list):
                scan_markets(mk["markets"])
                continue

            m_name = str(mk.get("marketName") or mk.get("name") or "").lower()
            selections = mk.get("odds") or mk.get("values") or []
            if not isinstance(selections, list):
                continue
            for sel in selections:
                if not isinstance(sel, dict):
                    continue
                nm = str(sel.get("name") or sel.get("valueName") or "").lower().strip()
                raw_v = sel.get("value") or sel.get("odd") or sel.get("closing")
                try:
                    v = float(raw_v)
                except (TypeError, ValueError):
                    continue
                if v <= 1.01:
                    continue

                if "match winner" in m_name or "1x2" in m_name or "full time" in m_name:
                    if nm in ("home", "1"): vals["1"].append(v)
                    elif nm in ("draw", "x"): vals["X"].append(v)
                    elif nm in ("away", "2"): vals["2"].append(v)
                elif "goals over/under" in m_name or "over/under" in m_name or "total" in m_name:
                    if nm in ("over 2.5", "o 2.5", "over2.5"): vals["O2.5"].append(v)
                    elif nm in ("under 2.5", "u 2.5", "under2.5"): vals["U2.5"].append(v)

    if isinstance(blocks, dict):
        for k in ("data", "odds", "bookmakers", "markets"):
            if k in blocks:
                scan_markets(blocks[k] if isinstance(blocks[k], list) else [blocks[k]])
    elif isinstance(blocks, list):
        scan_markets(blocks)

    out = {}
    for k, lst in vals.items():
        if lst:
            out[k] = round(float(np.median(lst)), 2)
    return out


@st.cache_data(ttl=300)
def fetch_sstats_closing_odds(game_id, sstats_key=""):
    """Seçili canlı maçın maç önü kapanış oranlarını SStats /Odds/{gameId} üzerinden otomatik çeker."""
    if not game_id:
        return {}, "Maç ID bulunamadı"

    params = {"apikey": sstats_key} if sstats_key else {}
    endpoints = [
        f"https://api.sstats.net/Odds/{game_id}",
        f"https://api.sstats.net/Games/{game_id}",
    ]
    last_err = ""
    for url in endpoints:
        try:
            r = requests.get(url, params=params, timeout=8)
            if r.status_code == 200:
                parsed = _parse_sstats_odds_blocks(r.json())
                if "1" in parsed and "X" in parsed and "2" in parsed:
                    return parsed, "OK"
            elif r.status_code in (401, 403):
                last_err = "SStats bu maçın kapanış oranı için ücretsiz API anahtarı istiyor."
            else:
                last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
    return {}, last_err or "Bu maç için kapanış oranı kaydı bulunamadı."


@st.cache_data(ttl=45)
def fetch_sstats_live(sstats_key=""):
    url = "https://api.sstats.net/Games/list?Live=true"
    params = {"apikey": sstats_key} if sstats_key else {}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()
    games = raw.get("data", raw) if isinstance(raw, dict) else raw
    parsed = []
    for g in games:
        gid = g.get("id") or g.get("gameId") or g.get("fixtureId")
        home = (g.get("homeTeam") or {}).get("name") or g.get("homeName") or "Ev Sahibi"
        away = (g.get("awayTeam") or {}).get("name") or g.get("awayName") or "Deplasman"
        league = ((g.get("season") or {}).get("league") or {}).get("name") or g.get("leagueName") or "Lig"
        hg = int(g.get("homeResult") or 0)
        ag = int(g.get("awayResult") or 0)
        minute = int(g.get("elapsed") or g.get("minute") or 0)

        inline_odds = _parse_sstats_odds_blocks(g.get("odds") or [])
        o1 = inline_odds.get("1")
        ox = inline_odds.get("X")
        o2 = inline_odds.get("2")
        oo = inline_odds.get("O2.5")
        ou = inline_odds.get("U2.5")

        parsed.append({
            "id": gid,
            "Etiket": f"[{league}] {home} - {away} ({hg}-{ag} | {minute}')",
            "league": league, "home": home, "away": away,
            "hg": hg, "ag": ag, "score": f"{hg}-{ag}", "minute": minute,
            "o1": o1 or 2.15, "ox": ox or 3.30, "o2": o2 or 3.40,
            "oo": oo, "ou": ou,
            "has_real_odds": (o1 is not None and ox is not None and o2 is not None),
        })
    return parsed


# ==============================================================================
# 3. ARAYÜZ
# ==============================================================================
st.title("RUSPRG Auto v4.6 Pro")
st.caption("Tam Otomatik Maç Önü Value Tarayıcı • Canlı Tempo & Otomatik Kapanış Oranı Motoru")

today_tr = datetime.now(TR_TZ).date()

with st.sidebar:
    st.header("Ayarlar")
    api_key = st.text_input(
        "The Odds API Anahtarı",
        value=os.getenv("ODDS_API_KEY", ""),
        type="password",
        help="the-odds-api.com ücretsiz API anahtarı",
    )
    sstats_key = st.text_input(
        "SStats API Anahtarı (İsteğe Bağlı)",
        value=os.getenv("SSTATS_API_KEY", ""),
        type="password",
        help="Boş bırakılabilir; SStats kapanış oranları için gerekirse sstats.net'ten ücretsiz alınabilir.",
    )
    selected_league = st.selectbox("Lig Seçin", list(LEAGUES.keys()))
    date_filter = st.selectbox("Tarih Filtresi", ["Sadece Bugün", "Bugün + Yarın", "Tüm Yaklaşanlar"], index=0)
    min_edge = st.slider("Minimum Edge Filtresi (%)", 0.0, 10.0, 2.0, 0.5)

    with st.expander("Gelişmiş Sabit Katsayılar"):
        bankroll = st.number_input("Kasa (Bankroll)", value=1000.0, step=100.0)
        rho = st.number_input("Dixon-Coles ρ", value=-0.10, step=0.01)
        l3 = st.number_input("Biv. Poisson λ3", value=0.12, step=0.01)
        phi = st.number_input("Neg. Binom φ", value=0.08, step=0.01)

tab_scan, tab_detail, tab_live = st.tabs([
    "1. Gün Tarayıcı (Maç Önü)",
    "2. Maç Detayı & Büro Kıyas",
    "3. Canlı Tarayıcı & Adil Oran",
])

# --- TAB 1 & 2: MAÇ ÖNÜ (THE ODDS API) ---
matches_api = []
if api_key:
    try:
        all_fetched, rem_quota = fetch_odds_api(api_key, LEAGUES[selected_league])
        st.sidebar.success(f"Bağlandı • Kalan Hak: {rem_quota}")
        if date_filter == "Sadece Bugün":
            matches_api = [m for m in all_fetched if m["date"] == today_tr]
        elif date_filter == "Bugün + Yarın":
            tomorrow_tr = today_tr + timedelta(days=1)
            matches_api = [m for m in all_fetched if m["date"] in (today_tr, tomorrow_tr)]
        else:
            matches_api = all_fetched
    except Exception as e:
        st.sidebar.error(str(e))
else:
    st.sidebar.info("Maç önü tarama için The Odds API anahtarı girin.")

with tab_scan:
    if not api_key:
        st.warning("Gün Tarayıcı için sol menüye The Odds API anahtarınızı girin.")
    elif not matches_api:
        st.info(
            f"{selected_league} için '{date_filter}' ({today_tr.strftime('%d.%m.%Y')}) "
            "kapsamında açık maç önü oran bulunamadı. Sol menüden farklı lig veya 'Bugün + Yarın' seçebilirsiniz."
        )
    else:
        all_values = []
        for m in matches_api:
            res = analyze_multibook_match(m, bankroll, rho, l3, phi)
            if res:
                for v in res["values"]:
                    if v["Edge (%)"] >= min_edge:
                        all_values.append(v)

        if not all_values:
            st.info(f"{len(matches_api)} maç tarandı. Edge ≥ %{min_edge} şartını sağlayan değer bahsi çıkmadı.")
        else:
            df_val = pd.DataFrame(all_values).sort_values("Edge (%)", ascending=False)
            st.success(f"{len(matches_api)} maç tarandı • {len(df_val)} değer bahsi (Value) bulundu.")
            st.dataframe(df_val, use_container_width=True, hide_index=True)
            st.caption("🟢 GÜÇLÜ DEĞER: Edge ≥ %5 | 🟡 DEĞERLİ: Edge %2–%5 | ⚠️ KONTROL (!): Büro oranı medyanın %8+ üstünde.")

with tab_detail:
    if not matches_api:
        st.info("Seçili tarih ve ligde maç bulunamadı. Sol menüden ligi veya tarih filtresini değiştirin.")
    else:
        labels = [f"{m['time']} | {m['home']} - {m['away']}" for m in matches_api]
        idx = st.selectbox("Maç Seçin", range(len(labels)), format_func=lambda i: labels[i])
        m = matches_api[idx]
        res = analyze_multibook_match(m, bankroll, rho, l3, phi)

        if res:
            c1, c2, c3 = st.columns(3)
            c1.metric("Otomatik Ev xG (λ_h)", f"{res['lam_h']:.2f}")
            c2.metric("Otomatik Dep xG (λ_a)", f"{res['lam_a']:.2f}")
            c3.metric("Piyasa Marjı", f"%{res['margin']*100:.1f}")

            st.subheader("Tüm Pazarlar: Olasılık, Adil Oran ve Otomatik Karar")
            st.dataframe(pd.DataFrame(res["markets"]), use_container_width=True, hide_index=True)

            st.subheader("Tüm Büroların Oran Karşılaştırması")
            book_names = sorted(set(m["books"]["1"]) | set(m["books"]["X"]) | set(m["books"]["2"]))
            bm_table = [{
                "Büro": b,
                "1": m["books"]["1"].get(b, "-"),
                "X": m["books"]["X"].get(b, "-"),
                "2": m["books"]["2"].get(b, "-"),
                "2.5 Üst": m["books"]["O2.5"].get(b, "-"),
                "2.5 Alt": m["books"]["U2.5"].get(b, "-"),
            } for b in book_names]
            st.dataframe(pd.DataFrame(bm_table), use_container_width=True, hide_index=True)

# --- TAB 3: CANLI TARAYICI & OTOMATİK KAPANIŞ ORANI ---
with tab_live:
    col_f1, col_f2 = st.columns([3, 1])
    hide_bad_min = col_f1.checkbox("Hatalı/Bitmiş dk (0' ve 86'+) gizle", value=True)
    if col_f2.button("Canlıyı Yenile"):
        fetch_sstats_live.clear()
        fetch_sstats_closing_odds.clear()

    use_pace = st.checkbox("Mevcut skor hızını (Bayesian Tempo) hesaba kat", value=True)

    try:
        raw_live = fetch_sstats_live(sstats_key)
    except Exception as e:
        raw_live = []
        st.warning(f"SStats canlı verisi alınamadı: {e}")

    live_matches = [m for m in raw_live if (1 <= m["minute"] <= 85)] if hide_bad_min else raw_live

    if not live_matches:
        st.info("Şu an filtreye uyan (1'–85' arası) aktif canlı maç bulunamadı.")
    else:
        scan_rows = []
        live_cache = {}
        for lm in live_matches:
            fair_1x2, _ = devig_all([lm["o1"], lm["ox"], lm["o2"]])
            p_o25 = devig_all([lm["oo"], lm["ou"]])[0][0] if (lm["oo"] and lm["ou"]) else None
            lh, la = invert_odds_to_xg(fair_1x2[0], fair_1x2[2], p_o25)
            lres = live_full_auto(lh, la, lm["minute"], lm["hg"], lm["ag"], use_pace=use_pace)
            live_cache[lm["Etiket"]] = (lm, lh, la, lres)
            scan_rows.append({
                "Sinyal": lres["signal"],
                "Maç": f"{lm['home']} - {lm['away']}",
                "Lig": lm["league"],
                "Skor": lm["score"],
                "Dk": f"{lm['minute']}'",
                "10 dk Gol (%)": lres["p10"],
                "Ev +1 (%)": lres["p_home_g"],
                "Dep +1 (%)": lres["p_away_g"],
                "+1 Gol (%)": lres["p_more1"],
                "+1 Adil Oran": lres["fair_more1"],
            })

        st.subheader("Otomatik Canlı Fırsat Tarayıcısı")
        df_live_scan = pd.DataFrame(scan_rows).sort_values("10 dk Gol (%)", ascending=False)
        st.dataframe(df_live_scan, use_container_width=True, hide_index=True)

        # Seçili Canlı Maç İçin Kapanış Oranlarını OTOMATİK Çek
        st.divider()
        sel = st.selectbox("Detaylı Canlı Analiz İçin Maç Seçin (Kapanış Oranları Otomatik Çekilir)", [x["Etiket"] for x in live_matches])
        lm, lh, la, lres = live_cache[sel]

        auto_cl, cl_status = fetch_sstats_closing_odds(lm["id"], sstats_key)
        def_1 = float(auto_cl.get("1", lm["o1"]))
        def_x = float(auto_cl.get("X", lm["ox"]))
        def_2 = float(auto_cl.get("2", lm["o2"]))
        def_o25 = float(auto_cl.get("O2.5", lm["oo"] or 1.85))
        def_u25 = float(auto_cl.get("U2.5", lm["ou"] or 1.95))

        st.markdown("#### Maç Önü Kapanış Oranları (Otomatik Doldurulur • İsterseniz Düzenleyebilirsiniz)")
        if cl_status == "OK":
            st.success("✅ Kapanış oranları SStats üzerinden otomatik çekildi ve kutulara yazıldı.")
        else:
            st.info(f"ℹ️ Otomatik kapanış oranı çekilemedi ({cl_status}). Kutularda varsayılan değerler var; dilerseniz elle düzeltebilirsiniz.")

        r1c1, r1c2, r1c3 = st.columns(3)
        cl_1 = r1c1.number_input("Kapanış 1", min_value=1.02, max_value=50.0, value=def_1, step=0.05, key=f"c1_{lm['id']}_{def_1}")
        cl_x = r1c2.number_input("Kapanış X", min_value=1.02, max_value=50.0, value=def_x, step=0.05, key=f"cx_{lm['id']}_{def_x}")
        cl_2 = r1c3.number_input("Kapanış 2", min_value=1.02, max_value=50.0, value=def_2, step=0.05, key=f"c2_{lm['id']}_{def_2}")

        r2c1, r2c2 = st.columns(2)
        cl_o25 = r2c1.number_input("Kapanış 2.5 Üst", min_value=1.02, max_value=20.0, value=def_o25, step=0.05, key=f"co_{lm['id']}_{def_o25}")
        cl_u25 = r2c2.number_input("Kapanış 2.5 Alt", min_value=1.02, max_value=20.0, value=def_u25, step=0.05, key=f"cu_{lm['id']}_{def_u25}")

        # Kutudaki (otomatik veya düzenlenmiş) kapanış oranlarıyla anında hesapla
        fair_1x2_cl, _ = devig_all([cl_1, cl_x, cl_2])
        p_o25_cl = devig_all([cl_o25, cl_u25])[0][0]
        lh, la = invert_odds_to_xg(fair_1x2_cl[0], fair_1x2_cl[2], p_o25_cl)
        lres = live_full_auto(lh, la, lm["minute"], lm["hg"], lm["ag"], use_pace=use_pace)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Skor / Dakika", f"{lm['score']} ({lm['minute']}')")
        k2.metric("Sinyal", lres["signal"])
        k3.metric("Canlı Ev Tempo (xG)", f"{lres['lam_h_live']:.2f}", f"Ön: {lh:.2f}")
        k4.metric("Canlı Dep Tempo (xG)", f"{lres['lam_a_live']:.2f}", f"Ön: {la:.2f}")

        st.subheader("Tüm Canlı Pazarlar ve Otomatik Adil Oranlar")
        st.dataframe(lres["markets_df"], use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        col_a.markdown("**Kısa Vadede Gol Olasılığı ve Adil Oran**")
        col_a.dataframe(lres["short_df"], use_container_width=True, hide_index=True)
        col_b.markdown("**Sıradaki Gol (Toplam %100) ve Adil Oran**")
        col_b.dataframe(lres["next_df"], use_container_width=True, hide_index=True)

        # Otomatik Canlı Bahis Karar Hesaplayıcı (Adil Oran Kuralı)
        st.divider()
        st.subheader("Canlı Bahis Karar Hesaplayıcı (Büro Oranını Test Et)")
        bc1, bc2 = st.columns([2, 1])
        chosen_mkt = bc1.selectbox("Oynamak İstediğiniz Canlı Pazar", list(lres["market_probs"].keys()))
        book_odd = bc2.number_input("Sitenizdeki Canlı Oran", min_value=1.01, max_value=50.0, value=1.85, step=0.05)

        prob_m = lres["market_probs"][chosen_mkt]
        fair_m = safe_fair_odd(prob_m)
        edge_m = prob_m * book_odd - 1.0
        kelly_m = round(0.25 * (edge_m / (book_odd - 1.0)) * bankroll, 1) if (edge_m > 0 and book_odd > 1.0) else 0.0

        if edge_m >= 0.05 and book_odd >= 1.40:
            st.success(f"✅ GÜÇLÜ DEĞER (ALINIR) — Olasılık: %{prob_m*100:.1f} | Adil Oran: {fair_m} | Büro: {book_odd:.2f} | Edge: +%{edge_m*100:.1f} | 1/4 Kelly: {kelly_m} birim")
        elif edge_m >= 0.02 and book_odd >= 1.40:
            st.info(f"🟡 DEĞERLİ (ALINABİLİR) — Olasılık: %{prob_m*100:.1f} | Adil Oran: {fair_m} | Büro: {book_odd:.2f} | Edge: +%{edge_m*100:.1f} | 1/4 Kelly: {kelly_m} birim")
        else:
            st.error(f"❌ PAS GEÇ (ORAN DÜŞÜK) — Olasılık: %{prob_m*100:.1f} | Adil Oran: {fair_m} | Büro: {book_odd:.2f} | Edge: %{edge_m*100:.1f}")
