import math
import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import brentq
from scipy.stats import poisson, nbinom

st.set_page_config(page_title="RUSPRG Auto v4.2 Pro", layout="wide")

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
# 1. İSTATİSTİKSEL MOTOR (P, Q, R, U, W, X, V ve CANLI v4.2)
# ==============================================================================
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
    o25 = 1.0 - sum(probs[:3])
    return {"O2.5": o25, "U2.5": 1.0 - o25}


def model_u_elo_from_fair(p_h, p_a):
    exp_h = np.clip(p_h / max(p_h + p_a, 1e-6), 0.05, 0.95)
    diff = -400.0 * math.log10((1.0 / exp_h) - 1.0)
    p_draw = 0.28 * math.exp(-((diff / 300.0) ** 2))
    return {"1": (1 - p_draw) * exp_h, "X": p_draw, "2": (1 - p_draw) * (1 - exp_h)}


def extract_markets(mat):
    h, d, a = np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()
    o25 = sum(mat[i, j] for i in range(mat.shape[0]) for j in range(mat.shape[1]) if i + j > 2.5)
    btts = mat[1:, 1:].sum()
    return {"1": h, "X": d, "2": a, "O2.5": o25, "U2.5": 1 - o25, "KG Var": btts, "KG Yok": 1 - btts}


def live_v42(lam_h, lam_a, minute):
    rem = max(0.0, 90.0 - float(minute))
    frac = rem / 90.0
    lh, la = lam_h * frac, lam_a * frac
    tot = lh + la
    goal_in = {f"{w} dk": round((1 - math.exp(-(lam_h + lam_a) * (min(w, rem) / 90.0))) * 100, 1) for w in (5, 10, 15)}
    p_no = math.exp(-tot)
    p_nh = (1 - p_no) * (lh / tot) if tot > 0 else 0.0
    p_na = (1 - p_no) * (la / tot) if tot > 0 else 0.0
    return goal_in, {"Ev": round(p_nh * 100, 1), "Dep": round(p_na * 100, 1), "Gol Olmaz": round(p_no * 100, 1)}


def analyze_multibook_match(match_obj, bankroll=1000.0, rho=-0.10, l3=0.12, phi=0.08):
    """
    Çoklu büro verisinden:
    1) Referans (Pinnacle varsa Pinnacle, yoksa medyan) oranlarla Model Q + X hesaplar.
    2) Tüm bürolar arasında en iyi oranı bulup Model V (Edge, 1/4 Kelly, '!' bayat hat) üretir.
    """
    books = match_obj["books"]
    if not (books["1"] and books["X"] and books["2"]):
        return None

    # Referans oran: Pinnacle varsa onu, yoksa tüm büroların medyanını kullan
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

    cons = {
        "1": (mp["1"] * 1.0 + mw["1"] * 0.9 + fair_1x2[0] * 1.0 + mu["1"] * 0.9) / 3.8,
        "X": (mp["X"] * 1.0 + mw["X"] * 0.9 + fair_1x2[1] * 1.0 + mu["X"] * 0.9) / 3.8,
        "2": (mp["2"] * 1.0 + mw["2"] * 0.9 + fair_1x2[2] * 1.0 + mu["2"] * 0.9) / 3.8,
        "O2.5": (mp["O2.5"] * 1.0 + mw["O2.5"] * 0.9 + mr["O2.5"] * 1.0) / 2.9,
        "U2.5": (mp["U2.5"] * 1.0 + mw["U2.5"] * 0.9 + mr["U2.5"] * 1.0) / 2.9,
        "KG Var": (mp["KG Var"] * 1.0 + mw["KG Var"] * 0.9) / 1.9,
    }

    market_rows = []
    value_rows = []

    for mkt, prob in cons.items():
        fair_odd = round(1.0 / prob, 2) if prob > 0 else 0.0
        mkt_books = books.get(mkt, {})
        if mkt_books:
            best_book = max(mkt_books, key=mkt_books.get)
            best_price = mkt_books[best_book]
            med_price = float(np.median(list(mkt_books.values())))
            edge = prob * best_price - 1.0
            stale = "!" if best_price >= med_price * 1.08 else ""

            is_value = (edge >= 0.02 and best_price >= 1.40 and prob >= 0.20)
            stake = round(0.25 * (edge / (best_price - 1.0)) * bankroll, 1) if is_value else 0.0
            signal = "YEŞİL" if edge >= 0.05 else ("SARI" if is_value else "-")

            row = {
                "Maç": f"{match_obj['home']} - {match_obj['away']}",
                "Saat": match_obj["time"],
                "Pazar": mkt,
                "Olasılık (%)": round(prob * 100, 1),
                "Adil Oran": fair_odd,
                "En İyi Oran": best_price,
                "Büro": best_book,
                "Medyan": round(med_price, 2),
                "Uyarı": stale,
                "Edge (%)": round(edge * 100, 2),
                "1/4 Kelly": stake if stake > 0 else 0.0,
                "Sinyal": signal,
                "Büro Sayısı": len(mkt_books),
            }
            market_rows.append(row)
            if is_value:
                value_rows.append(row)
        else:
            market_rows.append({
                "Maç": f"{match_obj['home']} - {match_obj['away']}",
                "Saat": match_obj["time"],
                "Pazar": mkt,
                "Olasılık (%)": round(prob * 100, 1),
                "Adil Oran": fair_odd,
                "En İyi Oran": None,
                "Büro": "-",
                "Medyan": None,
                "Uyarı": "",
                "Edge (%)": None,
                "1/4 Kelly": 0.0,
                "Sinyal": "-",
                "Büro Sayısı": 0,
            })

    return {
        "lam_h": lam_h,
        "lam_a": lam_a,
        "margin": margin,
        "markets": market_rows,
        "values": value_rows,
    }


# ==============================================================================
# 2. VERİ ÇEKİCİLER (The Odds API v4 + SStats Canlı)
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
        commence = ev.get("commence_time", "")[:16].replace("T", " ")
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

        parsed.append({"home": home, "away": away, "time": commence, "books": books})
    return parsed, remaining


@st.cache_data(ttl=60)
def fetch_sstats_live():
    url = "https://api.sstats.net/Games/list?Live=true"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    raw = resp.json()
    games = raw.get("data", raw) if isinstance(raw, dict) else raw
    parsed = []
    for g in games:
        home = (g.get("homeTeam") or {}).get("name") or g.get("homeName") or "Ev Sahibi"
        away = (g.get("awayTeam") or {}).get("name") or g.get("awayName") or "Deplasman"
        league = ((g.get("season") or {}).get("league") or {}).get("name") or g.get("leagueName") or "Lig"
        score = f"{g.get('homeResult', 0)}-{g.get('awayResult', 0)}"
        minute = g.get("elapsed") or g.get("minute") or 0

        o1 = ox = o2 = oo = ou = None
        for mk in (g.get("odds") or []):
            m_name = str(mk.get("marketName", "")).lower()
            for sel in mk.get("odds", []):
                nm = str(sel.get("name", "")).lower()
                val = sel.get("value")
                if "match winner" in m_name or "1x2" in m_name:
                    if nm in ("home", "1"): o1 = float(val)
                    elif nm in ("draw", "x"): ox = float(val)
                    elif nm in ("away", "2"): o2 = float(val)
                elif "goals over/under" in m_name:
                    if nm == "over 2.5": oo = float(val)
                    elif nm == "under 2.5": ou = float(val)

        parsed.append({
            "Etiket": f"[{league}] {home} - {away} ({score} | {minute}')",
            "home": home, "away": away, "score": score, "minute": minute,
            "o1": o1 or 2.15, "ox": ox or 3.30, "o2": o2 or 3.40, "oo": oo, "ou": ou,
        })
    return parsed


# ==============================================================================
# 3. ARAYÜZ
# ==============================================================================
st.title("RUSPRG Auto v4.2 Pro")
st.caption("Çoklu Büro Value Tarayıcı (The Odds API) • Canlı Analiz (SStats) • P / Q / R / U / W / X / V")

with st.sidebar:
    st.header("Ayarlar")
    api_key = st.text_input(
        "The Odds API Anahtarı",
        value=os.getenv("ODDS_API_KEY", ""),
        type="password",
        help="the-odds-api.com üzerinden ücretsiz (aylık 500 istek) anahtar alabilirsiniz.",
    )
    selected_league = st.selectbox("Lig Seçin", list(LEAGUES.keys()))
    min_edge = st.slider("Minimum Edge Filtresi (%)", 0.0, 10.0, 2.0, 0.5)

    with st.expander("Gelişmiş Sabit Katsayılar"):
        bankroll = st.number_input("Kasa (Bankroll)", value=1000.0, step=100.0)
        rho = st.number_input("Dixon-Coles ρ", value=-0.10, step=0.01)
        l3 = st.number_input("Biv. Poisson λ3", value=0.12, step=0.01)
        phi = st.number_input("Neg. Binom φ", value=0.08, step=0.01)

tab_scan, tab_detail, tab_live = st.tabs([
    "Gün Tarayıcı (Çoklu Büro Value)",
    "Maç Detayı & Büro Kıyaslama",
    "Canlı Maçlar (SStats v4.2)",
])

# --- TAB 1 & 2: THE ODDS API ---
matches_api = []
if api_key:
    try:
        matches_api, rem_quota = fetch_odds_api(api_key, LEAGUES[selected_league])
        st.sidebar.success(f"Bağlandı • Kalan Aylık Hak: {rem_quota}")
    except Exception as e:
        st.sidebar.error(str(e))
else:
    st.sidebar.info("Çoklu büro taraması için The Odds API anahtarı girin.")

with tab_scan:
    if not api_key:
        st.warning("Gün Tarayıcı için sol menüye ücretsiz The Odds API anahtarınızı girin.")
    elif not matches_api:
        st.info("Bu ligde şu an açık maç bulunamadı.")
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
            st.caption("Uyarı sütunundaki '!' işareti, en iyi oranın piyasa medyanından en az %8 yukarıda olduğunu (bayat oran veya limit riski) gösterir.")

with tab_detail:
    if not matches_api:
        st.info("Önce sol menüden API anahtarı girip bir lig yükleyin.")
    else:
        labels = [f"{m['time']} | {m['home']} - {m['away']}" for m in matches_api]
        idx = st.selectbox("Maç Seçin", range(len(labels)), format_func=lambda i: labels[i])
        m = matches_api[idx]
        res = analyze_multibook_match(m, bankroll, rho, l3, phi)

        if res:
            c1, c2, c3 = st.columns(3)
            c1.metric("Otomatik Ev xG (λ_h)", f"{res['lam_h']:.2f}")
            c2.metric("Otomatik Dep xG (λ_a)", f"{res['lam_a']:.2f}")
            c3.metric("Referans Piyasa Marjı", f"%{res['margin']*100:.1f}")

            st.subheader("Model X Konsensüs & En İyi Büro (Model V)")
            st.dataframe(pd.DataFrame(res["markets"]), use_container_width=True, hide_index=True)

            st.subheader("Tüm Büroların 1X2 Oranları")
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

# --- TAB 3: CANLI (SSTATS) ---
with tab_live:
    try:
        live_matches = fetch_sstats_live()
    except Exception as e:
        live_matches = []
        st.warning(f"SStats canlı verisi alınamadı: {e}")

    if not live_matches:
        st.info("Şu an canlı maç listesi boş.")
    else:
        sel = st.selectbox("Canlı Maç Seçin", [x["Etiket"] for x in live_matches])
        lm = next(x for x in live_matches if x["Etiket"] == sel)

        fair_1x2, _ = devig_all([lm["o1"], lm["ox"], lm["o2"]])
        p_o25 = devig_all([lm["oo"], lm["ou"]])[0][0] if (lm["oo"] and lm["ou"]) else None
        lh, la = invert_odds_to_xg(fair_1x2[0], fair_1x2[2], p_o25)

        k1, k2, k3 = st.columns(3)
        k1.metric("Skor / Dakika", f"{lm['score']} ({lm['minute']}')")
        k2.metric("Maç Öncesi Ev xG", f"{lh:.2f}")
        k3.metric("Maç Öncesi Dep xG", f"{la:.2f}")

        g_in, nxt = live_v42(lh, la, lm["minute"])
        col_a, col_b = st.columns(2)
        col_a.markdown("**Kısa Vadede Gol Olasılığı (%)**")
        col_a.table(pd.DataFrame(g_in.items(), columns=["Süre", "Olasılık (%)"]))
        col_b.markdown("**Sıradaki Gol Dağılımı (% — Toplam 100)**")
        col_b.table(pd.DataFrame(nxt.items(), columns=["İhtimal", "Olasılık (%)"]))
