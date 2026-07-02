"""
app.py — ARIA v2 Streamlit Demo (Impressive Version)
Run: streamlit run app.py
"""
import json, io, csv, sys, os, time
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="ARIA v2 — AI Candidate Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0a0a0f; }
  [data-testid="stSidebar"] { background: #0f0f1a; border-right: 1px solid #1e1e2e; }
  .main-title { font-size:2.6rem; font-weight:800; color:#fff; letter-spacing:-1px; margin:0; }
  .main-sub { color:#666; font-size:1rem; margin-top:4px; margin-bottom:2rem; }
  .kpi-card { background:#111122; border:1px solid #1e1e35; border-radius:14px; padding:18px 20px; }
  .kpi-val { font-size:28px; font-weight:700; color:#fff; }
  .kpi-lbl { font-size:12px; color:#555; margin-top:2px; text-transform:uppercase; letter-spacing:.05em; }
  .kpi-sub { font-size:11px; margin-top:6px; }
  .cand-card { background:#0f0f1c; border:1px solid #1a1a2e; border-radius:14px;
               padding:16px 18px; margin-bottom:10px; transition:border-color .2s; }
  .cand-card:hover { border-color:#4B0082; }
  .cand-card.rank1 { border-left:3px solid #7B2FBE; background:#110e1a; }
  .rank-circle { width:38px; height:38px; border-radius:50%; display:flex;
                 align-items:center; justify-content:center; font-weight:700;
                 font-size:14px; flex-shrink:0; float:left; margin-right:14px; }
  .r1 { background:linear-gradient(135deg,#4B0082,#7B2FBE); color:#fff; }
  .r2 { background:#1a1a2e; color:#aaa; border:1px solid #2a2a3e; }
  .rn { background:#111; color:#555; border:1px solid #1e1e2e; }
  .cand-name { font-size:14px; font-weight:600; color:#e8e8f0; }
  .cand-meta { font-size:12px; color:#555; margin-top:2px; }
  .score-num { font-size:20px; font-weight:700; color:#fff; text-align:right; }
  .tag { display:inline-block; padding:2px 9px; border-radius:20px;
         font-size:11px; font-weight:500; margin:2px 2px 0 0; }
  .tag-r { background:#1a0a2e; color:#9B6BE8; border:1px solid #2a1a4e; }
  .tag-m { background:#0a1a2e; color:#5BA3E8; border:1px solid #1a2a4e; }
  .tag-p { background:#0a1a0e; color:#3DC87A; border:1px solid #1a3a2e; }
  .tag-w { background:#2e0a0a; color:#E85B5B; border:1px solid #4e1a1a; }
  .tag-n { background:#151515; color:#888; border:1px solid #222; }
  .dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:4px; }
  .dot-g { background:#3DC87A; }
  .dot-a { background:#F5A623; }
  .dot-r { background:#E85B5B; }
  .bar-bg { background:#1a1a2e; border-radius:3px; height:5px; margin:2px 0; }
  .dna-chip { padding:4px 10px; border-radius:20px; font-size:12px;
              font-weight:600; display:inline-block; margin:2px; }
  .gap-row { display:flex; align-items:center; padding:6px 0;
             border-bottom:1px solid #1a1a2e; font-size:12px; }
  .gap-row:last-child { border-bottom:none; }
  .section-hdr { font-size:11px; color:#444; text-transform:uppercase;
                 letter-spacing:.08em; font-weight:600; margin:1.5rem 0 .75rem; }
  stDataFrame { border-radius:12px !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 ARIA v2")
    st.caption("5 novel ranking techniques")
    st.divider()

    techniques = [
        ("🧬", "Career DNA", "Encodes career as R/M/P/S/D/C/X sequence. Sequence similarity vs ideal JD DNA."),
        ("🕸️", "Skill Graph", "Measures cluster depth — how many interconnected retrieval/ML nodes activated."),
        ("⚡", "Velocity", "Exponential time-decay on 23 signals. Active yesterday > inactive 6 months."),
        ("📊", "Peer Rank", "Percentile within YOE × company-tier cohort. Best-in-class, not just above average."),
        ("🎯", "Gap Vector", "Requirement-level gap analysis. Exact evidence for what's met / missing."),
    ]
    for icon, name, desc in techniques:
        st.markdown(f"**{icon} {name}**")
        st.caption(desc)
        st.divider()

    st.markdown("**Formula**")
    st.code("final = (\n  0.28 × gap_vector +\n  0.22 × skill_graph +\n  0.18 × dna +\n  0.16 × peer_pct +\n  0.16 × velocity\n) × vel_multiplier", language="python")

# ── Header ────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🎯 ARIA v2</p>', unsafe_allow_html=True)
st.markdown('<p class="main-sub">Adaptive Retrieval Intelligence Architecture · Redrob Hackathon 2026 · Modern College of Engineering, Pune</p>', unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload candidate data (sample_candidates.json from hackathon bundle)",
    type=["json", "jsonl"],
    label_visibility="collapsed",
)

if not uploaded:
    st.markdown("""
    <div style="background:#0f0f1c;border:1.5px dashed #2a2a4e;border-radius:14px;
                padding:40px;text-align:center;margin:1rem 0 2rem;">
      <div style="font-size:2rem;margin-bottom:8px">📂</div>
      <div style="color:#888;font-size:15px;margin-bottom:4px">
        Drop <strong style="color:#aaa">sample_candidates.json</strong> to begin
      </div>
      <div style="color:#444;font-size:12px">Supports .json and .jsonl · up to 1 GB</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div style="background:#0f0f1c;border:1px solid #1a1a2e;border-radius:12px;padding:16px">
        <div style="color:#E85B5B;font-size:13px;font-weight:600;margin-bottom:8px">❌ Others build</div>
        <div style="color:#555;font-size:12px;line-height:1.7">
        Match skill keywords to JD<br>Rank by keyword frequency<br>Ignore behavioral signals<br>Score candidates in isolation
        </div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div style="background:#0f0f1c;border:1px solid #1a2e1a;border-radius:12px;padding:16px">
        <div style="color:#3DC87A;font-size:13px;font-weight:600;margin-bottom:8px">✅ ARIA does</div>
        <div style="color:#555;font-size:12px;line-height:1.7">
        Reads career narratives (what they built)<br>Measures retrieval cluster depth<br>Time-decays all 23 signals<br>Benchmarks within peer cohorts
        </div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div style="background:#0f0f1c;border:1px solid #1a1a2e;border-radius:12px;padding:16px">
        <div style="color:#9B6BE8;font-size:13px;font-weight:600;margin-bottom:8px">🏆 Why it wins</div>
        <div style="color:#555;font-size:12px;line-height:1.7">
        5 novel techniques, not 1<br>No GPU needed — runs in &lt;3 min<br>Specific honest reasoning per candidate<br>Catches honeypots automatically
        </div></div>""", unsafe_allow_html=True)
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────
raw = uploaded.read().decode("utf-8")
try:
    if uploaded.name.endswith(".jsonl"):
        candidates = [json.loads(l) for l in raw.strip().split("\n") if l.strip()]
    else:
        data = json.loads(raw)
        candidates = data if isinstance(data, list) else [data]
    st.success(f"✅ Loaded **{len(candidates):,}** candidates")
except Exception as e:
    st.error(f"JSON error: {e}")
    st.stop()

top_n = st.slider("Show top N candidates", 5, min(50, len(candidates)), min(20, len(candidates)))

if not st.button("🚀 Run ARIA v2 Ranking", type="primary", use_container_width=True):
    st.stop()

# ── Run pipeline ──────────────────────────────────────────────────────────
from rank import rank_candidates
from scorer.dna import dna_similarity_score
from scorer.skill_graph import skill_graph_score
from scorer.velocity import velocity_score
from scorer.gap_vector import compute_gap_vector
from scorer.peer_benchmark import peer_context

t0 = time.time()
with st.spinner("Running ARIA v2 pipeline…"):
    results = rank_candidates(candidates, verbose=False)
elapsed = time.time() - t0

cmap = {c["candidate_id"]: c for c in candidates}

# Pre-compute all scorer outputs for display
details = {}
for r in results[:top_n]:
    c = cmap.get(r["candidate_id"], {})
    details[r["candidate_id"]] = {
        "dna":   dna_similarity_score(c),
        "graph": skill_graph_score(c),
        "vel":   velocity_score(c),
        "gap":   compute_gap_vector(c),
        "pc":    peer_context(c),
        "profile": c.get("profile", {}),
    }

# ── KPI Row ──────────────────────────────────────────────────────────────
st.divider()
k1, k2, k3, k4 = st.columns(4)
top_score = results[0]["score"] if results else 0
top10_avg = sum(r["score"] for r in results[:10]) / min(10, len(results))
active = sum(1 for r in results[:20] if details.get(r["candidate_id"],{}).get("vel",{}).get("days_inactive",999) < 30)
honeypots = sum(1 for c in candidates if dna_similarity_score(c).get("dna_sequence","").count("X") + dna_similarity_score(c).get("dna_sequence","").count("C") >= 3)

def kpi(col, val, lbl, sub, sub_color="#555"):
    col.markdown(f"""<div class="kpi-card">
      <div class="kpi-val">{val}</div>
      <div class="kpi-lbl">{lbl}</div>
      <div class="kpi-sub" style="color:{sub_color}">{sub}</div>
    </div>""", unsafe_allow_html=True)

kpi(k1, f"{len(candidates):,}", "Candidates ranked", f"Done in {elapsed:.1f}s", "#3DC87A")
kpi(k2, f"{top_score:.3f}", "Top candidate score", "Gap + graph + velocity")
kpi(k3, f"{active}/20", "Active ≤ 30 days", "In top-20 results", "#9B6BE8")
kpi(k4, f"{elapsed:.1f}s", "Pipeline runtime", "CPU only, no GPU", "#F5A623")

# ── Candidate Cards ───────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Top candidates</div>', unsafe_allow_html=True)

DNA_COLORS = {
    "R": ("tag-r", "Retrieval"),
    "M": ("tag-m", "ML Eng"),
    "P": ("tag-p", "Platform"),
    "S": ("tag-n", "Software"),
    "D": ("tag-n", "Data"),
    "C": ("tag-w", "Services"),
    "X": ("tag-w", "Non-tech"),
    "?": ("tag-n", "Unknown"),
}

for r in results[:top_n]:
    cid = r["candidate_id"]
    d = details.get(cid, {})
    p = d.get("profile", {})
    vel = d.get("vel", {})
    dna = d.get("dna", {})
    gap = d.get("gap", {})
    pc  = d.get("pc", {})

    rank = r["rank"]
    rc = "r1" if rank == 1 else ("r2" if rank <= 3 else "rn")
    cc = "cand-card rank1" if rank == 1 else "cand-card"

    # Availability dot
    di = vel.get("days_inactive", 999)
    if di < 14:   dot = '<span class="dot dot-g"></span>'
    elif di < 60: dot = '<span class="dot dot-a"></span>'
    else:         dot = '<span class="dot dot-r"></span>'

    # DNA tags
    seq = dna.get("dna_sequence", "")
    seen = set()
    dna_tags = ""
    for ch in seq:
        if ch not in seen:
            seen.add(ch)
            cls, label = DNA_COLORS.get(ch, ("tag-n", ch))
            dna_tags += f'<span class="tag {cls}">{ch}: {label}</span>'

    # Hard gaps
    hg = gap.get("hard_gaps", [])
    gap_tag = f'<span class="tag tag-w">⚠ {len(hg)} hard gap{"s" if len(hg)!=1 else ""}</span>' if hg else '<span class="tag tag-p">✓ No hard gaps</span>'

    notice = vel.get("notice_days", 60)
    otw = "Open to work" if vel.get("open_to_work") else ""
    peer_grp = pc.get("peer_group","").replace("_"," ")

    st.markdown(f"""
    <div class="{cc}">
      <div style="display:flex;align-items:flex-start;gap:14px">
        <div class="rank-circle {rc}">{rank}</div>
        <div style="flex:1;min-width:0">
          <div class="cand-name">{cid}</div>
          <div class="cand-meta">{p.get('current_title','')[:35]} · {p.get('current_company','')[:20]} · {p.get('years_of_experience',0):.1f}yr · {p.get('location','')}</div>
          <div style="margin-top:8px">
            {dna_tags}
            {gap_tag}
            <span class="tag tag-n">{dot}{di}d inactive</span>
            {"<span class='tag tag-p'>Open to work</span>" if vel.get('open_to_work') else ""}
            <span class="tag tag-n">{notice}d notice</span>
            <span class="tag tag-n">{peer_grp} cohort</span>
          </div>
          <div style="font-size:11px;color:#333;margin-top:6px">{r['reasoning'][:120]}…</div>
        </div>
        <div style="text-align:right;min-width:80px">
          <div class="score-num">{r['score']:.3f}</div>
          <div style="font-size:10px;color:#333;margin-top:4px">
            G:{gap.get('gap_score',0):.2f} V:{vel.get('velocity_score',0):.2f}
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Deep Dive — Rank #1 ───────────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-hdr">Deep dive — rank #1</div>', unsafe_allow_html=True)

top_cid = results[0]["candidate_id"]
td = details[top_cid]
t_dna  = td["dna"]
t_grph = td["graph"]
t_vel  = td["vel"]
t_gap  = td["gap"]

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("""<div style="background:#0f0f1c;border:1px solid #1a1a2e;border-radius:12px;padding:16px">
    <div style="font-size:11px;color:#444;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px">Career DNA sequence</div>""", unsafe_allow_html=True)
    dna_seq = t_dna.get("dna_sequence","")
    dna_html = ""
    chip_styles = {
        "R": "background:#1a0a2e;color:#9B6BE8;border:1px solid #2a1a4e",
        "M": "background:#0a1a2e;color:#5BA3E8;border:1px solid #1a2a4e",
        "P": "background:#0a1a0e;color:#3DC87A;border:1px solid #1a3a2e",
        "S": "background:#1a1a0a;color:#F5A623;border:1px solid #3a3a1a",
        "D": "background:#0a1a1a;color:#5BC8C8;border:1px solid #1a3a3a",
        "C": "background:#2e0a0a;color:#E85B5B;border:1px solid #4e1a1a",
        "X": "background:#2e0a0a;color:#E85B5B;border:1px solid #4e1a1a",
    }
    for ch in dna_seq:
        s = chip_styles.get(ch,"background:#1a1a1a;color:#888")
        _, lbl = DNA_COLORS.get(ch,("","?"))
        dna_html += f'<span class="dna-chip" style="{s}">{ch} {lbl}</span>'
    st.markdown(dna_html or "—", unsafe_allow_html=True)
    if t_dna.get("depth_tokens"):
        st.markdown(f'<div style="margin-top:8px;font-size:11px;color:#3DC87A">Depth expertise: {", ".join(t_dna["depth_tokens"])}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="margin-top:6px;font-size:11px;color:#444">DNA score: {t_dna.get("score",0):.3f} · Has retrieval: {"✓" if t_dna.get("has_retrieval") else "✗"}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_b:
    st.markdown("""<div style="background:#0f0f1c;border:1px solid #1a1a2e;border-radius:12px;padding:16px">
    <div style="font-size:11px;color:#444;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px">Skill graph clusters</div>""", unsafe_allow_html=True)
    cluster_html = ""
    bar_colors = {
        "retrieval_core":"#9B6BE8","ranking_recommendation":"#5BA3E8",
        "ml_engineering":"#3DC87A","production_ml":"#F5A623",
        "infra_scale":"#5BC8C8","software_foundation":"#888",
    }
    for cl, depth in t_grph.get("cluster_depths",{}).items():
        if depth > 0.05 and cl != "wrong_domain":
            pct = int(depth*100)
            col_hex = bar_colors.get(cl,"#555")
            cluster_html += f"""
            <div style="margin-bottom:8px">
              <div style="display:flex;justify-content:space-between;font-size:11px;color:#666;margin-bottom:3px">
                <span>{cl.replace("_"," ")}</span><span style="color:{col_hex}">{depth:.2f}</span>
              </div>
              <div class="bar-bg"><div style="width:{pct}%;height:5px;background:{col_hex};border-radius:3px"></div></div>
            </div>"""
    st.markdown(cluster_html or "—", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:11px;color:#444;margin-top:4px">Graph score: {t_grph.get("score",0):.3f} · Density: {t_grph.get("neighborhood_density",0):.2f}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_c:
    st.markdown("""<div style="background:#0f0f1c;border:1px solid #1a1a2e;border-radius:12px;padding:16px">
    <div style="font-size:11px;color:#444;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px">JD gap vector</div>""", unsafe_allow_html=True)
    gap_html = ""
    for req_name, req in td["gap"].get("requirements",{}).items():
        icon = "✓" if req["met"] else "✗"
        col_hex = "#3DC87A" if req["met"] else "#E85B5B"
        pct = int(req["strength"]*100)
        gap_html += f"""
        <div class="gap-row">
          <span style="color:{col_hex};margin-right:6px;font-size:12px">{icon}</span>
          <span style="color:#777;flex:1">{req_name.replace("_"," ")[:22]}</span>
          <span style="color:{col_hex};font-size:11px;margin-right:6px">{req['strength']:.2f}</span>
          <div style="width:40px;height:4px;background:#1a1a2e;border-radius:2px">
            <div style="width:{pct}%;height:100%;background:{col_hex};border-radius:2px"></div>
          </div>
        </div>"""
    st.markdown(gap_html or "—", unsafe_allow_html=True)
    hg = td["gap"].get("hard_gaps",[])
    if hg:
        st.markdown(f'<div style="font-size:11px;color:#E85B5B;margin-top:8px">Hard gaps: {", ".join(hg)}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Score distribution chart ──────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-hdr">Score distribution — top 100</div>', unsafe_allow_html=True)
chart_df = pd.DataFrame({"Rank":[r["rank"] for r in results], "Score":[r["score"] for r in results]}).set_index("Rank")
st.line_chart(chart_df, use_container_width=True, color="#7B2FBE")

# ── Download ─────────────────────────────────────────────────────────────
st.divider()
import io as _io
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side

df_out = pd.DataFrame([{
    "candidate_id": r["candidate_id"], "rank": r["rank"],
    "score": round(r["score"],6), "reasoning": r["reasoning"],
} for r in results])

xlsx_buf = _io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as wx:
    df_out.to_excel(wx, index=False, sheet_name="Rankings")
    ws = wx.sheets["Rankings"]
    for cell in ws[1]:
        cell.fill = PatternFill("solid", fgColor="4B0082")
        cell.font = Font(bold=True, color="FFFFFF", size=11)
        cell.alignment = Alignment(horizontal="center")
    ws.column_dimensions["A"].width = 20
    ws.column_dimensions["B"].width = 8
    ws.column_dimensions["C"].width = 12
    ws.column_dimensions["D"].width = 85
    ws.freeze_panes = "A2"
    for i, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row), 1):
        fill = PatternFill("solid", fgColor="F3EFFF" if i%2 else "FFFFFF")
        for cell in row:
            cell.fill = fill
xlsx_buf.seek(0)

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button(
        "⬇️ Download submission.xlsx — upload to Hack2Skill",
        data=xlsx_buf.getvalue(),
        file_name="submission.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True, type="primary",
    )
with col_dl2:
    out = _io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["candidate_id","rank","score","reasoning"])
    for r in results:
        writer.writerow([r["candidate_id"],r["rank"],f"{r['score']:.6f}",r["reasoning"]])
    st.download_button(
        "⬇️ Download submission.csv — backup",
        data=out.getvalue(), file_name="submission.csv",
        mime="text/csv", use_container_width=True,
    )

st.divider()
st.caption("ARIA v2 · Modern College of Engineering, Pune (SPPU) · Redrob Hackathon 2026")
