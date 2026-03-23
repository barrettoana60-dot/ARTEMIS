# dashboard_consumidia_updated.py
# CONSUMIDIA — Dashboard completo (versão atualizada)
# - CORREÇÃO: Corrigido o erro 'StreamlitAPIException' ao limpar o anexo após o envio da mensagem.
# - MELHORIA VISUAL: Efeito Liquid-glass aprimorado, ícones integrados aos botões de navegação.
# - MELHORIA FUNCIONAL: Nome do usuário de origem agora é salvo e exibido nos Favoritos.
# - NOVO: Troca de arquivos (dropbox) integrada ao sistema de mensagens.

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from networkx.readwrite import json_graph
from fpdf import FPDF
import json, os, io, re, random, string, time
from pathlib import Path
from datetime import datetime
import glob
import unicodedata
from difflib import SequenceMatcher
from urllib.parse import quote_plus
from urllib.request import urlopen, Request

# extra para ML/search (mantido para futura expansão)
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Default CSS (liquid glass aprimorado)
# -------------------------
DEFAULT_CSS = r'''
:root{
  --glass-bg: rgba(255,255,255,0.05);
  --glass-border: rgba(255,255,255,0.1);
  --glass-highlight: rgba(255,255,255,0.06);
  --accent: #8e44ad;
  --muted-text: #d6d9dc;
  --icon-color: #ffffff;
}

/* Global body fallback */
.css-1d391kg { background: linear-gradient(180deg,#071428 0%, #031926 100%) !important; }

/* Liquid glass boxes used by the app */
.glass-box{
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
  border-radius: 16px;
  padding: 18px;
  box-shadow: 0 8px 32px rgba(4,9,20,0.5);
  backdrop-filter: blur(12px) saturate(1.2);
}

/* Small specular shine */
.glass-box .specular{ position:absolute; right:12px; top:8px; width:80px; height:80px; background: linear-gradient(120deg, rgba(255,255,255,0.05), rgba(255,255,255,0)); transform:rotate(18deg); pointer-events:none; border-radius:50%; }
.color-blob{ position:absolute; pointer-events:none; filter: blur(36px); opacity:0.55; }
.color-blob.b1{ right:-80px; top:-20px; width:240px; height:140px; background: linear-gradient(90deg, rgba(142,68,173,0.28), rgba(41,121,255,0.18)); }
.color-blob.b2{ left:-80px; bottom:-40px; width:220px; height:120px; background: linear-gradient(90deg, rgba(26,188,156,0.28), rgba(255,138,0,0.18)); }

/* Buttons: translucent, sheen, focus, pressed states */
.stButton>button, .stDownloadButton>button{
  background: var(--glass-bg) !important;
  color: var(--muted-text) !important;
  border: 1px solid var(--glass-border) !important;
  padding: 8px 14px !important;
  border-radius: 12px !important;
  box-shadow: 0 4px 12px rgba(3,7,15,0.45) !important;
  backdrop-filter: blur(8px) saturate(1.1) !important;
  transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover{ transform: translateY(-2px); box-shadow: 0 6px 18px rgba(3,7,15,0.6) !important; background: rgba(255,255,255,0.08) !important; }
.stButton>button:active, .stDownloadButton>button:active{ transform: translateY(0); box-shadow: 0 2px 6px rgba(0,0,0,0.45); }

/* Large icon container */
.icon{ display:inline-flex; align-items:center; justify-content:center; }
.icon svg{ width:100%; height:100%; stroke:currentColor; fill:none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.icon.animate svg .draw{ stroke-dashoffset:0; transition: stroke-dashoffset 0.6s ease-out; }

/* Title animation gradient */
.consumidia-title{ font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; font-weight:800; font-size:36px; background: linear-gradient(90deg, #8e44ad, #2979ff, #1abc9c, #ff8a00); -webkit-background-clip: text; background-clip: text; color:transparent; display:inline-block; animation: hue 6s linear infinite; }
@keyframes hue{ 0%{ filter:hue-rotate(0deg);} 100%{ filter:hue-rotate(360deg);} }

/* Make dataframes more readable */
[data-testid='stDataFrameContainer']{ background:transparent !important; }

/* Reduce red severity for Streamlit error boxes by tonalizing them (keeps accessibility) */
.stAlert[data-alert='error']{ background: rgba(255,77,79,0.06); border-color: rgba(255,77,79,0.12); }
'''

# write default CSS file if missing so user can edit later
try:
    p = Path("style.css")
    if not p.exists():
        p.write_text(DEFAULT_CSS, encoding='utf-8')
except Exception:
    pass

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="CONSUMIDIA", layout="wide", initial_sidebar_state="expanded")

# load css (prefer file so user may edit)
css_path = Path("style.css")
if css_path.exists():
    try:
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    except Exception:
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
else:
    st.markdown(f"<style>{DEFAULT_CSS}</style>", unsafe_allow_html=True)

# -------------------------
# Helpers: users + auth
# -------------------------
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
                return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def gen_password(length=8):
    choices = string.ascii_letters + string.digits
    return ''.join(random.choice(choices) for _ in range(length))

# -------------------------
# Sistema de Favoritos (Adicionado para a nova Busca)
# -------------------------
def get_session_favorites():
    """Retorna a lista de favoritos da sessão atual."""
    return st.session_state.get("favorites", [])

def add_to_favorites(result_data):
    """Adiciona um resultado aos favoritos na sessão."""
    favorites = get_session_favorites()
    result_id = f"{int(time.time())}_{random.randint(1000,9999)}"
    favorite_item = {
        "id": result_id,
        "data": result_data,
        "added_at": datetime.utcnow().isoformat()
    }
    # Checagem de duplicatas mais robusta, ignorando a chave de usuário na comparação
    temp_data_to_check = result_data.copy()
    temp_data_to_check.pop('_artemis_username', None)

    existing_contents = []
    for fav in favorites:
        temp_existing = fav["data"].copy()
        temp_existing.pop('_artemis_username', None)
        existing_contents.append(json.dumps(temp_existing, sort_keys=True))

    if json.dumps(temp_data_to_check, sort_keys=True) not in existing_contents:
        favorites.append(favorite_item)
        st.session_state.favorites = favorites
        return True
    return False

def remove_from_favorites(favorite_id):
    """Remove um favorito da lista na sessão."""
    favorites = get_session_favorites()
    new_favorites = [fav for fav in favorites if fav["id"] != favorite_id]
    st.session_state.favorites = new_favorites
    return len(new_favorites) != len(favorites)

def clear_all_favorites():
    """Remove todos os favoritos da sessão."""
    st.session_state.favorites = []
    return True

# -------------------------
# Color palette (pt -> hex)
# -------------------------
PALETA = {
    "verde": "#00c853",
    "laranja": "#ff8a00",
    "amarelo": "#ffd600",
    "vermelho": "#ff3d00",
    "azul": "#2979ff",
    "roxo": "#8e44ad",
    "cinza": "#7f8c8d",
    "preto": "#000000",
    "turquesa": "#1abc9c"
}

def normalize_color(name_or_hex: str):
    if not name_or_hex:
        return None
    s = str(name_or_hex).strip().lower()
    if s in PALETA:
        return PALETA[s]
    if re.match(r"^#([0-9a-f]{3}|[0-9a-f]{6})$", s):
        return s
    return s

# -------------------------
# Session defaults
# -------------------------
for k, v in {
    "authenticated": False,
    "username": None,
    "user_obj": None,
    "df": None,
    "G": nx.Graph(),
    "notes": "",
    "autosave": False,
    "page": "planilha",
    "restored_from_saved": False,
    "favorites": [] # Adicionado para a nova Busca
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------
# Per-user state helpers
# -------------------------
def user_state_file(username):
    return f"artemis_state_{username}.json"

def user_backup_dir(username):
    p = Path("backups") / username
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_state_for_user(username):
    path = user_state_file(username)
    backup_path = None
    try:
        if st.session_state.df is not None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            safe = re.sub(r"[^\w\-_.]", "_", st.session_state.get("uploaded_name") or "planilha")
            backup_filename = f"{safe}_{ts}.csv"
            backup_path = str((user_backup_dir(username) / backup_filename).resolve())
            st.session_state.df.to_csv(backup_path, index=False, encoding="utf-8")
    except Exception:
        backup_path = None
    data = {
        "graph": json_graph.node_link_data(st.session_state.G),
        "notes": st.session_state.notes,
        "uploaded_name": st.session_state.get("uploaded_name", None),
        "backup_csv": backup_path,
        "saved_at": datetime.utcnow().isoformat(),
        "favorites": st.session_state.get("favorites", []) # Adicionado para a nova Busca
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path

def load_state_for_user(username, load_backup_csv=True):
    path = user_state_file(username)
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        try:
            meta = json.load(f)
        except Exception:
            return False
    try:
        st.session_state.G = json_graph.node_link_graph(meta.get("graph", {}))
    except Exception:
        st.session_state.G = nx.Graph()
    st.session_state.notes = meta.get("notes", "")
    st.session_state.uploaded_name = meta.get("uploaded_name", None)
    st.session_state.favorites = meta.get("favorites", []) # Adicionado para a nova Busca
    backup_csv = meta.get("backup_csv")
    if load_backup_csv and backup_csv and os.path.exists(backup_csv):
        try:
            st.session_state.df = pd.read_csv(backup_csv, encoding="utf-8")
        except Exception:
            try:
                st.session_state.df = pd.read_csv(backup_csv, encoding="latin1")
            except Exception:
                st.session_state.df = None
    st.session_state.restored_from_saved = True
    return True

# -------------------------
# Robust spreadsheet reader
# -------------------------
def read_spreadsheet(uploaded_file):
    b = uploaded_file.read()
    buf = io.BytesIO(b)
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        for enc in ("utf-8", "latin1", "cp1252"):
            try:
                buf.seek(0)
                return pd.read_csv(buf, encoding=enc)
            except Exception:
                continue
        buf.seek(0)
        return pd.read_csv(buf, engine="python", on_bad_lines="skip")
    else:
        buf.seek(0)
        return pd.read_excel(buf)

def _normalize_text(value):
    value = str(value or "").strip().lower()
    value = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in value if not unicodedata.combining(ch))

def build_corpus_from_df(df):
    if df is None or df.empty:
        return pd.DataFrame(), []
    safe = df.fillna("").copy()
    safe["_doc_text"] = safe.astype(str).agg(" ".join, axis=1).map(_normalize_text)
    return safe, safe["_doc_text"].tolist()

def semantic_search(df, query, top_k=20):
    indexed, docs = build_corpus_from_df(df)
    if indexed.empty or not query.strip():
        return pd.DataFrame()
    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=6000)
    matrix = vect.fit_transform(docs + [_normalize_text(query)])
    query_vec = matrix[-1]
    scores = cosine_similarity(matrix[:-1], query_vec).ravel()
    indexed["_score"] = scores
    indexed = indexed[indexed["_score"] > 0]
    return indexed.sort_values("_score", ascending=False).head(top_k)

def suggest_web_articles(query, max_results=6):
    if not query.strip():
        return []
    q = quote_plus(query)
    url = f"https://api.crossref.org/works?query={q}&rows={max_results}&sort=score&order=desc"
    try:
        req = Request(url, headers={"User-Agent": "ARTEMIS/1.0 (Research assistant)"})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = data.get("message", {}).get("items", [])
    except Exception:
        return []
    out = []
    for it in items:
        title = (it.get("title") or ["Sem título"])[0]
        doi = it.get("DOI")
        link = f"https://doi.org/{doi}" if doi else (it.get("URL") or "")
        year = (it.get("published-print", {}).get("date-parts", [[None]])[0][0]
                or it.get("published-online", {}).get("date-parts", [[None]])[0][0])
        out.append({"title": title, "year": year, "link": link})
    return out

def suggest_related_terms(df, query, top_k=8):
    if df is None or df.empty:
        return []
    _, docs = build_corpus_from_df(df)
    vect = TfidfVectorizer(stop_words="english", max_features=1500)
    matrix = vect.fit_transform(docs + [_normalize_text(query)])
    query_vec = matrix[-1]
    terms = np.array(vect.get_feature_names_out())
    weights = (query_vec.toarray()[0])
    ranked = terms[np.argsort(weights)[::-1]]
    return [t for t in ranked if len(t) > 3][:top_k]

NATIONALITY_COORDS = {
    "brasil": (-14.2, -51.9), "brasileiro": (-14.2, -51.9),
    "portugal": (39.4, -8.2), "portugues": (39.4, -8.2),
    "eua": (37.1, -95.7), "estados unidos": (37.1, -95.7), "americano": (37.1, -95.7),
    "canada": (56.1, -106.3), "franca": (46.2, 2.2), "frances": (46.2, 2.2),
    "espanha": (40.4, -3.7), "alemanha": (51.2, 10.4), "italia": (41.9, 12.5),
    "japao": (36.2, 138.3), "china": (35.9, 104.2), "india": (20.6, 78.9)
}

def build_image_index():
    base_dirs = [Path("user_files"), Path("backups"), Path(".")]
    image_paths = []
    for base in base_dirs:
        if not base.exists():
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
            image_paths.extend(base.rglob(ext))
    return sorted(set(image_paths))

# -------------------------
# Graph creation + plotly 3D
# -------------------------
def criar_grafo(df):
    G = nx.Graph()
    if df is None:
        return G
    cols_lower = {c.lower(): c for c in df.columns}
    for _, row in df.iterrows():
        autor = str(row.get(cols_lower.get("autor", ""), "") or "").strip()
        titulo = str(row.get(cols_lower.get("título", cols_lower.get("titulo", "")), "") or "").strip()
        ano = str(row.get(cols_lower.get("ano", ""), "") or "").strip()
        tema = str(row.get(cols_lower.get("tema", ""), "") or "").strip()
        if autor:
            k = f"Autor: {autor}"
            G.add_node(k, tipo="Autor", label=autor)
        if titulo:
            k = f"Título: {titulo}"
            G.add_node(k, tipo="Título", label=titulo)
        if ano:
            k = f"Ano: {ano}"
            G.add_node(k, tipo="Ano", label=ano)
        if tema:
            k = f"Tema: {tema}"
            G.add_node(k, tipo="Tema", label=tema)
        if autor and titulo:
            G.add_edge(f"Autor: {autor}", f"Título: {titulo}")
        if titulo and ano:
            G.add_edge(f"Título: {titulo}", f"Ano: {ano}")
        if titulo and tema:
            G.add_edge(f"Título: {titulo}", f"Tema: {tema}")
    return G


def graph_to_plotly_3d(G, show_labels=False, height=600):
    if len(G.nodes()) == 0:
        fig = go.Figure()
        fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig
    pos = nx.spring_layout(G, dim=3, seed=42)
    degrees = dict(G.degree())
    deg_values = [degrees.get(n, 1) for n in G.nodes()]
    min_deg, max_deg = min(deg_values), max(deg_values)
    node_sizes = [8 + ((d - min_deg) / (max_deg - min_deg + 1e-6)) * 18 for d in deg_values]
    x_nodes = [pos[n][0] for n in G.nodes()]
    y_nodes = [pos[n][1] for n in G.nodes()]
    z_nodes = [pos[n][2] for n in G.nodes()]
    x_edges, y_edges, z_edges = [], [], []
    for e in G.edges():
        x_edges += [pos[e[0]][0], pos[e[1]][0], None]
        y_edges += [pos[e[0]][1], pos[e[1]][1], None]
        z_edges += [pos[e[0]][2], pos[e[1]][2], None]
    color_map = {"Autor": PALETA["verde"], "Título": PALETA["roxo"], "Ano": PALETA["azul"], "Tema": PALETA["laranja"]}
    node_colors = [color_map.get(G.nodes[n].get("tipo", ""), PALETA["vermelho"]) for n in G.nodes()]
    labels = [G.nodes[n].get("label", str(n)) for n in G.nodes()]
    hover = [f"<b>{G.nodes[n].get('label','')}</b><br>Tipo: {G.nodes[n].get('tipo','')}" for n in G.nodes()]
    edge_trace = go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges, mode="lines",
        line=dict(color="rgba(200,200,200,0.12)", width=1.2), hoverinfo="none"
    )
    node_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode="markers+text" if show_labels else "markers",
        marker=dict(size=node_sizes, color=node_colors, opacity=0.95, line=dict(width=1)),
        hovertext=hover, hoverinfo="text",
        text=labels if show_labels else None, textposition="top center"
    )
    legend_items = []
    for label, cor in [("Autor", PALETA["verde"]), ("Título", PALETA["roxo"]), ("Ano", PALETA["azul"]), ("Tema", PALETA["laranja"])]:
        legend_items.append(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers",
                                        marker=dict(size=8, color=cor), name=label))
    fig = go.Figure(data=[edge_trace, node_trace] + legend_items)
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=20, b=0), height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#d6d9dc"), orientation="h", y=1.05, x=0.02)
    )
    fig.update_layout(scene_camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)))
    return fig

# -------------------------
# PDF with highlights (returns bytes)
# -------------------------
def _safe_for_pdf(s: str):
    if s is None:
        return ""
    s2 = s.replace("—", "-").replace("–", "-")
    return s2.encode("latin-1", "replace").decode("latin-1")

def generate_pdf_with_highlights(texto, highlight_hex="#ffd600"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for linha in (texto or "").split("\n"):
        parts = re.split(r"(==.*?==)", linha)
        for part in parts:
            if not part:
                continue
            if part.startswith("==") and part.endswith("=="):
                inner = part[2:-2]
                inner_safe = _safe_for_pdf(inner)
                hexv = (highlight_hex or "#ffd600").lstrip("#")
                if len(hexv) == 3:
                    hexv = ''.join([c*2 for c in hexv])
                try:
                    r, g, b = int(hexv[0:2], 16), int(hexv[2:4], 16), int(hexv[4:6], 16)
                except Exception:
                    r, g, b = (255, 214, 0)
                pdf.set_fill_color(r, g, b)
                pdf.set_text_color(0, 0, 0)
                w = pdf.get_string_width(inner_safe) + 2
                pdf.cell(w, 6, txt=inner_safe, border=0, ln=0, fill=True)
                pdf.set_text_color(0, 0, 0)
            else:
                if part:
                    safe_part = _safe_for_pdf(part)
                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(pdf.get_string_width(safe_part), 6, txt=safe_part, border=0, ln=0)
        pdf.ln(6)
    raw = pdf.output(dest="S")
    if isinstance(raw, str):
        return raw.encode("latin-1", "replace")
    elif isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    else:
        return str(raw).encode("latin-1", "replace")

# -------------------------
# ICONS + animation helper
# -------------------------
ICON_KEYS = [
    "login","register","planilha","mapa","anotacoes","graficos",
    "save","logout","restore","download_backup","upload_file","export",
    "add_node","del_node","rename_node","down_pdf","busca",
    "favoritos", "trash", "attachment"
]
for k in ICON_KEYS:
    if f"anim_ts_{k}" not in st.session_state:
        st.session_state[f"anim_ts_{k}"] = 0.0

def _now():
    return time.time()

def mark_icon_animate(key):
    st.session_state[f"anim_ts_{key}"] = _now()

ICON_SVGS = {
    "planilha": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><rect class="draw" x="2" y="3" width="20" height="18" rx="2"/><path class="draw" d="M7 8h10M7 12h10M7 16h10"/></svg>',
    "anotacoes": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path class="draw" d="M3 21v-3a4 4 0 0 1 4-4h2l6-6 4 4-6 6" /><path class="draw pen" d="M14 7l3 3" /><path class="draw" d="M6 18l4-4"/></svg>',
    "mapa": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><polyline class="draw" points="3 6 9 3 15 6 21 3"/><polyline class="draw" points="3 18 9 15 15 18 21 15"/><line class="draw" x1="9" y1="3" x2="9" y2="15"/></svg>',
    "graficos": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><rect class="draw" x="3" y="10" width="4" height="10" rx="1"/><rect class="draw" x="9" y="6" width="4" height="14" rx="1"/><rect class="draw" x="15" y="2" width="4" height="18" rx="1"/></svg>',
    "save": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path class="draw" d="M5 21h14a1 1 0 0 0 1-1V7L16 3H8L4 7v13a1 1 0 0 0 1 1z"/><path class="draw" d="M9 9h6v6H9z"/></svg>',
    "logout": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path class="draw" d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><path class="draw" d="M16 17l5-5-5-5"/><path class="draw" d="M21 12H9"/></svg>',
    "upload_file": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path class="draw" d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline class="draw" points="17 8 12 3 7 8"/><line class="draw" x1="12" y1="3" x2="12" y2="15"/></svg>',
    "download_backup": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path class="draw" d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline class="draw" points="7 10 12 15 17 10"/><line class="draw" x1="12" y1="15" x2="12" y2="3"/></svg>',
    "export": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path class="draw" d="M12 3v12"/><polyline class="draw" points="8 7 12 3 16 7"/><rect class="draw" x="3" y="15" width="18" height="6" rx="1"/></svg>',
    "add_node": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><circle class="draw" cx="12" cy="12" r="9"/><line class="draw" x1="12" y1="8" x2="12" y2="16"/><line class="draw" x1="8" y1="12" x2="16" y2="12"/></svg>',
    "del_node": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><circle class="draw" cx="12" cy="12" r="9"/><line class="draw" x1="9" y1="9" x2="15" y2="15"/><line class="draw" x1="15" y1="9" x2="9" y2="15"/></svg>',
    "rename_node": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path class="draw" d="M3 21v-3a4 4 0 0 1 4-4h2"/><path class="draw" d="M14 7l3 3"/><path class="draw" d="M10 14l6-6"/></svg>',
    "down_pdf": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path class="draw" d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline class="draw" points="7 10 12 15 17 10"/></svg>',
    "login": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path class="draw" d="M12 2a5 5 0 0 1 5 5v3"/><path class="draw" d="M21 21v-6a4 4 0 0 0-4-4H7"/></svg>',
    "register": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><circle class="draw" cx="12" cy="8" r="3"/><path class="draw" d="M21 21v-2a4 4 0 0 0-4-4H7a4 4 0 0 0-4 4v2"/></svg>',
    "restore": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path class="draw" d="M21 12a9 9 0 1 1-3-6.7"/><polyline class="draw" points="21 6 21 12 15 12"/></svg>',
    "busca": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="currentColor"><circle cx="11" cy="11" r="6"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
    "favoritos": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path class="draw" d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>',
    "trash": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>',
    "attachment": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>'
}

def icon_html_svg(key, extra_class="", size=36, color=None):
    svg = ICON_SVGS.get(key, "")
    ts = st.session_state.get(f"anim_ts_{key}", 0.0)
    animate = (_now() - ts) < 1.8
    classes = f"icon {key}"
    if extra_class: classes += " " + extra_class
    if animate: classes += " animate"
    col = color or "var(--icon-color)"
    style = f"color:{col}; width:{size}px; height:{size}px; display:inline-block; vertical-align:middle;"
    return f'<span class="{classes}" style="{style}">{svg}</span>'

# -------------------------
# Unified action button helper
# -------------------------
def action_button(label, icon_key, st_key, expanded_label=None, wide=False):
    c_icon, c_btn = st.columns([0.15, 0.85])
    with c_icon:
        st.markdown(f"<div style='margin-top:6px'>{icon_html_svg(icon_key, size=28)}</div>", unsafe_allow_html=True)
    with c_btn:
        clicked = st.button(expanded_label or label, key=st_key, use_container_width=True)
    if clicked:
        mark_icon_animate(icon_key)
    return clicked

# -------------------------
# NAV and UI
# -------------------------
st.markdown("<div style='display:flex; justify-content:center;'><h1 class='consumidia-title'>CONSUMIDIA</h1></div>", unsafe_allow_html=True)

users = load_users()

if not st.session_state.authenticated:
    st.markdown("<div class='glass-box auth' style='max-width:1100px;margin:0 auto; position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Acesso — Faça login ou cadastre-se")
    tabs = st.tabs(["Entrar", "Criar conta"])
    with tabs[0]:
        login_user = st.text_input("Usuário", key="ui_login_user")
        login_pass = st.text_input("Senha", type="password", key="ui_login_pass")
        if st.button("Entrar", "btn_login_main"):
            if login_user and login_user in users and users[login_user].get("password") == login_pass:
                st.session_state.authenticated = True
                st.session_state.username = login_user
                st.session_state.user_obj = users[login_user]
                st.rerun()
            else:
                st.warning("Usuário ou senha incorretos.")
    with tabs[1]:
        reg_name = st.text_input("Nome completo", key="ui_reg_name")
        reg_user = st.text_input("Escolha um username", key="ui_reg_user")
        if st.button("Criar conta", "btn_register_main"):
            if not reg_user or not reg_user.strip():
                st.warning("Escolha um username válido.")
            elif reg_user in users:
                st.warning("Username já existe.")
            else:
                pwd = gen_password(8)
                users[reg_user] = {"name": reg_name or reg_user, "password": pwd, "created_at": datetime.utcnow().isoformat()}
                save_users(users)
                st.success(f"Usuário criado. Username: **{reg_user}** — Senha gerada: **{pwd}**")
                st.info("Anote a senha.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# -------------------------
# After auth: per-user variables
# -------------------------
USERNAME = st.session_state.username
USER_OBJ = st.session_state.user_obj or users.get(USERNAME, {})
USER_STATE = user_state_file(USERNAME)

# try auto restore
if not st.session_state.restored_from_saved and os.path.exists(USER_STATE):
    try:
        load_state_for_user(USERNAME)
        st.success("Estado salvo do usuário restaurado automaticamente.")
    except Exception:
        pass

# -------------------------
# Top controls + nav
# -------------------------
st.markdown("<div class='glass-box' style='padding-top:10px; padding-bottom:10px;'><div class='specular'></div>", unsafe_allow_html=True)

top1, top2 = st.columns([0.6, 0.4])
with top1:
    st.markdown(f"<div style='color:var(--muted-text);font-weight:700; padding-top:8px;'>Usuário: {USER_OBJ.get('name','')}</div>", unsafe_allow_html=True)
with top2:
    nav_right1, nav_right2, nav_right3 = st.columns([1,1,1])
    with nav_right1:
        ui_aut = st.checkbox("Auto-save", value=st.session_state.autosave, key="ui_autosave")
        st.session_state.autosave = ui_aut
    with nav_right2:
        if st.button("Salvar", key=f"btn_save_now", use_container_width=True):
            p = save_state_for_user(USERNAME)
            st.success(f"Progresso salvo.")
    with nav_right3:
        if st.button("Sair", key=f"btn_logout", use_container_width=True):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

st.markdown("<div style='margin-top:-20px'>", unsafe_allow_html=True)
nav1, nav2, nav3, nav4, nav5, nav6 = st.columns(6)
with nav1:
    if st.button("Planilha", key="nav_planilha", use_container_width=True):
        st.session_state.page = "planilha"
        st.rerun()
with nav2:
    if st.button("Mapa", key="nav_mapa", use_container_width=True):
        st.session_state.page = "mapa"
        st.rerun()
with nav3:
    if st.button("Anotações", key="nav_anotacoes", use_container_width=True):
        st.session_state.page = "anotacoes"
        st.rerun()
with nav4:
    if st.button("Gráficos", key="nav_graficos", use_container_width=True):
        st.session_state.page = "graficos"
        st.rerun()
with nav5:
    if st.button("Pesquisa IA", key="nav_busca", use_container_width=True):
        st.session_state.page = "busca"
        st.rerun()
with nav6:
    if st.button("Análise IA", key="nav_analise", use_container_width=True):
        st.session_state.page = "analise"
        st.rerun()
st.markdown("</div></div>", unsafe_allow_html=True)
st.markdown("---")


# -------------------------
# Page content dispatcher
# -------------------------
if st.session_state.page == "planilha":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Planilha / Backup")
    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("Restaurar estado salvo", key="btn_restore_state"):
            ok = load_state_for_user(USERNAME)
            if ok:
                st.success("Estado salvo restaurado (grafo + anotações).")
            else:
                st.info("Nenhum estado salvo encontrado.")
    with col2:
        meta = None
        if os.path.exists(USER_STATE):
            try:
                with open(USER_STATE, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = None
        if meta and meta.get("backup_csv") and os.path.exists(meta.get("backup_csv")):
            st.write("Backup CSV encontrado:")
            st.text(meta.get("backup_csv"))
            with open(meta.get("backup_csv"), "rb") as fp:
                st.download_button("⬇ Baixar backup CSV", data=fp, file_name=os.path.basename(meta.get("backup_csv")), mime="text/csv")
        else:
            st.write("Nenhum backup CSV automático encontrado ainda.")

    uploaded = st.file_uploader("Carregue .csv ou .xlsx (cada linha será um nó)", type=["csv", "xlsx"], key=f"u_{USERNAME}")
    if uploaded:
        try:
            df = read_spreadsheet(uploaded)
            st.session_state.df = df
            st.session_state.uploaded_name = uploaded.name
            st.session_state.G = criar_grafo(df)
            st.success("Planilha carregada com sucesso.")
            if st.session_state.autosave:
                save_state_for_user(USERNAME)
        except Exception as e:
            st.error(f"Erro ao ler planilha: {e}")

    if st.session_state.df is not None:
        st.write("Visualização da planilha:")
        st.dataframe(st.session_state.df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "mapa":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Mapa Mental 3D — Editor")

    if st.session_state.G.nodes():
        with st.expander("Editar Nós do Mapa"):
            left, right = st.columns([2,1])
            with left:
                new_node = st.text_input("Nome do novo nó", value="", key=f"nm_name_{USERNAME}")
                new_tipo = st.selectbox("Tipo", ["Outro", "Autor", "Título", "Ano", "Tema"], index=0, key=f"nm_tipo_{USERNAME}")
                connect_to = st.selectbox("Conectar a (opcional)", options=["Nenhum"] + list(st.session_state.G.nodes), index=0, key=f"nm_connect_{USERNAME}")
                if st.button("Adicionar nó", key=f"btn_add_{USERNAME}"):
                    n = new_node.strip()
                    if not n: st.warning("Nome inválido.")
                    elif n in st.session_state.G.nodes: st.warning("Nó já existe.")
                    else:
                        st.session_state.G.add_node(n, tipo=new_tipo, label=n)
                        if connect_to != "Nenhum":
                            st.session_state.G.add_edge(n, connect_to)
                        st.success(f"Nó '{n}' adicionado.")
                        if st.session_state.autosave: save_state_for_user(USERNAME)
                        st.rerun()
            with right:
                del_n = st.selectbox("Excluir nó", options=[""] + list(st.session_state.G.nodes), key=f"del_{USERNAME}")
                if st.button("Excluir nó", key=f"btn_del_{USERNAME}"):
                    if del_n and del_n in st.session_state.G:
                        st.session_state.G.remove_node(del_n)
                        st.success(f"Nó '{del_n}' removido.")
                        if st.session_state.autosave: save_state_for_user(USERNAME)
                        st.rerun()
                st.markdown("---")
                r_old = st.selectbox("Renomear: selecione nó", options=[""] + list(st.session_state.G.nodes), key=f"r_old_{USERNAME}")
                r_new = st.text_input("Novo nome", key=f"r_new_{USERNAME}")
                if st.button("Renomear", key=f"btn_ren_{USERNAME}"):
                    if r_old and r_new and r_old in st.session_state.G and r_new not in st.session_state.G:
                        nx.relabel_nodes(st.session_state.G, {r_old: r_new}, copy=False)
                        st.success(f"'{r_old}' → '{r_new}'")
                        if st.session_state.autosave: save_state_for_user(USERNAME)
                        st.rerun()

    st.markdown("### Visualização 3D")
    try:
        fig = graph_to_plotly_3d(st.session_state.G, show_labels=False, height=700)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
    except Exception as e:
        st.error(f"Erro ao renderizar grafo: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "anotacoes":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Anotações com Marca-texto")
    st.info("Use ==texto== para marcar (destacar) trechos que serão realçados no PDF.")
    notes = st.text_area("Digite suas anotações (use ==texto== para destacar)", value=st.session_state.notes, height=260, key=f"notes_{USERNAME}")
    st.session_state.notes = notes

    pdf_bytes = generate_pdf_with_highlights(st.session_state.notes)
    st.download_button("Baixar Anotações (PDF)", data=pdf_bytes, file_name="anotacoes_consumidia.pdf", mime="application/pdf")

    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "graficos":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Gráficos Personalizados")
    if st.session_state.df is None:
        st.warning("Carregue uma planilha na aba 'Planilha' para gerar gráficos.")
    else:
        df = st.session_state.df.copy()
        cols = df.columns.tolist()
        c1, c2 = st.columns(2)
        with c1:
            eixo_x = st.selectbox("Eixo X", options=cols, index=0, key=f"x_{USERNAME}")
        with c2:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            eixo_y = st.selectbox("Eixo Y (Opcional)", options=[None] + numeric_cols, key=f"y_{USERNAME}")

        if st.button("Gerar Gráfico"):
            try:
                if eixo_y:
                    fig = px.bar(df, x=eixo_x, y=eixo_y, title=f"{eixo_y} por {eixo_x}")
                else:
                    fig = px.histogram(df, x=eixo_x, title=f"Contagem por {eixo_x}")

                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#d6d9dc"))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao gerar gráficos: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "busca":
    st.markdown("<div class='glass-box' style='position:relative; padding:12px;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Pesquisa IA Unificada")

    backups_df = None
    base = Path("backups")
    if base.exists():
        dfs = []
        for user_dir in sorted(base.iterdir()):
            if not user_dir.is_dir():
                continue
            csvs = sorted(user_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not csvs:
                continue
            try:
                part = pd.read_csv(csvs[0], encoding="utf-8", on_bad_lines="skip")
            except Exception:
                continue
            part["_artemis_username"] = user_dir.name
            dfs.append(part)
        if dfs:
            backups_df = pd.concat(dfs, ignore_index=True)

    if backups_df is None or backups_df.empty:
        st.warning("Nenhum conteúdo indexado ainda. Faça upload e salve para habilitar a busca inteligente.")
    else:
        query = st.text_input("Tema ou pergunta de pesquisa", placeholder="Ex.: impactos da IA na educação superior")
        c1, c2 = st.columns([1, 1])
        with c1:
            run_search = st.button("Executar busca semântica", use_container_width=True)
        with c2:
            run_articles = st.button("Sugerir artigos da internet", use_container_width=True)

        if run_search and query:
            results = semantic_search(backups_df.drop(columns=[c for c in ["_score"] if c in backups_df.columns]), query, top_k=25)
            st.session_state.search_results = results

        results = st.session_state.get("search_results", pd.DataFrame())
        if isinstance(results, pd.DataFrame) and not results.empty:
            st.markdown(f"**{len(results)} resultados semânticos encontrados.**")
            show = results.head(20).copy()
            if "_doc_text" in show.columns:
                show = show.drop(columns=["_doc_text"])
            st.dataframe(show, use_container_width=True)

            st.markdown("### Conexões entre pesquisas semelhantes")
            link_graph = nx.Graph()
            top_res = results.head(15)
            for i, row in top_res.iterrows():
                label = str(row.get("Título", row.get("titulo", f"Registro {i}")))[:80]
                link_graph.add_node(label)
            labels = list(link_graph.nodes())
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    a, b = labels[i], labels[j]
                    sim = SequenceMatcher(None, _normalize_text(a), _normalize_text(b)).ratio()
                    if sim >= 0.5:
                        link_graph.add_edge(a, b, weight=sim)
            st.plotly_chart(graph_to_plotly_3d(link_graph, show_labels=True, height=550), use_container_width=True)

        if run_articles and query:
            web_items = suggest_web_articles(query, max_results=8)
            st.markdown("### Sugestões de artigos na internet")
            if not web_items:
                st.info("Não foi possível buscar artigos externos no momento.")
            for item in web_items:
                st.markdown(f"- **{item['title']}** ({item.get('year') or 's/d'}) — {item.get('link','')}")

        if query:
            st.markdown("### Termos sugeridos para refinar a pesquisa")
            terms = suggest_related_terms(backups_df, query)
            if terms:
                st.write(", ".join(terms))
            else:
                st.caption("Sem termos suficientes para sugerir no momento.")

            st.markdown("### Busca de imagens semelhantes (pastas de usuários + internet)")
            image_idx = build_image_index()
            local_hits = [p for p in image_idx if _normalize_text(query) in _normalize_text(p.name)]
            if local_hits:
                st.write("Arquivos locais semelhantes:")
                for p in local_hits[:12]:
                    st.markdown(f"- {p}")
            else:
                st.caption("Nenhuma imagem local semelhante encontrada pelo nome do arquivo.")
            st.markdown(f"Pesquise imagens externas: https://www.google.com/search?tbm=isch&q={quote_plus(query)}")

    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "analise":
    st.markdown("<div class='glass-box' style='position:relative; padding:12px;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Análise completa da pesquisa")

    if st.session_state.df is None or st.session_state.df.empty:
        st.warning("Carregue uma planilha para gerar análise completa por ano, tema, autor e nacionalidade.")
    else:
        df = st.session_state.df.copy()
        cols_map = {c.lower(): c for c in df.columns}

        c_ano = cols_map.get("ano")
        c_tema = cols_map.get("tema")
        c_autor = cols_map.get("autor")
        c_nat = cols_map.get("nacionalidade")
        c_resumo = cols_map.get("resumo")

        if c_ano:
            fig_ano = px.histogram(df, x=c_ano, title="Distribuição por ano")
            fig_ano.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_ano, use_container_width=True)

        if c_tema:
            top_tema = df[c_tema].astype(str).value_counts().head(12).reset_index()
            top_tema.columns = ["tema", "qtd"]
            fig_tema = px.bar(top_tema, x="tema", y="qtd", title="Temas mais recorrentes")
            fig_tema.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_tema, use_container_width=True)

        if c_autor:
            top_autor = df[c_autor].astype(str).value_counts().head(12).reset_index()
            top_autor.columns = ["autor", "qtd"]
            fig_autor = px.bar(top_autor, x="autor", y="qtd", title="Autores com mais artigos")
            fig_autor.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_autor, use_container_width=True)

        if c_nat:
            nat_count = df[c_nat].astype(str).map(_normalize_text).value_counts().reset_index()
            nat_count.columns = ["nacionalidade", "qtd"]
            nat_count["lat"] = nat_count["nacionalidade"].map(lambda n: NATIONALITY_COORDS.get(n, (None, None))[0])
            nat_count["lon"] = nat_count["nacionalidade"].map(lambda n: NATIONALITY_COORDS.get(n, (None, None))[1])
            geo_df = nat_count.dropna(subset=["lat", "lon"])
            if not geo_df.empty:
                fig_geo = px.scatter_geo(
                    geo_df,
                    lat="lat", lon="lon", size="qtd", color="qtd",
                    hover_name="nacionalidade", projection="orthographic",
                    title="Mapa 3D de nacionalidade dos autores"
                )
                fig_geo.update_geos(showland=True, showcountries=True)
                fig_geo.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_geo, use_container_width=True)
            else:
                st.info("Não foi possível mapear nacionalidades automaticamente. Use nomes de países no campo 'nacionalidade'.")

        if c_resumo:
            st.markdown("### Resumos dos artigos")
            for i, resumo in enumerate(df[c_resumo].dropna().astype(str).head(20), start=1):
                st.markdown(f"**Resumo {i}:** {resumo}")

    st.markdown("</div>", unsafe_allow_html=True)
