# ===== Imports =====
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from networkx.readwrite import json_graph
from fpdf import FPDF
import io
import math
import os
import zipfile

# ===== Configuração da página =====
st.set_page_config(page_title="Artemis Refinado — Dark (Undo/Redo & Export All)", layout="wide")
# CSS de estilo superior
st.markdown(
    """
    <style>
    .top-banner{
        background: linear-gradient(90deg,#0b3a5a 0%, #071428 50%, #001219 100%);
        padding: 18px;
        border-radius: 8px;
        color: #e6f0ff;
        margin-bottom: 12px;
    }
    .title-big { font-size:22px; font-weight:700; margin:0; color:#ffffff }
    .subtitle { font-size:12px; color:#bcd3ee; margin-top:4px }
    .help-card { background:#071428; border:1px solid rgba(255,255,255,0.05); padding:12px; border-radius:8px; color:#dfeffd }
    .cta-btn { font-weight:700; padding:10px 12px; border-radius:8px; }
    .small-note { color:#9fb7d7; font-size:12px }
    </style>
    """, unsafe_allow_html=True
)

# Top banner com instruções e botões de atalho
with st.container():
    st.markdown('<div class="top-banner">', unsafe_allow_html=True)
    cols = st.columns([3,2])
    with cols[0]:
        st.markdown('<p class="title-big">✨ ARTEMIS — Dashboard Refinado (Tema Escuro)</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Mapa mental 3D, editor de nós, merge, export profissional, undo/redo por ação e export ZIP.</p>', unsafe_allow_html=True)
    with cols[1]:
        bcol1, bcol2, bcol3 = st.columns([1,1,1])
        with bcol1:
            if st.button("↶ Desfazer", key="top_undo"):
                st.session_state._top_undo_clicked = True
        with bcol2:
            if st.button("↷ Avançar", key="top_redo"):
                st.session_state._top_redo_clicked = True
        with bcol3:
            if st.button("📦 Exportar tudo (ZIP)", key="top_export_all"):
                st.session_state._top_export_all = True
    st.markdown('</div>', unsafe_allow_html=True)

# Painel de instruções rápidas
with st.container():
    instr_cols = st.columns([1,1,1])
    instr_cols[0].markdown(
        '<div class="help-card"><b>1 — Carregar planilha</b><br>Use .csv ou .xlsx com colunas Autor / Título / Ano / Tema.</div>',
        unsafe_allow_html=True)
    instr_cols[1].markdown(
        '<div class="help-card"><b>2 — Editar o mapa</b><br>Adicione / exclua / renomeie / una nós no editor do Mapa Mental 3D. Cada ação tem rótulo no histórico.</div>',
        unsafe_allow_html=True)
    instr_cols[2].markdown(
        '<div class="help-card"><b>3 — Exportar</b><br>Baixe HTML interativo, CSV, PNG (se suportado), PDF de anotações ou ZIP com tudo.</div>',
        unsafe_allow_html=True)

st.write("")

# ===== Helpers: Export / PDF / Imagens / ZIP =====
def salvar_pdf_bytes(texto, img_png_bytes=None):
    """Cria PDF com texto e opcionalmente insere uma imagem PNG (se possível)."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    # Inserção de imagem só se tivermos bytes e ambiente suportar gravação temporária
    if img_png_bytes:
        try:
            tmp_path = os.path.join(os.getcwd(), "artemis_tmp_img.png")
            with open(tmp_path, "wb") as f:
                f.write(img_png_bytes)
            # inserir imagem (redimensionada)
            pdf.image(tmp_path, x=10, y=8, w=190)
            pdf.ln(85)
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        except Exception:
            pass
    pdf.set_font("Arial", size=12)
    for linha in (texto or " " ).split("\n"):
        pdf.multi_cell(0, 8, linha)
    pdf_bytes = pdf.output(dest='S').encode('latin-1', 'replace')
    return pdf_bytes


def fig_to_html_bytes(fig):
    html = fig.to_html(full_html=True, include_plotlyjs='cdn')
    return html.encode('utf-8')


def fig_to_png_bytes(fig):
    """Tenta gerar PNG via Plotly (kaleido). Se falhar, retorna None."""
    try:
        png = fig.to_image(format='png', scale=2)
        return png
    except Exception:
        return None


def nodes_to_csv_bytes(G):
    rows = []
    for n, attrs in G.nodes(data=True):
        row = {'node_key': n}
        row.update({f'attr_{k}': v for k, v in attrs.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode('utf-8')


def create_export_zip(G, notes_text="", include_png=True):
    """Gera um ZIP em memória com: grafo.html, nos.csv, anotacoes.pdf, opcionalmente grafo.png e grafo.json"""
    # prepara arquivos
    fig = graph_to_plotly_3d(G, anim_mode="none", show_labels=False)
    html_bytes = fig_to_html_bytes(fig)
    csv_bytes = nodes_to_csv_bytes(G)
    png_bytes = None
    if include_png:
        png_bytes = fig_to_png_bytes(fig)
    pdf_bytes = salvar_pdf_bytes(notes_text or "", img_png_bytes=png_bytes)
    json_bytes = pd.io.json.dumps(json_graph.node_link_data(G), indent=2).encode('utf-8')

    # cria zip
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('grafo_interativo.html', html_bytes)
        z.writestr('nos.csv', csv_bytes)
        z.writestr('anotacoes.pdf', pdf_bytes)
        z.writestr('grafo.json', json_bytes)
        if png_bytes:
            z.writestr('grafo.png', png_bytes)
    mem.seek(0)
    return mem

# ===== Grafo e visualização (tema escuro forçado) =====
def criar_grafo(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        autor = str(row.get("Autor", "Desconhecido"))
        titulo = str(row.get("Título", row.get("Titulo", "Desconhecido")))
        ano = str(row.get("Ano", "Desconhecido"))
        tema = str(row.get("Tema", "Desconhecido"))

        if autor:
            G.add_node(autor, tipo="Autor", label=autor, anos=ano)
        if titulo:
            G.add_node(titulo, tipo="Título", label=titulo)
        if ano:
            G.add_node(ano, tipo="Ano", label=ano)
        if tema:
            G.add_node(tema, tipo="Tema", label=tema)

        if autor and titulo:
            G.add_edge(autor, titulo)
        if titulo and ano:
            G.add_edge(titulo, ano)
        if titulo and tema:
            G.add_edge(titulo, tema)
    return G

# ===== Histórico de Ações: undo/redo por rótulo =====
def init_action_stack(G, label="Inicial"):
    data = json_graph.node_link_data(G)
    st.session_state.action_stack = [{'label': label, 'data': data}]
    st.session_state.action_pos = 0


def push_action(label):
    """Registra o estado atual do grafo com um rótulo. Trunca possíveis "redo" após a posição atual."""
    G = st.session_state.get('G', nx.Graph())
    data = json_graph.node_link_data(G)
    stack = st.session_state.get('action_stack', [])
    pos = st.session_state.get('action_pos', -1)
    # trunca
    stack = stack[:pos+1]
    stack.append({'label': label, 'data': data})
    st.session_state.action_stack = stack
    st.session_state.action_pos = len(stack)-1


def undo_action():
    stack = st.session_state.get('action_stack', [])
    pos = st.session_state.get('action_pos', -1)
    if pos <= 0:
        return False, "Nada para desfazer."
    pos -= 1
    st.session_state.action_pos = pos
    G = json_graph.node_link_graph(stack[pos]['data'])
    st.session_state.G = G
    return True, f"Revertido para: {stack[pos]['label']}"


def redo_action():
    stack = st.session_state.get('action_stack', [])
    pos = st.session_state.get('action_pos', -1)
    if pos >= len(stack)-1:
        return False, "Nada para avançar."
    pos += 1
    st.session_state.action_pos = pos
    G = json_graph.node_link_graph(stack[pos]['data'])
    st.session_state.G = G
    return True, f"Avançado para: {stack[pos]['label']}"


def go_to_action(index):
    stack = st.session_state.get('action_stack', [])
    if not (0 <= index < len(stack)):
        return False, "Índice inválido."
    st.session_state.action_pos = index
    G = json_graph.node_link_graph(stack[index]['data'])
    st.session_state.G = G
    return True, f"Definido estado para: {stack[index]['label']}"

# ===== Funções de edição de nós (chamam push_action após a alteração) =====
def add_node_graph(name, tipo="Outro", connect_to=None):
    if not name or not name.strip():
        return False, "Nome inválido."
    G = st.session_state.get("G", nx.Graph())
    if name in G:
        return False, "Já existe um nó com esse nome."
    G.add_node(name, tipo=tipo, label=name)
    if connect_to and connect_to in G:
        G.add_edge(name, connect_to)
    st.session_state.G = G
    push_action(f"Adicionar nó: {name}")
    return True, "Nó criado com sucesso."


def delete_node_graph(name):
    G = st.session_state.get("G", nx.Graph())
    if name not in G:
        return False, "Nó não encontrado."
    G.remove_node(name)
    st.session_state.G = G
    push_action(f"Excluir nó: {name}")
    return True, "Nó excluído com sucesso."


def rename_node_graph(old_name, new_name):
    G = st.session_state.get("G", nx.Graph())
    if old_name not in G:
        return False, "Nó original não encontrado."
    if not new_name or not new_name.strip():
        return False, "Novo nome inválido."
    if new_name in G:
        return False, "Já existe um nó com o novo nome. Escolha outro."
    mapping = {old_name: new_name}
    nx.relabel_nodes(G, mapping, copy=False)
    if new_name in G:
        G.nodes[new_name]['label'] = new_name
    st.session_state.G = G
    push_action(f"Renomear nó: {old_name} → {new_name}")
    return True, "Nó renomeado com sucesso."


def merge_nodes(G, a, b, new_name=None, keep_attrs='merge'):
    if a not in G or b not in G:
        return False, "Um dos nós não existe."
    if a == b:
        return False, "Escolha dois nós diferentes."
    if new_name is None or not new_name.strip():
        new_name = a

    if new_name in G and new_name not in (a, b):
        return False, "Já existe um nó com o novo nome."

    # cria novo nó com atributos combinados
    attrs_a = dict(G.nodes[a])
    attrs_b = dict(G.nodes[b])

    merged_attrs = {}
    if keep_attrs == 'a':
        merged_attrs = attrs_a
    elif keep_attrs == 'b':
        merged_attrs = attrs_b
    else:  # merge heurístico
        keys = set(list(attrs_a.keys()) + list(attrs_b.keys()))
        for k in keys:
            va = attrs_a.get(k)
            vb = attrs_b.get(k)
            if va and not vb:
                merged_attrs[k] = va
            elif vb and not va:
                merged_attrs[k] = vb
            elif va and vb:
                if isinstance(va, str) and isinstance(vb, str):
                    if va == vb:
                        merged_attrs[k] = va
                    else:
                        merged_attrs[k] = f"{va} | {vb}"
                else:
                    merged_attrs[k] = va
            else:
                merged_attrs[k] = va or vb

    final_name = new_name
    G.add_node(final_name, **merged_attrs)
    neighbors = set(G.neighbors(a)) | set(G.neighbors(b))
    neighbors.discard(a)
    neighbors.discard(b)
    for n in neighbors:
        G.add_edge(final_name, n)
    try:
        G.remove_node(a)
    except Exception:
        pass
    try:
        if b in G:
            G.remove_node(b)
    except Exception:
        pass
    st.session_state.G = G
    push_action(f"Unir nós: {a} + {b} → {final_name}")
    return True, f"Nós unidos em '{final_name}'."

# ===== Rendering do grafo (sem alterações significativas) =====
def graph_to_plotly_3d(G, anim_mode="none", anim_steps=36, angle_deg=0, zoom_pulse_strength=0.12, show_labels=False, anim_speed=50):
    if len(G.nodes()) == 0:
        fig = go.Figure()
        fig.update_layout(height=350, paper_bgcolor="#071428", plot_bgcolor="#071428")
        return fig

    pos = nx.spring_layout(G, dim=3, seed=42)

    degrees = dict(G.degree())
    deg_values = [degrees.get(n, 1) for n in G.nodes()]
    min_deg, max_deg = min(deg_values), max(deg_values)
    node_sizes = [10 + ((d - min_deg) / (max_deg - min_deg + 1e-6)) * 18 for d in deg_values]

    x_nodes = [pos[n][0] for n in G.nodes()]
    y_nodes = [pos[n][1] for n in G.nodes()]
    z_nodes = [pos[n][2] for n in G.nodes()]

    x_edges, y_edges, z_edges = [], [], []
    for e in G.edges():
        x_edges += [pos[e[0]][0], pos[e[1]][0], None]
        y_edges += [pos[e[0]][1], pos[e[1]][1], None]
        z_edges += [pos[e[0]][2], pos[e[1]][2], None]

    paper_bg = "#071428"
    plot_bg = "#071428"
    node_outline = "rgba(255,255,255,0.9)"
    label_color = "white"
    hover_bg = "white"
    hover_font = "black"
    legend_font_color = "white"

    edge_trace = go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges, mode="lines",
        line=dict(color="rgba(200,200,200,0.15)", width=1.5), hoverinfo="none"
    )

    color_map = {"Autor": "#2ecc71", "Título": "#8e44ad", "Titulo": "#8e44ad", "Ano": "#3498db", "Tema": "#f39c12"}
    node_colors = []
    hover_texts = []
    labels = []
    for n in G.nodes():
        tipo = G.nodes[n].get("tipo", "")
        color = color_map.get(tipo, "#e74c3c")
        node_colors.append(color)
        label = G.nodes[n].get("label", str(n))
        anos = G.nodes[n].get("anos", "")
        labels.append(label)
        if tipo == "Autor":
            hover_texts.append(f"<b style='color:black'>{label}</b><br>Tipo: Autor<br>Ano(s): {anos}")
        else:
            hover_texts.append(f"<b style='color:black'>{label}</b><br>Tipo: {tipo}")

    node_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode="markers+text" if show_labels else "markers",
        marker=dict(size=node_sizes, color=node_colors, opacity=0.98,
                    line=dict(width=1.5, color=node_outline)),
        hovertext=hover_texts, hoverinfo="text",
        text=labels if show_labels else None,
        textposition="top center",
        textfont=dict(color=label_color, size=11)
    )

    legend_items = []
    for label, cor in [("Autor", "#2ecc71"), ("Título", "#8e44ad"), ("Ano", "#3498db"), ("Tema", "#f39c12")]:
        legend_items.append(
            go.Scatter3d(x=[None], y=[None], z=[None], mode="markers",
                         marker=dict(size=10, color=cor), name=label)
        )

    fig = go.Figure(data=[edge_trace, node_trace] + legend_items)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='auto',
            bgcolor=plot_bg
        ),
        margin=dict(l=0, r=0, t=20, b=0),
        height=700,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        legend=dict(font=dict(color=legend_font_color)),
        hoverlabel=dict(bgcolor=hover_bg, font_size=12, font_color=hover_font)
    )

    base_r = 1.35
    base_z = 0.95

    if anim_mode == "manual_slider":
        rad = math.radians(angle_deg % 360)
        x = base_r * math.cos(rad)
        y = base_r * math.sin(rad)
        z = base_z
        fig.update_layout(scene_camera=dict(eye=dict(x=x, y=y, z=z)))
    elif anim_mode == "autorotate":
        frames = []
        r = base_r
        z = base_z
        for i in range(anim_steps):
            ang = 2 * math.pi * i / anim_steps
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            cam = dict(eye=dict(x=x, y=y, z=z))
            frames.append(go.Frame(layout=dict(scene=dict(camera=cam)), name=str(i)))
        fig.frames = frames
        fig.update_layout(updatemenus=[
            dict(type="buttons", showactive=False, y=0.05, x=0.0, xanchor="left",
                 buttons=[
                     dict(label="▶︎ Play", method="animate",
                          args=[None, {"frame": {"duration": anim_speed, "redraw": True}, "fromcurrent": True,
                                       "transition": {"duration": 300}}]),
                     dict(label="⏸ Pause", method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate",
                                         "transition": {"duration": 0}}])
                 ])
        ])
    elif anim_mode == "zoom_pulse":
        frames = []
        r = base_r
        for i in range(anim_steps):
            t = i / anim_steps
            z = base_z * (1 + zoom_pulse_strength * math.sin(2 * math.pi * t))
            ang = 0.5 * math.sin(2 * math.pi * t)
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            cam = dict(eye=dict(x=x, y=y, z=z))
            frames.append(go.Frame(layout=dict(scene=dict(camera=cam)), name=str(i)))
        fig.frames = frames
        fig.update_layout(updatemenus=[
            dict(type="buttons", showactive=False, y=0.05, x=0.0, xanchor="left",
                 buttons=[
                     dict(label="▶︎ Play", method="animate",
                          args=[None, {"frame": {"duration": anim_speed, "redraw": True}, "fromcurrent": True,
                                       "transition": {"duration": 300}}]),
                     dict(label="⏸ Pause", method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate",
                                         "transition": {"duration": 0}}])
                 ])
        ])
    else:
        fig.update_layout(scene_camera=dict(eye=dict(x=base_r, y=base_r, z=base_z)))

    fig.layout.transition = {'duration': 400, 'easing': 'cubic-in-out'}
    return fig

# ===== Upload da planilha e interface =====
uploaded_file = st.file_uploader("Carregue sua planilha (.csv ou .xlsx)", type=["csv", "xlsx"])

# Captura cliques do topo
if st.session_state.get("_top_undo_clicked", False):
    st.session_state["_top_undo_clicked"] = False
    ok, msg = undo_action()
    if ok:
        st.success(msg)
        st.experimental_rerun()
    else:
        st.info(msg)

if st.session_state.get("_top_redo_clicked", False):
    st.session_state["_top_redo_clicked"] = False
    ok, msg = redo_action()
    if ok:
        st.success(msg)
        st.experimental_rerun()
    else:
        st.info(msg)

if st.session_state.get("_top_export_all", False):
    st.session_state["_top_export_all"] = False
    if uploaded_file and "G" in st.session_state:
        mem = create_export_zip(st.session_state.G, notes_text=st.session_state.get('notas', ''), include_png=True)
        st.download_button("⬇️ Baixar ZIP (Exportar tudo)", data=mem, file_name="artemis_export_all.zip", mime="application/zip")
    else:
        st.info("Carregue uma planilha e gere um grafo antes de exportar tudo.")

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        else:
            df = pd.read_excel(uploaded_file)

        df.columns = [c.strip().capitalize() for c in df.columns]
        st.subheader("📊 Visualização da Planilha")
        st.dataframe(df, use_container_width=True)

        # inicializa sessão e action stack se necessário
        if "G" not in st.session_state:
            st.session_state.G = criar_grafo(df)
            init_action_stack(st.session_state.G, label="Carregado da planilha")
        if "action_stack" not in st.session_state:
            init_action_stack(st.session_state.G, label="Estado inicial")

        G = st.session_state.G

        tabs = st.tabs(["Mapa Mental 3D", "Gráfico Personalizado", "Anotações & Export"])

        # --- Aba 1: Mapa Mental 3D ---
        with tabs[0]:
            st.markdown("### 🌐 Mapa Mental 3D — Tema Escuro")
            top_cols = st.columns([3,1,1])
            with top_cols[0]:
                st.caption("Editor rápido de nós (criar / excluir / renomear / unir). Tema escuro fixo.")
            with top_cols[1]:
                if st.button("Desfazer (Undo)", key="undo_btn"):
                    ok, msg = undo_action()
                    if ok:
                        st.success(msg)
                        st.experimental_rerun()
                    else:
                        st.info(msg)
            with top_cols[2]:
                if st.button("Avançar (Redo)", key="redo_btn"):
                    ok, msg = redo_action()
                    if ok:
                        st.success(msg)
                        st.experimental_rerun()
                    else:
                        st.info(msg)

            # Histórico interativo: selecionar e ir para ação específica
            st.markdown("**Histórico de ações (clique para navegar)**")
            stack = st.session_state.get('action_stack', [])
            labels = [f"{i}: {it['label']}" for i, it in enumerate(stack)]
            cur_pos = st.session_state.get('action_pos', 0)
            sel = st.selectbox("Ir para ação (histórico)", options=labels, index=cur_pos, key='action_history_select')
            if st.button("Ir para ação selecionada"):
                idx = labels.index(sel)
                ok, msg = go_to_action(idx)
                if ok:
                    st.success(msg)
                    st.experimental_rerun()
                else:
                    st.error(msg)

            with st.expander("🛠️ Editor de Nós (Adicionar / Excluir / Renomear / Unir)", expanded=True):
                st.markdown("**Criar novo nó**")
                c1, c2, c3 = st.columns([3,2,3])
                with c1:
                    new_node_name = st.text_input("Nome do novo nó", value="", key="new_node_name")
                with c2:
                    new_node_tipo = st.selectbox("Tipo", options=["Autor", "Título", "Ano", "Tema", "Outro"], index=4, key="new_node_tipo")
                with c3:
                    connect_to = st.selectbox("Conectar a (opcional)", options=["Nenhum"] + list(G.nodes()), index=0, key="connect_to_node")
                if st.button("➕ Adicionar nó"):
                    conn = None if connect_to == "Nenhum" else connect_to
                    ok, msg = add_node_graph(new_node_name.strip(), tipo=new_node_tipo, connect_to=conn)
                    if ok:
                        st.success(msg)
                        st.experimental_rerun()
                    else:
                        st.error(msg)

                st.markdown("---")
                st.markdown("**Excluir nó**")
                if G.nodes():
                    del_node = st.selectbox("Selecione o nó para excluir", options=list(G.nodes()), key="del_node_select")
                    if st.button("🗑️ Excluir nó"):
                        ok, msg = delete_node_graph(del_node)
                        if ok:
                            st.success(msg)
                            st.experimental_rerun()
                        else:
                            st.error(msg)
                else:
                    st.info("Grafo vazio.")

                st.markdown("---")
                st.markdown("**Renomear nó**")
                if G.nodes():
                    rcol1, rcol2 = st.columns(2)
                    with rcol1:
                        old_name = st.selectbox("Nó atual", options=list(G.nodes()), key="rename_old")
                    with rcol2:
                        new_name = st.text_input("Novo nome", value="", key="rename_new")
                    if st.button("✏️ Renomear nó"):
                        ok, msg = rename_node_graph(old_name, new_name.strip())
                        if ok:
                            st.success(msg)
                            st.experimental_rerun()
                        else:
                            st.error(msg)

                st.markdown("---")
                st.markdown("**Unir (merge) dois nós**")
                if len(G.nodes()) >= 2:
                    mcol1, mcol2 = st.columns(2)
                    with mcol1:
                        node_a = st.selectbox("Nó A", options=list(G.nodes()), key="merge_a")
                    with mcol2:
                        node_b = st.selectbox("Nó B", options=[n for n in G.nodes() if n != node_a], key="merge_b")
                    merge_name = st.text_input("Nome do nó resultante (deixe vazio para usar A)", key="merge_newname")
                    keep_choice = st.selectbox("Preservar atributos de:", options=["merge (inteligente)", "a", "b"], index=0, key="merge_keep")
                    if st.button("🔗 Unir nós"):
                        keep_mode = 'merge' if keep_choice.startswith('merge') else ( 'a' if keep_choice=='a' else 'b' )
                        ok, msg = merge_nodes(G, node_a, node_b, new_name=(merge_name.strip() or None), keep_attrs=keep_mode)
                        if ok:
                            st.success(msg)
                            st.experimental_rerun()
                        else:
                            st.error(msg)
                else:
                    st.info("Precisam existir pelo menos 2 nós para unir.")

            # Controles de visualização e animação
            with st.expander("🔧 Controles de Visualização", expanded=False):
                anim_mode = st.selectbox("Modo de Animação", options=[
                    ("Nenhuma", "none"),
                    ("Auto-rotacionar (play/pause)", "autorotate"),
                    ("Controle manual (slider)", "manual_slider"),
                    ("Zoom pulse", "zoom_pulse")
                ], index=0, format_func=lambda x: x[0], key="anim_mode")[1]
                anim_steps = st.slider("Qualidade/frames", min_value=12, max_value=96, value=36, step=12, key="anim_steps")
                anim_speed = st.slider("Velocidade da animação (ms por frame)", min_value=10, max_value=400, value=60, step=10, key="anim_speed")
                if anim_mode == "manual_slider":
                    angle_deg = st.slider("Controle de ângulo (graus)", min_value=0, max_value=360, value=0, key="angle_deg")
                else:
                    angle_deg = 0
                if anim_mode == "zoom_pulse":
                    zoom_pulse_strength = st.slider("Força do zoom pulse", min_value=0.0, max_value=0.6, value=0.12, step=0.02, key="zoom_pulse_strength")
                else:
                    zoom_pulse_strength = 0.12
                show_labels = st.checkbox("Mostrar rótulos ao lado dos nós", value=False, key="show_labels")
                st.caption("Tema escuro aplicado automaticamente — sem opção de tema claro.")

            # Renderiza grafo com o modo de animação selecionado
            G = st.session_state.G
            fig = graph_to_plotly_3d(G,
                                     anim_mode=anim_mode,
                                     anim_steps=anim_steps,
                                     angle_deg=angle_deg,
                                     zoom_pulse_strength=zoom_pulse_strength,
                                     show_labels=show_labels,
                                     anim_speed=anim_speed)
            config = {"displayModeBar": True,
                      "toImageButtonOptions": {"format": "png", "filename": "artemis_grafo", "scale": 2}}
            st.plotly_chart(fig, use_container_width=True, config=config)

        # --- Aba 2: Gráfico Personalizado ---
        with tabs[1]:
            st.subheader("📈 Gráfico Personalizável")
            numeric_cols = df.select_dtypes(include=['int64','float64','float32','int32']).columns.tolist()
            str_cols = df.select_dtypes(include=['object']).columns.tolist()

            if len(str_cols) + len(numeric_cols) == 0:
                st.info("Planilha não contém colunas utilizáveis para o gráfico.")
            else:
                x_axis = st.selectbox("Eixo X", options=str_cols + numeric_cols, index=0)
                if numeric_cols:
                    y_axis = st.selectbox("Eixo Y", options=numeric_cols, index=0)
                else:
                    y_axis = None
                color_col = st.selectbox("Cor (opcional)", options=[None] + str_cols, index=0)

                if st.button("Gerar gráfico"):
                    try:
                        if y_axis:
                            fig_custom = px.bar(df, x=x_axis, y=y_axis, color=color_col, barmode="group",
                                                title=f"{y_axis} por {x_axis}")
                        else:
                            fig_custom = px.histogram(df, x=x_axis, color=color_col, title=f"Contagem por {x_axis}")
                        fig_custom.update_layout(transition={'duration':300}, paper_bgcolor="#071428", plot_bgcolor="#071428")
                        st.plotly_chart(fig_custom, use_container_width=True)
                    except Exception as e:
                        st.error(f"Erro ao gerar gráfico: {e}")

        # --- Aba 3: Anotações & Export ---
        with tabs[2]:
            st.subheader("📝 Anotações & Export Profissional")
            if "notas" not in st.session_state:
                st.session_state.notas = ""
            st.session_state.notas = st.text_area("Escreva suas observações aqui:", value=st.session_state.notas, height=220)

            left, right = st.columns([2,1])
            with left:
                st.markdown("#### 📦 Exportar / Download (atalhos disponíveis no topo)")
                fig_for_export = graph_to_plotly_3d(st.session_state.G, anim_mode="none", show_labels=False)
                html_bytes = fig_to_html_bytes(fig_for_export)
                csv_nodes = nodes_to_csv_bytes(st.session_state.G)
                png_bytes = fig_to_png_bytes(fig_for_export)
                pdf_bytes = salvar_pdf_bytes(st.session_state.notas or "Sem anotações.", img_png_bytes=png_bytes)

                st.download_button("⬇️ Baixar gráfico interativo (HTML)", data=html_bytes,
                                   file_name="grafo_artemis.html", mime="text/html")
                st.download_button("⬇️ Baixar tabela de nós (CSV)", data=csv_nodes,
                                   file_name="nos_artemis.csv", mime="text/csv")
                st.download_button("⬇️ Baixar anotações (PDF)", data=pdf_bytes,
                                   file_name="anotacoes_artemis.pdf", mime="application/pdf")

                if png_bytes:
                    st.download_button("⬇️ Baixar imagem do grafo (PNG)", data=png_bytes,
                                       file_name="grafo_artemis.png", mime="image/png")
                else:
                    st.info("Geração PNG não disponível no ambiente (kaleido ausente). Use o HTML interativo ou o botão de imagem do modo-bar do gráfico.")

                st.markdown("---")
                st.markdown("#### Histórico de ações")
                stack = st.session_state.get('action_stack', [])
                cur = st.session_state.get('action_pos', 0)
                st.write(f"Posição atual: {cur} — Ação: {stack[cur]['label'] if stack else '—'}")
                # mostra lista simples com os últimos 20 rótulos
                for i, it in enumerate(stack[-20:]):
                    idx = len(stack) - len(stack[-20:]) + i
                    marker = " <-- atual" if idx == cur else ""
                    st.text(f"{idx}: {it['label']}{marker}")

                if st.button("⬇️ Exportar tudo (ZIP)"):
                    mem = create_export_zip(st.session_state.G, notes_text=st.session_state.get('notas', ''), include_png=True)
                    st.download_button("⬇️ Baixar ZIP (Exportar tudo)", data=mem, file_name="artemis_export_all.zip", mime="application/zip")

            with right:
                st.markdown("#### Ações rápidas")
                if st.button("🔁 Resetar grafo para planilha"):
                    st.session_state.G = criar_grafo(df)
                    init_action_stack(st.session_state.G, label="Reset por planilha")
                    st.session_state.undo_stack = []
                    st.success("Grafo resetado.")
                    st.experimental_rerun()
                if st.button("📥 Exportar JSON do grafo"):
                    data = json_graph.node_link_data(st.session_state.G)
                    b = io.BytesIO()
                    b.write(pd.io.json.dumps(data, indent=2).encode('utf-8'))
                    b.seek(0)
                    st.download_button("⬇️ Baixar JSON", data=b, file_name="grafo_artemis.json", mime="application/json")

    except Exception as e:
        st.error(f"Erro ao processar a planilha: {e}")
else:
    st.info("Carregue uma planilha (.csv ou .xlsx) para começar.")