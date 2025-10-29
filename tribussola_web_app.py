#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tribussula Web Application - Streamlit Version
Complete web application with all client requirements:
- Triangle interface for priority selection
- Full ranking list (all 12 solutions)
- Ranking plot (interactive chart)
- Static tree structure
- Static perspective view
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
import glob
import io
import json
import tempfile
import os
from typing import Optional

# Clustering helpers (provided by client)
try:
    from clustering_solutions import cluster_indices_gmm
    CLUSTERING_AVAILABLE = True
except Exception:
    CLUSTERING_AVAILABLE = False

# Optional imports for PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    st.warning("⚠️ PDF generation not available. Install reportlab for full functionality.")

# Page configuration
st.set_page_config(
    page_title="Tribussula Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    .triangle-container {
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4, #45b7d1);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
    }
    
    .ranking-card {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #e9ecef;
    }
    
    .gold { color: #FFD700; font-weight: bold; }
    .silver { color: #C0C0C0; font-weight: bold; }
    .bronze { color: #CD7F32; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Configuration
CANON_VAL = ["ZCusto", "ZQualidade", "ZPrazo"]
CANON_SIG = ["sZCusto", "sZQualidade", "sZPrazo"]

class TribussolaWebApp:
    """Web application for Tribussula decision support"""
    
    def __init__(self):
        self.v_data = None
        self.s_data = None
        self.nomes_data = None
        self.solution_descriptions = None
        self.cov_data = None  # holds covariance columns if available
        self.load_data()
        self.load_solution_descriptions()
    
    def load_data(self):
        """Load CSV data files with proper encoding handling"""
        try:
            # Find CSV files
            zscores_file = self.find_file("*Zscores*", "*decisao*")
            nomes_file = self.find_file("*nomes*", "*coordenadas*")
            
            if not zscores_file or not nomes_file:
                st.error("❌ Could not find required CSV files!")
                st.info("Please ensure you have files with 'Zscores' and 'nomes' in their names.")
                return False
            
            # Load Z-scores data (may already include covariance columns)
            df = self._read_csv_smart(zscores_file)
            
            # Map columns (case-insensitive)
            cols_lower = {c.lower(): c for c in df.columns}
            has_named = all((name.lower() in cols_lower) for name in CANON_VAL + CANON_SIG)
            
            if has_named:
                self.v_data = df[[cols_lower[n.lower()] for n in CANON_VAL]].copy()
                self.s_data = df[[cols_lower[n.lower()] for n in CANON_SIG]].copy()
                self.v_data.columns = CANON_VAL
                self.s_data.columns = CANON_SIG
                # Try to capture covariance columns by name patterns
                cov_cols = [c for c in df.columns if 'cov' in str(c).lower()]
                if len(cov_cols) >= 3:
                    # Heuristic ordering consistent with file headers
                    # Expect something like cov(Zcusto,Zqual), cov(Zcusto,Zprazo), cov(Zqual,Zprazo)
                    def _cov_sort_key(c):
                        cl = str(c).lower()
                        return (0 if 'custo' in cl and 'qual' in cl else
                                1 if 'custo' in cl and 'prazo' in cl else
                                2)
                    cov_cols_sorted = sorted(cov_cols[:3], key=_cov_sort_key)
                    self.cov_data = df[cov_cols_sorted].copy()
                    self.cov_data.columns = [
                        'cov_custo_qual', 'cov_custo_prazo', 'cov_qual_prazo'
                    ]
            else:
                # Positional mode
                if df.shape[1] < 6:
                    raise ValueError("CSV needs at least 6 columns")
                val_cols = [0, 2, 4]
                sig_cols = [1, 3, 5]
                self.v_data = df.iloc[:, val_cols].copy()
                self.s_data = df.iloc[:, sig_cols].copy()
                self.v_data.columns = CANON_VAL
                self.s_data.columns = CANON_SIG
                # If covariance columns are present, capture them (expected at cols 6,7,8)
                if df.shape[1] >= 9:
                    self.cov_data = df.iloc[:, [6, 7, 8]].copy()
                    self.cov_data.columns = [
                        'cov_custo_qual', 'cov_custo_prazo', 'cov_qual_prazo'
                    ]
            
            # Coerce to numeric
            for col in CANON_VAL:
                self.v_data[col] = self._coerce_num(self.v_data[col])
            for col in CANON_SIG:
                self.s_data[col] = self._coerce_num(self.s_data[col])
            
            # Load names
            self.nomes_data = self._read_csv_smart(nomes_file)
            
            # Fix single column issue by splitting on comma
            if self.nomes_data.shape[1] == 1:
                col_name = self.nomes_data.columns[0]
                if ',' in col_name:
                    # Split the single column into multiple columns
                    split_data = self.nomes_data[col_name].str.split(',', expand=True)
                    split_data.columns = ['coordenadas na árvore', 'nome']
                    self.nomes_data = split_data
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return False
    
    def find_file(self, *patterns):
        """Find file matching any of the patterns"""
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
        return None
    
    def _coerce_num(self, s: pd.Series) -> pd.Series:
        """Convert series with BR formats to float"""
        if pd.api.types.is_numeric_dtype(s):
            return s.astype(float)
        s2 = (s.astype(str)
              .str.strip()
              .str.replace("\u00a0", " ")
              .str.replace(".", "", regex=False)
              .str.replace(",", ".", regex=False))
        return pd.to_numeric(s2, errors="coerce")
    
    def _read_csv_smart(self, path: str) -> pd.DataFrame:
        """Smart CSV reading with multiple encoding attempts"""
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                df = pd.read_csv(path, encoding=encoding)
                if not df.empty and df.shape[1] > 1:  # Check if we got multiple columns
                    break
            except:
                continue
        else:
            df = pd.read_csv(path)
        
        # If we still have only one column, try different separators
        if df.shape[1] == 1:
            for sep in [';', ',', '\t']:
                try:
                    df_test = pd.read_csv(path, sep=sep, encoding='utf-8')
                    if df_test.shape[1] > 1:
                        df = df_test
                        break
                except:
                    continue
        
        # Try BR format if needed
        if (df.select_dtypes(include="number").shape[1] == 0) or \
           (df.dtypes.value_counts().get("object", 0) >= max(1, df.shape[1] - 1)):
            try:
                df = pd.read_csv(path, sep=";", decimal=",")
            except:
                pass
        return df
    
    def compute_ranking(self, r: float, g: float, b: float) -> pd.DataFrame:
        """Compute Z-ranking and corrected uncertainty based on RGB input.
        r,g,b are pure numbers in [0,1] and sum to 1.
        """
        # Linear ranking: cost (-), quality (+), deadline (-)
        zrank = (-r) * self.v_data["ZCusto"] + (g) * self.v_data["ZQualidade"] + (-b) * self.v_data["ZPrazo"]

        # Corrected uncertainty using client's formula
        sC = self.s_data["sZCusto"]
        sQ = self.s_data["sZQualidade"]
        sP = self.s_data["sZPrazo"]

        s0_sq = (1.0 / 9.0) * ( (r ** 2) * (sC ** 2) + (g ** 2) * (sQ ** 2) + (b ** 2) * (sP ** 2) )

        if self.cov_data is not None:
            c_cq = self.cov_data['cov_custo_qual']
            c_cp = self.cov_data['cov_custo_prazo']
            c_qp = self.cov_data['cov_qual_prazo']
            cov_term = (2.0 / 9.0) * ( (r * g * (c_cq ** 2)) - (r * b * (c_cp ** 2)) + (2.0 * g * b * (c_qp ** 2)) )
            s_z_sq = np.abs(s0_sq - cov_term)
        else:
            # Fallback to previous simple propagation if covariances not available
            s_z_sq = ( ( (-r) * sC ) ** 2 + ( (g) * sQ ) ** 2 + ( (-b) * sP ) ** 2 )

        s_zrank = np.sqrt(s_z_sq)

        out = pd.DataFrame({"Zranking": zrank, "s_Zranking": s_zrank})
        return out
    
    def get_full_ranking(self, ranking: pd.DataFrame) -> pd.DataFrame:
        """Get complete ranking with names and details"""
        # Join with names
        joined = ranking.join(self.nomes_data, how="left")
        
        # Add rank column with ties (same score => same rank)
        joined = (joined
                 .reset_index()
                 .rename(columns={"index": "idx"})
                 .sort_values("Zranking", ascending=False))
        joined["Rank"] = joined["Zranking"].rank(method='min', ascending=False).astype(int)
        joined = joined.reset_index(drop=True)
        
        # Get solution name
        name_col = next((c for c in self.nomes_data.columns 
                       if c.lower() in ["nome", "name", "titulo"]), None)
        
        if name_col:
            joined["Solution"] = joined[name_col]
        else:
            joined["Solution"] = [f"Solution {i+1}" for i in range(len(joined))]
        
        # Get coordinates
        coord_col = next((c for c in self.nomes_data.columns 
                         if "coordenadas" in c.lower() or "coordinates" in c.lower()), None)
        
        if coord_col:
            joined["Coordinates"] = joined[coord_col]
        else:
            joined["Coordinates"] = [f"Pos {i+1}" for i in range(len(joined))]
        
        return joined[["Rank", "Solution", "Zranking", "s_Zranking", "Coordinates"]]
    
    def load_solution_descriptions(self):
        """Load solution descriptions from JSON file"""
        try:
            with open('solution_descriptions.json', 'r', encoding='utf-8') as f:
                self.solution_descriptions = json.load(f)
        except FileNotFoundError:
            st.warning("⚠️ Solution descriptions file not found. Some features may not work properly.")
            self.solution_descriptions = None
        except Exception as e:
            st.error(f"❌ Error loading solution descriptions: {e}")
            self.solution_descriptions = None

def create_triangle_interface():
    """Create interactive triangle interface using Streamlit"""
    st.markdown('<div class="triangle-container">', unsafe_allow_html=True)
    
    st.markdown("### 🎯 Priority Selection")
    st.markdown("Adjust the sliders below to set your priorities for Cost, Quality, and Deadline:")
    
    # Create three columns for sliders
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔴 Cost (Custo)")
        cost_weight = st.slider("Cost Priority", 0.0, 100.0, 33.33, 0.1, key="cost_slider")
    
    with col2:
        st.markdown("#### 🟢 Quality (Qualidade)")
        quality_weight = st.slider("Quality Priority", 0.0, 100.0, 33.33, 0.1, key="quality_slider")
    
    with col3:
        st.markdown("#### 🔵 Deadline (Prazo)")
        deadline_weight = st.slider("Deadline Priority", 0.0, 100.0, 33.33, 0.1, key="deadline_slider")
    
    # Normalize weights
    total = cost_weight + quality_weight + deadline_weight
    if total > 0:
        r = cost_weight / total
        g = quality_weight / total
        b = deadline_weight / total
    else:
        r = g = b = 1/3
    
    # Display current weights
    st.markdown("### Current Weights:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cost", f"{r*100:.1f}%", delta=None)
    with col2:
        st.metric("Quality", f"{g*100:.1f}%", delta=None)
    with col3:
        st.metric("Deadline", f"{b*100:.1f}%", delta=None)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return r, g, b

def create_ranking_plot(ranking_df):
    """Create interactive ranking plot"""
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Decide coloring: clusters if available, else by score
    if 'cluster' in ranking_df.columns:
        clusters = ranking_df['cluster'].astype(int).fillna(-1)
        unique_clusters = sorted(clusters.unique())
        # Map clusters to colors
        palette = [
            '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
            '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
        ]
        cluster_to_color = {c: palette[i % len(palette)] for i, c in enumerate(unique_clusters)}
        bar_colors = clusters.map(cluster_to_color)
        hover_text = [f"Rank {row['Rank']} • Cluster {int(row['cluster'])}"
                      if not pd.isna(row.get('cluster')) else f"Rank {row['Rank']}"
                      for _, row in ranking_df.iterrows()]
        fig.add_trace(go.Bar(
            y=ranking_df['Solution'],
            x=ranking_df['Zranking'],
            error_x=dict(type='data', array=ranking_df['s_Zranking']),
            orientation='h',
            marker=dict(color=bar_colors),
            text=hover_text,
            textposition='outside',
            hovertemplate='%{y}<br>Z=%{x:.3f} ± %{customdata:.3f}<br>%{text}<extra></extra>',
            customdata=ranking_df['s_Zranking']
        ))
        # Add legend by dummy traces
        for c in unique_clusters:
            fig.add_trace(go.Bar(
                x=[None], y=[None], name=f"Cluster {int(c)}",
                marker=dict(color=cluster_to_color[c])
            ))
    else:
        # Add bars with error bars colored by value
        fig.add_trace(go.Bar(
            y=ranking_df['Solution'],
            x=ranking_df['Zranking'],
            error_x=dict(type='data', array=ranking_df['s_Zranking']),
            orientation='h',
            marker=dict(
                color=ranking_df['Zranking'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Z-Ranking Score")
            ),
            text=[f"Rank {row['Rank']}" for _, row in ranking_df.iterrows()],
            textposition='outside'
        ))
    
    fig.update_layout(
        title="Solution Rankings - Z-Score Analysis",
        xaxis_title="Z-Ranking Score",
        yaxis_title="Solutions",
        height=600,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_grade_plot(ranking_df):
    """Create 0–10 relative grade plot above original Zranking plot"""
    df = ranking_df.copy()
    z = df['Zranking']
    if len(z) > 0 and (z.max() - z.min()) > 0:
        grade = 10.0 * (z - z.min()) / (z.max() - z.min())
    else:
        grade = np.full_like(z, 5.0)
    df['Grade0_10'] = grade

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df['Solution'],
        x=df['Grade0_10'],
        orientation='h',
        marker=dict(color=df['Grade0_10'], colorscale='Bluered', showscale=True,
                    colorbar=dict(title="Grade (0–10)")),
        text=[f"{g:.1f}" for g in df['Grade0_10']],
        textposition='outside'
    ))
    fig.update_layout(
        title="Relative Grade (0–10)",
        xaxis_title="Grade (0–10)",
        yaxis_title="Solutions",
        height=500,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig

def create_visual_tree_diagram(app):
    """Create interactive visual tree diagram using Plotly"""
    if not app.solution_descriptions or 'itens' not in app.solution_descriptions:
        return None
    
    items = app.solution_descriptions['itens']
    
    # Build tree structure matching the client's image
    # Use a better layout: top to bottom, left to right
    
    tree_nodes = {}
    
    # Level 0: Root
    tree_nodes['Seu Problema'] = {
        'type': 'root', 'level': 0, 'parent': None,
        'pos': (100, 90)
    }
    
    # Level 1: Main branches
    tree_nodes['Soluções Prontas'] = {
        'type': 'branch', 'level': 1, 'parent': 'Seu Problema', 'tronco': 'I',
        'pos': (20, 60)
    }
    tree_nodes['IA por API'] = {
        'type': 'branch', 'level': 1, 'parent': 'Seu Problema', 'tronco': 'II',
        'pos': (60, 60)
    }
    tree_nodes['IA Própria'] = {
        'type': 'branch', 'level': 1, 'parent': 'Seu Problema', 'tronco': 'III',
        'pos': (100, 60)
    }
    
    # Level 2: Sub-branches
    tree_nodes['Sem Anonimização'] = {
        'type': 'sub', 'level': 2, 'parent': 'IA por API', 'tronco': 'II',
        'pos': (50, 35)
    }
    tree_nodes['Com Anonimização'] = {
        'type': 'sub', 'level': 2, 'parent': 'IA por API', 'tronco': 'II',
        'pos': (70, 35)
    }
    
    tree_nodes['PLIM 2.0 em DMZ'] = {
        'type': 'sub', 'level': 2, 'parent': 'IA Própria', 'tronco': 'III',
        'pos': (90, 35)
    }
    tree_nodes['Com Engenharia Reversa'] = {
        'type': 'sub', 'level': 2, 'parent': 'IA Própria', 'tronco': 'III',
        'pos': (110, 35)
    }
    
    # Group solutions by their parent branch
    solution_groups = {
        'Soluções Prontas': [],
        'Sem Anonimização': [],
        'Com Anonimização': [],
        'PLIM 2.0 em DMZ': [],
        'Com Engenharia Reversa': []
    }
    
    for item in items:
        item_id = item['id']
        item_name = item['nome']
        tronco = item['tronco']
        
        if tronco == 'I':
            solution_groups['Soluções Prontas'].append((item_id, item_name))
        elif tronco == 'II':
            if item_id.startswith('II.1'):
                solution_groups['Sem Anonimização'].append((item_id, item_name))
            elif item_id.startswith('II.2'):
                solution_groups['Com Anonimização'].append((item_id, item_name))
        elif tronco == 'III':
            if item_id == 'III.1.b':
                solution_groups['PLIM 2.0 em DMZ'].append((item_id, item_name))
            else:
                solution_groups['Com Engenharia Reversa'].append((item_id, item_name))
    
    # Add solution leaves with proper positioning
    for parent_name, solutions in solution_groups.items():
        parent_x = tree_nodes[parent_name]['pos'][0]
        total = len(solutions)
        
        for idx, (item_id, item_name) in enumerate(solutions):
            # Spread solutions horizontally under their parent
            if total == 1:
                x_pos = parent_x
            else:
                spread = min(30, total * 8)  # Maximum spread
                x_pos = parent_x + (idx - (total - 1) / 2) * (spread / max(1, total - 1))
            
            tree_nodes[item_name] = {
                'type': 'leaf',
                'level': 3,
                'parent': parent_name,
                'id': item_id,
                'tronco': tronco,
                'pos': (x_pos, 10)
            }
    
    # Create figure with dark background
    fig = go.Figure()
    
    # Add edges (connections)
    for node_name, node_data in tree_nodes.items():
        if node_data['parent']:
            parent_data = tree_nodes[node_data['parent']]
            x0, y0 = parent_data['pos']
            x1, y1 = node_data['pos']
            
            # Determine color and width
            if node_data['type'] == 'branch':
                color = '#3498db'
                width = 3
            elif node_data['type'] == 'sub':
                color = '#16a085'
                width = 2.5
            else:  # leaf
                color = '#e74c3c'
                width = 2
            
            # Create curved line for better visual
            x_curve = [x0, (x0 + x1) / 2, (x0 + x1) / 2, x1]
            y_curve = [y0, y0 + abs(y0 - y1) * 0.3, y1 + abs(y0 - y1) * 0.3, y1]
            
            fig.add_trace(go.Scatter(
                x=x_curve,
                y=y_curve,
                mode='lines',
                line=dict(color=color, width=width),
                hoverinfo='none',
                showlegend=False
            ))
    
    # Add nodes
    for node_name, node_data in tree_nodes.items():
        x, y = node_data['pos']
        
        # Node styling
        if node_data['type'] == 'root':
            color = '#e74c3c'
            size = 35
            textcolor = 'white'
            textsize = 14
        elif node_data['type'] == 'branch':
            color = '#3498db'
            size = 25
            textcolor = 'white'
            textsize = 12
        elif node_data['type'] == 'sub':
            color = '#16a085'
            size = 20
            textcolor = 'white'
            textsize = 10
        else:  # leaf
            color = '#95a5a6'
            size = 15
            textcolor = 'white'
            textsize = 8
        
        custom_data = [node_data.get('id', '')] if node_data.get('id') else None
        hover_text = f"<b>{node_name}</b>"
        if node_data.get('id'):
            hover_text += f"<br>ID: {node_data.get('id')}"
        hover_text += "<extra></extra>"
        
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(
                size=size,
                color=color,
                line=dict(width=3, color='white')
            ),
            text=node_name,
            textposition="middle center",
            textfont=dict(size=textsize, color=textcolor),
            hovertemplate=hover_text,
            customdata=custom_data,
            name=node_data.get('id', node_name),
            showlegend=False
        ))
    
    fig.update_layout(
        title={
            'text': '🌳 Your Problem → Solution Decision Tree',
            'font': {'size': 20, 'color': 'white'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 130]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 100]),
        plot_bgcolor='#2c3e50',
        paper_bgcolor='#2c3e50',
        height=700,
        showlegend=False,
        margin=dict(l=10, r=10, t=60, b=10)
    )
    
    return fig

def create_interactive_tree(app):
    """Create interactive hierarchical tree structure with clickable nodes"""
    st.markdown("### 🌳 Interactive Solution Tree")
    st.markdown("**Use the tree diagram and clickable solutions below to explore:**")
    
    if not app.solution_descriptions:
        st.error("❌ Solution descriptions not available. Please ensure solution_descriptions.json is present.")
        return
    
    # Create visual tree diagram
    tree_fig = create_visual_tree_diagram(app)
    
    if tree_fig:
        st.plotly_chart(tree_fig, use_container_width=True)
    
    # Create solution list with clickable buttons
    st.markdown("---")
    st.markdown("### 📝 Solutions (Click to View Details)")
    
    if 'itens' in app.solution_descriptions:
        items = app.solution_descriptions['itens']
        
        # Organize by Tronco
        troncos = {}
        for item in items:
            tronco = item.get('tronco', 'Unknown')
            if tronco not in troncos:
                troncos[tronco] = []
            troncos[tronco].append(item)
        
        # Create expandable sections for each Tronco
        for tronco in sorted(troncos.keys()):
            tronco_name = f"Tronco {tronco}"
            if tronco == 'I':
                tronco_name = "🔷 Soluções Prontas"
            elif tronco == 'II':
                tronco_name = "🔶 IA por API"
            elif tronco == 'III':
                tronco_name = "🔸 IA Própria"
            
            with st.expander(tronco_name):
                # Create button grid
                for item in sorted(troncos[tronco], key=lambda x: x.get('id', '')):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        button_label = f"🌿 {item['id']}: {item['nome']}"
                    with col2:
                        if st.button("View Details", key=f"btn_{item['id']}"):
                            st.session_state.selected_solution = item['id']
    
    # Display selected solution details
    if 'selected_solution' in st.session_state:
        coord = st.session_state.selected_solution
        
        # Check new structure first (itens), then fallback to old structure (solutions)
        if 'itens' in app.solution_descriptions:
            # New structure
            items = app.solution_descriptions['itens']
            solution = next((item for item in items if item['id'] == coord), None)
            
            if solution:
                st.markdown("---")
                st.markdown("### 📋 Solution Details")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{solution['nome']}**")
                    st.markdown(f"*ID: {solution['id']} | Tronco: {solution['tronco']}*")
                with col2:
                    if st.button("❌ Close Details", key=f"close_{coord}"):
                        del st.session_state.selected_solution
                
                st.markdown("#### Descrição")
                st.write(solution['descricao'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 📊 Custos e Prazos")
                    st.metric("CAPEX (BRL)", f"R$ {solution['capex_brl']:,.0f}")
                    st.metric("OPEX Mensal (BRL)", f"R$ {solution['opex_mensal_brl']:,.0f}")
                    st.metric("Prazo Total (dias)", f"{solution['prazos_dias']['total']} ± {solution['prazos_dias']['sigma']}")
                    
                    st.markdown("#### ⚖️ Qualidade")
                    qual = solution['qualidade_objetiva']
                    st.write(f"**Dedução Top3:** {qual['deduplicacao_top3_pct']}")
                    st.write(f"**Latência:** {qual['latencia_seg']}s")
                    st.write(f"**Cobertura:** {qual['cobertura_classificacao_pct']}")
                
                with col2:
                    st.markdown("#### ✅ Benefícios")
                    for benef in solution['beneficios']:
                        st.markdown(f"• {benef}")
                    
                    st.markdown("#### ⚠️ Limitações")
                    for limit in solution['limitacoes']:
                        st.markdown(f"• {limit}")
                
                st.markdown("#### 🎯 Quando Escolher")
                st.info(solution['quando_escolher'])
                
                if 'extensoes_futuras' in solution and solution['extensoes_futuras']:
                    st.markdown("#### 🔮 Extensões Futuras")
                    for ext in solution['extensoes_futuras']:
                        st.markdown(f"• {ext}")
        
        elif 'solutions' in app.solution_descriptions:
            # Old structure (backward compatibility)
            if coord in app.solution_descriptions['solutions']:
                solution = app.solution_descriptions['solutions'][coord]
                
                st.markdown("---")
                st.markdown("### 📋 Solution Details")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{solution['name']}**")
                    st.markdown(f"*Coordinates: {solution['coordinates']}*")
                with col2:
                    if st.button("❌ Close Details", key=f"close_{coord}"):
                        del st.session_state.selected_solution
                
                st.markdown("#### Description")
                st.write(solution['description'])
                
                st.markdown("#### Technical Details")
                tech_details = solution['technical_details']
                for key, value in tech_details.items():
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                
                st.markdown("#### Pros")
                for pro in solution['pros']:
                    st.markdown(f"✅ {pro}")
                
                st.markdown("#### Cons")
                for con in solution['cons']:
                    st.markdown(f"❌ {con}")
                
                st.markdown("#### Use Cases")
                for use_case in solution['use_cases']:
                    st.markdown(f"🎯 {use_case}")

def create_static_perspective():
    """Create static perspective analysis view"""
    st.markdown("### 🔍 Analysis Perspective")
    
    # Create perspective cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Statistical Analysis</h4>
            <p><strong>Method:</strong> Z-Score Normalization</p>
            <p><strong>Error Propagation:</strong> Gaussian Error Analysis</p>
            <p><strong>Confidence Level:</strong> 95% (2σ)</p>
            <p><strong>Sample Size:</strong> 12 Solutions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Decision Criteria</h4>
            <p><strong>Cost:</strong> Lower is better (negative weight)</p>
            <p><strong>Quality:</strong> Higher is better (positive weight)</p>
            <p><strong>Deadline:</strong> Lower is better (negative weight)</p>
            <p><strong>Normalization:</strong> Weights sum to 100%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Analysis methodology
    st.markdown("""
    <div class="metric-card">
        <h4>📈 Methodology</h4>
        <p><strong>Z-Ranking Formula:</strong> Z = (-Cost × ZCusto) + (Quality × ZQualidade) + (-Deadline × ZPrazo)</p>
        <p><strong>Error Calculation:</strong> σZ = √[(-Cost × σCusto)² + (Quality × σQualidade)² + (-Deadline × σPrazo)²]</p>
        <p><strong>Ranking:</strong> Solutions ranked by Z-Ranking score (higher is better)</p>
        <p><strong>Confidence:</strong> Error bars represent ±1 standard deviation</p>
    </div>
    """, unsafe_allow_html=True)

def create_pdf_report(ranking_df, app, top_n=3):
    """Create PDF report for podium solutions with clickable elements"""
    if not REPORTLAB_AVAILABLE:
        st.error("❌ PDF generation requires reportlab. Please install it or use the web interface.")
        return None
    
    if not app.solution_descriptions:
        st.error("❌ Solution descriptions not available for PDF generation.")
        return None
    
    # Create temporary file for PDF
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_file.close()
    
    # Create PDF document
    doc = SimpleDocTemplate(temp_file.name, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    # Build PDF content
    story = []
    
    # Title
    story.append(Paragraph("Tribussula Podium Solutions Report", title_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(
        f"This report presents the top {top_n} solutions from the Tribussula decision analysis, "
        f"ranked based on weighted criteria including cost, quality, and deadline considerations. "
        f"Each solution includes detailed technical specifications, advantages, disadvantages, "
        f"and recommended use cases for informed decision-making.",
        styles['Normal']
    ))
    story.append(Spacer(1, 20))
    
    # Podium Solutions
    story.append(Paragraph("Podium Solutions", heading_style))
    
    for i, (_, row) in enumerate(ranking_df.head(top_n).iterrows()):
        # Solution header
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        story.append(Paragraph(f"{medal} Rank {row['Rank']}: {row['Solution']}", heading_style))
        
        # Get solution details
        coord = row['Coordinates']
        solution = None
        
        # Check new structure first (itens), then fallback to old structure (solutions)
        if 'itens' in app.solution_descriptions:
            items = app.solution_descriptions['itens']
            solution = next((item for item in items if item['id'] == coord), None)
        elif 'solutions' in app.solution_descriptions and coord in app.solution_descriptions['solutions']:
            solution = app.solution_descriptions['solutions'][coord]
        
        if solution:
            
            # Basic info table - handle both new and old structures
            if 'nome' in solution:
                # New structure
                info_data = [
                    ['Z-Ranking Score', f"{row['Zranking']:.3f} ± {row['s_Zranking']:.3f}"],
                    ['ID', coord],
                    ['Tronco', solution.get('tronco', 'N/A')],
                    ['CAPEX (BRL)', f"R$ {solution.get('capex_brl', 0):,.0f}"],
                    ['OPEX Mensal (BRL)', f"R$ {solution.get('opex_mensal_brl', 0):,.0f}"],
                    ['Prazo Total (dias)', f"{solution.get('prazos_dias', {}).get('total', 'N/A')} ± {solution.get('prazos_dias', {}).get('sigma', 'N/A')}"]
                ]
            else:
                # Old structure
                info_data = [
                    ['Z-Ranking Score', f"{row['Zranking']:.3f} ± {row['s_Zranking']:.3f}"],
                    ['Coordinates', coord],
                    ['Platform', solution['technical_details']['platform']],
                    ['Deployment', solution['technical_details']['deployment']],
                    ['Development Approach', solution['technical_details']['development_approach']]
                ]
            
            info_table = Table(info_data, colWidths=[2*inch, 4*inch])
            info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (1, 0), (1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(info_table)
            story.append(Spacer(1, 12))
            
            # Description
            story.append(Paragraph("Description", styles['Heading3']))
            desc_text = solution.get('descricao') or solution.get('description', 'No description available.')
            story.append(Paragraph(desc_text, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Pros and Cons in two columns - handle both structures
            if 'beneficios' in solution:
                # New structure
                col1_data = [['Benefits']] + [[f"• {ben}"] for ben in solution.get('beneficios', [])]
                col2_data = [['Limitations']] + [[f"• {lim}"] for lim in solution.get('limitacoes', [])]
            else:
                # Old structure
                col1_data = [['Advantages']] + [[f"• {pro}"] for pro in solution.get('pros', [])]
                col2_data = [['Disadvantages']] + [[f"• {con}"] for con in solution.get('cons', [])]
            
            pros_table = Table(col1_data, colWidths=[2.5*inch])
            cons_table = Table(col2_data, colWidths=[2.5*inch])
            
            table_style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ])
            
            pros_table.setStyle(table_style)
            cons_table.setStyle(table_style)
            
            # Create two-column layout
            story.append(Paragraph("Analysis", styles['Heading3']))
            story.append(Table([[pros_table, cons_table]], colWidths=[2.5*inch, 2.5*inch]))
            story.append(Spacer(1, 12))
            
            # Use Cases - handle both structures
            if 'extensoes_futuras' in solution:
                # New structure
                story.append(Paragraph("Extensions", styles['Heading3']))
                exts = solution.get('extensoes_futuras', [])
                if exts:
                    exts_text = "• " + "\\n• ".join(exts)
                    story.append(Paragraph(exts_text, styles['Normal']))
                
                # Add "quando escolher" if available
                if 'quando_escolher' in solution:
                    story.append(Paragraph("When to Choose", styles['Heading3']))
                    story.append(Paragraph(solution['quando_escolher'], styles['Normal']))
                story.append(Spacer(1, 20))
            elif 'use_cases' in solution:
                # Old structure
                story.append(Paragraph("Recommended Use Cases", styles['Heading3']))
                use_cases_text = "• " + "\\n• ".join(solution['use_cases'])
                story.append(Paragraph(use_cases_text, styles['Normal']))
                story.append(Spacer(1, 20))
        
        else:
            story.append(Paragraph(f"Details not available for {coord}", styles['Normal']))
            story.append(Spacer(1, 20))
    
    # Methodology section
    story.append(PageBreak())
    story.append(Paragraph("Analysis Methodology", heading_style))
    story.append(Paragraph(
        "The ranking is based on Z-score normalization across three key criteria:",
        styles['Normal']
    ))
    
    methodology_data = [
        ['Criterion', 'Weight Direction', 'Description'],
        ['Cost (Custo)', 'Negative (Lower is Better)', 'Financial investment required'],
        ['Quality (Qualidade)', 'Positive (Higher is Better)', 'Solution quality and capabilities'],
        ['Deadline (Prazo)', 'Negative (Lower is Better)', 'Time to implementation']
    ]
    
    method_table = Table(methodology_data, colWidths=[1.5*inch, 2*inch, 2.5*inch])
    method_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(method_table)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph(
        "Z-Ranking Formula: Z = (-Cost × ZCusto) + (Quality × ZQualidade) + (-Deadline × ZPrazo)",
        styles['Normal']
    ))
    story.append(Paragraph(
        "Error Propagation: σZ = √[(-Cost × σCusto)² + (Quality × σQualidade)² + (-Deadline × σPrazo)²]",
        styles['Normal']
    ))
    
    # Build PDF
    doc.build(story)
    
    return temp_file.name

def create_pdf_report_section(ranking_df, app):
    """Create PDF report section in the web app"""
    st.markdown("### 📄 PDF Report Generator")
    
    if 'ranking' not in st.session_state:
        st.info("👆 Please run the analysis first to generate the PDF report.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        top_n = st.selectbox("Number of top solutions to include:", [3, 5, 10, 12], index=0)
    
    with col2:
        if st.button("📄 Generate PDF Report", type="primary"):
            with st.spinner("Generating PDF report..."):
                pdf_path = create_pdf_report(ranking_df, app, top_n)
                
                if pdf_path:
                    # Read the generated PDF
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_data = pdf_file.read()
                    
                    # Provide download button
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_data,
                        file_name=f"tribussula_podium_report_top_{top_n}.pdf",
                        mime="application/pdf"
                    )
                    
                    st.success(f"✅ PDF report generated successfully with top {top_n} solutions!")
                    
                    # Clean up temporary file
                    os.unlink(pdf_path)
                else:
                    st.error("❌ Failed to generate PDF report.")

def main():
    """Main application function"""
    # Header
    st.markdown('<div class="main-header">🎯 Tribussula Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Interactive Decision Support System for Solution Selection**")
    
    # Initialize app
    app = TribussolaWebApp()
    
    if not app.v_data is not None:
        st.error("❌ Failed to load data. Please check your CSV files.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Priority Selection", 
        "📊 Full Ranking", 
        "📈 Ranking Plot", 
        "🌳 Interactive Tree", 
        "📄 PDF Report",
        "🔍 Perspective"
    ])
    
    with tab1:
        st.markdown('<div class="section-header">Priority Selection Interface</div>', unsafe_allow_html=True)
        r, g, b = create_triangle_interface()
        
        # Compute ranking
        if st.button("🔄 Analyze Solutions", type="primary"):
            ranking = app.compute_ranking(r, g, b)
            full_ranking = app.get_full_ranking(ranking)
            # Clustering on [Zranking, s_Zranking]
            if CLUSTERING_AVAILABLE:
                out = full_ranking[["Zranking", "s_Zranking"]].copy()
                try:
                    labels, cluster_meta, probabilities = cluster_indices_gmm(out, use_uncertainty=True)
                except Exception as e:
                    labels, cluster_meta, probabilities = (None, None, None)
                if labels is not None:
                    full_ranking = full_ranking.copy()
                    full_ranking["cluster"] = labels
                    if probabilities is not None:
                        full_ranking["cluster_probability"] = np.max(probabilities, axis=1)
                    # Persist metadata
                    try:
                        with open("gmm_clusters_meta.json", "w", encoding="utf-8") as f:
                            json.dump(cluster_meta, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
            
            # Store in session state
            st.session_state.ranking = full_ranking
            st.session_state.weights = (r, g, b)
            
            st.success("✅ Analysis complete! Check the other tabs for results.")
    
    with tab2:
        st.markdown('<div class="section-header">Full Ranking List</div>', unsafe_allow_html=True)
        
        if 'ranking' in st.session_state:
            ranking_df = st.session_state.ranking
            
            # Display podium with ties supported
            st.markdown("### 🏆 Podium (with ties)")
            podium = ranking_df[ranking_df['Rank'] <= 3]
            for rank in sorted(podium['Rank'].unique()):
                group = podium[podium['Rank'] == rank]
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
                color_class = "gold" if rank == 1 else "silver" if rank == 2 else "bronze"
                names = ", ".join(group['Solution'].astype(str).tolist())
                zvals = "; ".join([f"{z:.3f} ± {s:.3f}" for z, s in zip(group['Zranking'], group['s_Zranking'])])
                st.markdown(f"""
                <div class="ranking-card">
                    <h4 class="{color_class}">{medal} Rank {rank}: {names}</h4>
                    <p><strong>Z-Ranking(s):</strong> {zvals}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display complete table
            st.markdown("### 📋 Complete Ranking Table")
            st.dataframe(
                ranking_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Export options
            st.markdown("### 📥 Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = ranking_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv,
                    file_name="tribussula_ranking.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create Excel file
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    ranking_df.to_excel(writer, sheet_name='Ranking', index=False)
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📈 Download Excel",
                    data=excel_data,
                    file_name="tribussula_ranking.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        else:
            st.info("👆 Please go to the Priority Selection tab and click 'Analyze Solutions' first.")
    
    with tab3:
        st.markdown('<div class="section-header">Interactive Ranking Plot</div>', unsafe_allow_html=True)
        
        if 'ranking' in st.session_state:
            ranking_df = st.session_state.ranking
            # Grade plot (0–10) above
            grade_fig = create_grade_plot(ranking_df)
            st.plotly_chart(grade_fig, use_container_width=True)
            # Original Zranking plot below
            fig = create_ranking_plot(ranking_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional statistics
            st.markdown("### 📊 Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Best Score", f"{ranking_df['Zranking'].max():.3f}")
            with col2:
                st.metric("Worst Score", f"{ranking_df['Zranking'].min():.3f}")
            with col3:
                st.metric("Average Score", f"{ranking_df['Zranking'].mean():.3f}")
            with col4:
                st.metric("Std Deviation", f"{ranking_df['Zranking'].std():.3f}")
        
        else:
            st.info("👆 Please go to the Priority Selection tab and click 'Analyze Solutions' first.")
    
    with tab4:
        st.markdown('<div class="section-header">Interactive Solution Tree</div>', unsafe_allow_html=True)
        create_interactive_tree(app)
    
    with tab5:
        st.markdown('<div class="section-header">PDF Report Generator</div>', unsafe_allow_html=True)
        if 'ranking' in st.session_state:
            create_pdf_report_section(st.session_state.ranking, app)
        else:
            st.info("👆 Please run the analysis first to generate PDF reports.")
    
    with tab6:
        st.markdown('<div class="section-header">Static Perspective Analysis</div>', unsafe_allow_html=True)
        create_static_perspective()
    
    # Footer
    st.markdown("---")
    st.markdown("**Tribussula Dashboard** - Interactive Decision Support System")
    st.markdown("*Built with Streamlit | Powered by Python*")

if __name__ == "__main__":
    main()
