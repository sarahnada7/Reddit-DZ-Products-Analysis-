import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from collections import Counter
import re
from pathlib import Path

# ─── Load & Prepare Data ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "data" / "cleaned" / "product_reviews_cleaned.csv")
df['date']       = pd.to_datetime(df['date'])
df['month_name'] = df['date'].dt.strftime('%b')
df['month_num']  = df['date'].dt.month
df['day_name']   = df['date'].dt.day_name()
df['engagement'] = df['score'] + df['comments']
knn_df = pd.read_csv(BASE_DIR / "outputs" / "reports" / "knn_accuracy_by_k.csv")

STOPWORDS = {
    'i','my','me','if','what','can','how','about','the','is','and','to','a',
    'in','it','for','of','that','this','be','have','has','was','are','we',
    'you','your','or','not','on','do','its','an','but','at','as','with',
    'they','their','any','all','also','so','very','just','more','by','our',
    'no','from','there','up','out','were','would','which','some','get','got','im',
    'don','doesn','really','know','like','use','one','want','good','best','new',
    'will','need','please','looking','anyone','think','something','make','used',
    'find','recommend','help','try','buy','price','product','algeria','algerian',
}
CATEGORIES = sorted(df['category'].unique())

# ─── Color System — Light Pink Editorial ────────────────────────────────────
C = {
    'bg':          '#FDE8EF',
    'bg2':         '#FAD4E0',
    'card':        'rgba(255,255,255,0.72)',
    'card_solid':  '#FFFFFF',
    'panel':       'rgba(255,255,255,0.55)',
    'border':      'rgba(220,100,140,0.18)',
    'border_med':  'rgba(220,100,140,0.30)',
    'ink':         '#2A1120',
    'ink_mid':     '#7A3A55',
    'ink_soft':    '#B07090',
    'grid':        'rgba(220,100,140,0.10)',
    'magenta':     '#E8186D',
    'coral':       '#FF5C5C',
    'tangerine':   '#FF8C42',
    'gold':        '#FFCA3A',
    'mint':        '#2DC653',
    'teal':        '#14B8A6',
    'violet':      '#7C3AED',
    'orchid':      '#D946EF',
    'sky':         '#0EA5E9',
    'rose':        '#FB7185',
}

CAT_COLORS = {
    'appliances':  '#E8186D',
    'food':        '#FF8C42',
    'electronics': '#14B8A6',
    'cosmetics':   '#D946EF',
}

# ─── Reddit icon (base64 data URI) ──────────────────────────────────────────
_REDDIT_B64 = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB2aWV3Qm94PSIwIDAgMjAgMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+"
    "PGNpcmNsZSBjeD0iMTAiIGN5PSIxMCIgcj0iMTAiIGZpbGw9IiNGRjQ1MDAiLz48cGF0aCBmaWxsPSJ3"
    "aGl0ZSIgZD0iTTE2LjY3IDEwYTEuNDYgMS40NiAwIDAwLTIuNDctMSA3LjEyIDcuMTIgMCAwMC0zLjg1"
    "LTEuMjNsLjY1LTMuMDggMi4xMy40NWExIDEgMCAxMDEuMDctMSAxIDEgMCAwMC0uOTYuNjhsLTIuMzgt"
    "LjVhLjE2LjE2IDAgMDAtLjE5LjEybC0uNzMgMy40NGE3LjE0IDcuMTQgMCAwMC0zLjg5IDEuMjMgMS40"
    "NiAxLjQ2IDAgMTAtMS42MSAyLjM5IDIuODcgMi44NyAwIDAwMCAuNDRjMCAyLjI0IDIuNjEgNC4wNiA1"
    "LjgzIDQuMDZzNS44My0xLjgyIDUuODMtNC4wNmEyLjg3IDIuODcgMCAwMDAtLjQ0IDEuNDYgMS40NiAw"
    "IDAwLjQ3LTEuNXpNNy4yNyAxMWExIDEgMCAxMTEgMSAxIDEgMCAwMS0xLTF6bTUuNTggMi42NWEzLjQ3"
    "IDMuNDcgMCAwMS0yLjg1LjY2IDMuNDcgMy40NyAwIDAxLTIuODUtLjY2LjE5LjE5IDAgMDEuMjctLjI3"
    "IDMuMTIgMy4xMiAwIDAwMi41OC40OSAzLjEyIDMuMTIgMCAwMDIuNTgtLjQ5LjE5LjE5IDAgMDEuMjcu"
    "Mjd6bS0uMTctMS42NWExIDEgMCAxMTEtMSAxIDEgMCAwMS0xIDF6Ii8+PC9zdmc+"
)
def reddit_icon(size=22):
    return html.Img(src=_REDDIT_B64, style={'width': f'{size}px', 'height': f'{size}px'})

# ─── Plot theme ──────────────────────────────────────────────────────────────
def apply_theme(fig, title=""):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="'Playfair Display','Georgia',serif", color=C['ink'], size=12),
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=13, color=C['ink'], family="'DM Sans',sans-serif"),
            x=0.01, xref='paper', pad=dict(l=6, t=6),
        ),
        margin=dict(l=12, r=12, t=48, b=12),
        legend=dict(
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor=C['border_med'], borderwidth=1,
            font=dict(size=11, color=C['ink_mid'], family="'DM Sans',sans-serif"),
        ),
        xaxis=dict(
            gridcolor=C['grid'], gridwidth=1,
            zerolinecolor=C['border'],
            tickfont=dict(color=C['ink_mid'], size=11, family="'DM Sans',sans-serif"),
            linecolor=C['border_med'],
        ),
        yaxis=dict(
            gridcolor=C['grid'], gridwidth=1,
            zerolinecolor=C['border'],
            tickfont=dict(color=C['ink_mid'], size=11, family="'DM Sans',sans-serif"),
            linecolor=C['border_med'],
        ),
    )
    return fig

# ─── Chart functions ──────────────────────────────────────────────────────────
def fig_donut(filtered):
    counts = filtered['category'].value_counts().reset_index()
    counts.columns = ['category', 'count']
    colors = [CAT_COLORS.get(c, '#ccc') for c in counts['category']]
    fig = go.Figure(go.Pie(
        labels=counts['category'].str.capitalize(), values=counts['count'],
        hole=0.64,
        marker=dict(colors=colors, line=dict(color='white', width=3)),
        textinfo='label+percent',
        textfont=dict(size=12, color=C['ink'], family="'DM Sans',sans-serif"),
        pull=[0.05] * len(counts),
        hovertemplate="<b>%{label}</b><br>Posts: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{len(filtered)}</b><br><span style='font-size:10px;color:{C['ink_mid']}'>posts</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=22, color=C['ink']), align='center',
    )
    return apply_theme(fig, "Category Distribution")

def fig_score_bar(filtered):
    stats = filtered.groupby('category').agg(avg_score=('score','mean')).reset_index()
    stats = stats.sort_values('avg_score', ascending=True)
    colors = [CAT_COLORS.get(c,'#ccc') for c in stats['category']]
    fig = go.Figure(go.Bar(
        x=stats['avg_score'], y=stats['category'].str.capitalize(),
        orientation='h',
        marker=dict(color=colors, opacity=0.88, line=dict(color='rgba(0,0,0,0)')),
        text=[f"  {v:.1f}" for v in stats['avg_score']],
        textposition='outside', textfont=dict(size=13, color=C['ink']),
        hovertemplate="<b>%{y}</b><br>Avg Score: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(showlegend=False, bargap=0.38, xaxis_title="Average Score")
    return apply_theme(fig, "Average Score by Category")

def fig_timeline(filtered):
    monthly = (
        filtered.groupby(['month_num','month_name','category'])
        .size().reset_index(name='count').sort_values('month_num')
    )
    fig = px.line(monthly, x='month_name', y='count', color='category',
        color_discrete_map=CAT_COLORS, markers=True, line_shape='spline')
    fig.update_traces(line=dict(width=2.5), marker=dict(size=8, line=dict(width=2, color='white')))
    fig.update_layout(xaxis_title="Month", yaxis_title="Posts")
    return apply_theme(fig, "Monthly Post Activity")

def fig_scatter(filtered):
    f = filtered.copy()
    f['size_col'] = np.log1p(f['text_length']) * 4 + 6
    fig = px.scatter(f, x='score', y='comments', color='category',
        color_discrete_map=CAT_COLORS, size='size_col', size_max=22, opacity=0.78,
        hover_data={'title': True, 'score': True, 'comments': True, 'size_col': False})
    fig.update_traces(marker=dict(line=dict(width=1.5, color='white')))
    fig.update_xaxes(title="Score (Upvotes)", type='log')
    fig.update_yaxes(title="Comments")
    return apply_theme(fig, "Score vs Comments  (size = text length)")

def fig_keywords(filtered):
    all_text = ' '.join(filtered['clean_content'].dropna())
    words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
    freq = Counter(w for w in words if w not in STOPWORDS)
    top = pd.DataFrame(freq.most_common(12), columns=['word','freq'])
    spectrum = [
        C['magenta'], C['rose'],   C['tangerine'], C['gold'],
        C['mint'],    C['teal'],   C['sky'],        C['violet'],
        C['orchid'],  C['coral'],  C['magenta'],    C['teal'],
    ]
    colors = spectrum[:len(top)]
    fig = go.Figure(go.Bar(
        x=top['freq'], y=top['word'], orientation='h',
        marker=dict(color=colors[::-1], opacity=0.85, line=dict(color='rgba(0,0,0,0)')),
        text=[f"  {v}" for v in top['freq']], textposition='outside',
        textfont=dict(size=12, color=C['ink']),
        hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>",
    ))
    fig.update_layout(yaxis=dict(autorange='reversed'), xaxis_title="Frequency", showlegend=False)
    return apply_theme(fig, "Top Keywords in Posts")

def fig_violin(filtered):
    fig = go.Figure()
    for cat in CATEGORIES:
        sub = filtered[filtered['category'] == cat]['text_length']
        if len(sub) < 3: continue
        fig.add_trace(go.Violin(
            y=sub, name=cat.capitalize(),
            line_color=CAT_COLORS.get(cat,'#ccc'),
            fillcolor=CAT_COLORS.get(cat,'#ccc'),
            opacity=0.5, box_visible=True, meanline_visible=True, hoverinfo='name+y',
        ))
    fig.update_layout(yaxis_title="Characters", showlegend=False)
    return apply_theme(fig, "Post Length Distribution")

def fig_heatmap(filtered):
    DAY_ORDER = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    heat = filtered.groupby(['day_name','month_name']).size().reset_index(name='count')
    pivot = heat.pivot(index='day_name', columns='month_name', values='count').fillna(0)
    pivot = pivot.reindex([d for d in DAY_ORDER if d in pivot.index])
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[
            [0.0, '#FDE8EF'], [0.3, '#F9A8C9'],
            [0.65, '#E8186D'], [1.0, '#7C0040'],
        ],
        hovertemplate="<b>%{y}</b> · %{x}<br>Posts: %{z}<extra></extra>",
        showscale=True,
        colorbar=dict(
            tickfont=dict(color=C['ink_mid'], size=10),
            thickness=12, len=0.85,
            title=dict(text="Posts", font=dict(color=C['ink_mid'], size=10)),
        ),
    ))
    return apply_theme(fig, "Activity Heatmap  (Day × Month)")

def fig_top_posts(filtered):
    top = filtered.nlargest(8, 'score')
    colors = [CAT_COLORS.get(c,'#ccc') for c in top['category']]
    labels = [t[:48]+'...' if len(t) > 48 else t for t in top['title']]
    fig = go.Figure(go.Bar(
        x=top['score'], y=labels, orientation='h',
        marker=dict(color=colors, opacity=0.88, line=dict(color='rgba(0,0,0,0)')),
        text=[f"  {s}" for s in top['score']], textposition='outside',
        textfont=dict(size=12, color=C['ink']),
        customdata=top[['category','comments']].values,
        hovertemplate="<b>%{y}</b><br>Category: %{customdata[0]}<br>Score: %{x}<br>Comments: %{customdata[1]}<extra></extra>",
    ))
    fig.update_layout(yaxis=dict(autorange='reversed'), showlegend=False, xaxis_title="Score")
    return apply_theme(fig, "Top 8 Posts by Score")

def fig_knn():
    best_idx = knn_df['accuracy'].idxmax()
    best_k   = knn_df.loc[best_idx, 'K']
    best_acc = knn_df.loc[best_idx, 'accuracy']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=knn_df['K'], y=knn_df['accuracy'],
        mode='lines+markers',
        line=dict(color=C['magenta'], width=3),
        marker=dict(size=9, color=C['orchid'], line=dict(width=2, color='white')),
        fill='tozeroy', fillcolor='rgba(232,24,109,0.08)',
        hovertemplate="K=%{x}<br>Accuracy: %{y:.2f}%<extra></extra>",
    ))
    fig.add_annotation(
        x=best_k, y=best_acc,
        text=f"<b>Best K={int(best_k)}<br>{best_acc:.1f}%</b>",
        showarrow=True, arrowhead=2, arrowcolor=C['magenta'],
        font=dict(color=C['ink'], size=11),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor=C['border_med'], borderwidth=1,
    )
    fig.update_layout(xaxis_title="K value", yaxis_title="Accuracy (%)", showlegend=False)
    return apply_theme(fig, "KNN Accuracy by K")

# ─── KPI Card ────────────────────────────────────────────────────────────────
def kpi_card(label, value, icon, accent):
    return dbc.Col(
        html.Div([
            html.Div(icon, style={
                'fontSize': '28px', 'marginBottom': '10px',
                'filter': 'drop-shadow(0 2px 6px rgba(0,0,0,0.12))',
            }),
            html.Div(value, style={
                'fontSize': '36px', 'fontWeight': '900',
                'color': accent, 'lineHeight': '1',
                'letterSpacing': '-1.5px',
                'fontFamily': "'Playfair Display', serif",
            }),
            html.Div(label, style={
                'fontSize': '9px', 'fontWeight': '700',
                'color': C['ink_soft'], 'marginTop': '6px',
                'textTransform': 'uppercase', 'letterSpacing': '0.14em',
                'fontFamily': "'DM Sans', sans-serif",
            }),
            html.Div(style={
                'height': '3px', 'width': '32px',
                'background': accent,
                'borderRadius': '2px', 'marginTop': '10px',
                'opacity': '0.7',
            }),
        ], style={
            'background': 'rgba(255,255,255,0.75)',
            'backdropFilter': 'blur(12px)',
            'WebkitBackdropFilter': 'blur(12px)',
            'border': '1.5px solid rgba(255,255,255,0.9)',
            'borderRadius': '20px',
            'padding': '22px 20px',
            'boxShadow': '0 8px 32px rgba(232,24,109,0.10), inset 0 1px 0 rgba(255,255,255,0.8)',
            'position': 'relative',
            'overflow': 'hidden',
        }),
        xs=6, sm=6, md=3,
        style={'marginBottom': '16px'},
    )

CARD = {
    'background': 'rgba(255,255,255,0.70)',
    'backdropFilter': 'blur(14px)',
    'WebkitBackdropFilter': 'blur(14px)',
    'border': '1.5px solid rgba(255,255,255,0.90)',
    'borderRadius': '20px',
    'padding': '6px',
    'marginBottom': '16px',
    'boxShadow': '0 4px 24px rgba(232,24,109,0.08)',
    'position': 'relative',
}

# ─── Dataset Intro ────────────────────────────────────────────────────────────
def dataset_intro():
    def pill(text, color):
        return html.Span(text, style={
            'background': f'{color}18', 'color': color,
            'border': f'1.5px solid {color}55',
            'borderRadius': '20px', 'padding': '4px 13px',
            'fontSize': '11px', 'fontWeight': '700',
            'marginRight': '8px', 'marginBottom': '7px',
            'display': 'inline-block',
            'fontFamily': "'DM Sans', sans-serif",
            'letterSpacing': '0.02em',
        })

    def stat_block(num, label, accent):
        return html.Div([
            html.Div(num, style={
                'fontSize': '26px', 'fontWeight': '900', 'color': accent,
                'lineHeight': '1', 'fontFamily': "'Playfair Display', serif",
                'letterSpacing': '-1px',
            }),
            html.Div(label, style={
                'fontSize': '9px', 'color': C['ink_soft'], 'marginTop': '5px',
                'textTransform': 'uppercase', 'letterSpacing': '0.12em',
                'fontFamily': "'DM Sans', sans-serif", 'fontWeight': '700',
            }),
            html.Div(style={'height':'2px','width':'24px','background':accent,'borderRadius':'2px','marginTop':'8px'}),
        ], style={
            'background': 'rgba(255,255,255,0.65)',
            'backdropFilter': 'blur(10px)',
            'border': '1.5px solid rgba(255,255,255,0.9)',
            'borderRadius': '16px',
            'padding': '18px 20px',
            'textAlign': 'left',
            'flex': '1',
            'marginRight': '10px',
            'boxShadow': '0 4px 16px rgba(232,24,109,0.07)',
        })

    def step_card(num, title, desc, accent):
        return html.Div([
            html.Div([
                html.Span(num, style={
                    'fontSize': '11px', 'fontWeight': '800', 'color': accent,
                    'fontFamily': "'DM Sans', sans-serif", 'letterSpacing': '0.06em',
                }),
            ], style={
                'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                'background': f'{accent}18', 'border': f'1.5px solid {accent}44',
                'borderRadius': '8px', 'padding': '4px 10px',
                'marginBottom': '12px',
            }),
            html.Div(title, style={
                'fontSize': '13px', 'fontWeight': '700', 'color': C['ink'],
                'marginBottom': '7px', 'fontFamily': "'DM Sans', sans-serif",
            }),
            html.Div(desc, style={
                'fontSize': '11.5px', 'color': C['ink_mid'], 'lineHeight': '1.7',
                'fontFamily': "'DM Sans', sans-serif",
            }),
        ], style={
            'background': 'rgba(255,255,255,0.65)',
            'backdropFilter': 'blur(10px)',
            'border': f'1.5px solid rgba(255,255,255,0.9)',
            'borderTop': f'3px solid {accent}',
            'borderRadius': '16px',
            'padding': '18px',
            'flex': '1',
            'marginRight': '10px',
            'boxShadow': '0 4px 14px rgba(0,0,0,0.05)',
        })

    def insight_card(icon, title, desc, accent):
        return html.Div([
            html.Div([
                html.Div(icon, style={
                    'fontSize': '20px',
                    'width': '38px', 'height': '38px',
                    'background': f'{accent}18',
                    'border': f'1.5px solid {accent}33',
                    'borderRadius': '10px',
                    'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
                    'marginRight': '12px', 'flexShrink': '0',
                }),
                html.Div(title, style={
                    'fontSize': '13px', 'fontWeight': '700', 'color': C['ink'],
                    'fontFamily': "'DM Sans', sans-serif",
                }),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'}),
            html.Div(desc, style={
                'fontSize': '11.5px', 'color': C['ink_mid'],
                'lineHeight': '1.65', 'fontFamily': "'DM Sans', sans-serif",
            }),
        ], style={
            'background': 'rgba(255,255,255,0.65)',
            'backdropFilter': 'blur(10px)',
            'border': '1.5px solid rgba(255,255,255,0.9)',
            'borderRadius': '16px',
            'padding': '16px',
            'boxShadow': '0 4px 14px rgba(0,0,0,0.04)',
        })

    return html.Div([

        html.Div([
            html.Div([
                html.Div("✦  ABOUT THIS PROJECT", style={
                    'fontSize': '10px', 'fontWeight': '800', 'color': C['magenta'],
                    'letterSpacing': '0.22em', 'marginBottom': '12px',
                    'fontFamily': "'DM Sans', sans-serif",
                }),
                html.Div([
                    html.Span("Algerian ", style={
                        'fontFamily': "'Playfair Display', serif",
                        'fontSize': '32px', 'fontWeight': '900',
                        'fontStyle': 'italic', 'color': C['magenta'],
                    }),
                    html.Span("Product Reviews", style={
                        'fontFamily': "'Playfair Display', serif",
                        'fontSize': '32px', 'fontWeight': '900', 'color': C['ink'],
                    }),
                ]),
                html.Div(
                    "A scraped & analysed dataset of consumer opinions from Algerian Reddit "
                    "communities, covering home appliances, food, electronics, and cosmetics. "
                    "Built end-to-end with custom keyword scraping, text cleaning, and KNN classification.",
                    style={
                        'fontSize': '13px', 'color': C['ink_mid'], 'lineHeight': '1.75',
                        'maxWidth': '560px', 'marginTop': '12px',
                        'fontFamily': "'DM Sans', sans-serif",
                    }
                ),
                html.Div([
                    pill("r/AlgeriaRates", C['magenta']),
                    pill("r/Algeria",      C['tangerine']),
                    pill("April 2026",     C['teal']),
                    pill("Python · pandas · sklearn", C['violet']),
                ], style={'marginTop': '16px'}),
            ], style={'flex': '1', 'paddingRight': '30px', 'minWidth': '260px'}),

            html.Div([
                stat_block("430",   "Reviews",      C['magenta']),
                stat_block("4",     "Categories",   C['tangerine']),
                stat_block("60.5%", "KNN Accuracy", C['teal']),
                stat_block("K = 5", "Best K",       C['violet']),
            ], style={'display':'flex','flexWrap':'wrap','marginRight':'-10px','gap':'0','alignItems':'stretch'}),
        ], style={
            'display': 'flex', 'alignItems': 'flex-start',
            'flexWrap': 'wrap', 'gap': '24px', 'marginBottom': '28px',
        }),

        html.Div([
            html.Div("✦  HOW IT WAS BUILT", style={
                'fontSize': '10px', 'fontWeight': '800', 'color': C['ink_soft'],
                'letterSpacing': '0.22em', 'marginBottom': '14px',
                'fontFamily': "'DM Sans', sans-serif",
            }),
            html.Div([
                step_card("01", "Data Collection",
                    "Scraped 500+ posts from r/AlgeriaRates & r/Algeria via the Reddit JSON API with retry logic and 80+ product keywords (Condor, Samsung, Rouiba...).",
                    C['magenta']),
                step_card("02", "Data Cleaning",
                    "Removed duplicates, handled missing values, cleaned text, stripped English & French stopwords and filler review words.",
                    C['tangerine']),
                step_card("03", "Analysis",
                    "Statistical & frequency analysis (Counter). Category breakdowns on scores, comments, text length, and brand mentions.",
                    C['teal']),
                step_card("04", "ML — KNN",
                    "KNN trained on score, comments & text_length (StandardScaler, 80/20 split). Best accuracy at K=5: 60.47%.",
                    C['violet']),
            ], style={'display':'flex','flexWrap':'wrap','marginRight':'-10px'}),
        ], style={'marginBottom': '28px'}),

        html.Div("✦  KEY FINDINGS", style={
            'fontSize': '10px', 'fontWeight': '800', 'color': C['ink_soft'],
            'letterSpacing': '0.22em', 'marginBottom': '14px',
            'fontFamily': "'DM Sans', sans-serif",
        }),
        dbc.Row([
            dbc.Col(insight_card("🏠", "Appliances Dominate",
                "62.1% of posts discuss home appliances: fridges, washing machines & ACs are the most talked-about.", C['magenta']),
                md=4, style={'marginBottom': '10px'}),
            dbc.Col(insight_card("🍽️", "Food Gets the Most Upvotes",
                "Food posts earn the highest avg score (23.0); quality & availability resonate widely.", C['tangerine']),
                md=4, style={'marginBottom': '10px'}),
            dbc.Col(insight_card("📝", "Electronics = Longest Reviews",
                "Electronics posts average 588 characters which indicates buyers research thoroughly before purchasing.", C['teal']),
                md=4, style={'marginBottom': '10px'}),
        ]),
        dbc.Row([
            dbc.Col(insight_card("🏷️", "LG Leads Brand Mentions",
                "LG appears 462 times vs. only 3 for Samsung: it is deeply embedded in Algerian appliance culture.", C['violet']),
                md=4, style={'marginBottom': '4px'}),
            dbc.Col(insight_card("💬", "High Comment Engagement",
                "Appliances (20.8 avg) and Electronics (20.2 avg) generate the most discussion.", C['orchid']),
                md=4, style={'marginBottom': '4px'}),
            dbc.Col(insight_card("🤖", "Promising KNN Baseline",
                "With only 3 simple features, KNN reached 60.47% , a solid baseline for richer future models.", C['coral']),
                md=4, style={'marginBottom': '4px'}),
        ]),

    ], style={
        'background': 'rgba(255,255,255,0.50)',
        'backdropFilter': 'blur(20px)',
        'WebkitBackdropFilter': 'blur(20px)',
        'border': '2px solid rgba(255,255,255,0.85)',
        'borderRadius': '24px',
        'padding': '30px 28px',
        'marginBottom': '24px',
        'boxShadow': '0 8px 40px rgba(232,24,109,0.09), inset 0 1px 0 rgba(255,255,255,0.8)',
        'position': 'relative',
        'overflow': 'hidden',
    })


# ─── NEW: Data Documentation Panel ──────────────────────────────────────────
def data_documentation():
    """
    Covers the three academic requirements that were missing:
      1. Attribute types (categorical nominal/ordinal, quantitative continuous/discrete)
      2. Visual channel mapping (position, color, size, …)
      3. Layout design justification
    """

    # ── shared mini-helpers ──────────────────────────────────────────────────
    def section_label(text):
        return html.Div(text, style={
            'fontSize': '10px', 'fontWeight': '800', 'color': C['ink_soft'],
            'letterSpacing': '0.22em', 'marginBottom': '16px',
            'textTransform': 'uppercase', 'fontFamily': "'DM Sans', sans-serif",
        })

    # ── 1. Items description block ───────────────────────────────────────────
    items_block = html.Div([

        # What are the items?
        html.Div([
            html.Div([
                html.Div("📄", style={
                    'fontSize': '24px', 'width': '48px', 'height': '48px',
                    'background': f'{C["magenta"]}12', 'border': f'1.5px solid {C["magenta"]}30',
                    'borderRadius': '14px', 'display': 'flex',
                    'alignItems': 'center', 'justifyContent': 'center',
                    'marginRight': '16px', 'flexShrink': '0',
                }),
                html.Div([
                    html.Div("The Items", style={
                        'fontSize': '14px', 'fontWeight': '800', 'color': C['ink'],
                        'fontFamily': "'DM Sans', sans-serif", 'marginBottom': '5px',
                    }),
                    html.Div(
                        "Each item in this dataset is a Reddit post or a single consumer opinion "
                        "or question published by an Algerian user on r/AlgeriaRates or r/Algeria. "
                        "Each post represents one real-world review event: a person sharing their "
                        "experience with, or asking about, a specific Algerian product.",
                        style={
                            'fontSize': '12px', 'color': C['ink_mid'], 'lineHeight': '1.75',
                            'fontFamily': "'DM Sans', sans-serif",
                        }
                    ),
                ]),
            ], style={'display': 'flex', 'alignItems': 'flex-start', 'marginBottom': '16px'}),

            # 3 property pills: what kind of item, how many, what they represent
            html.Div([
                html.Div([
                    html.Span("Type of item", style={
                        'fontSize': '9px', 'fontWeight': '800', 'color': C['ink_soft'],
                        'textTransform': 'uppercase', 'letterSpacing': '0.12em',
                        'display': 'block', 'marginBottom': '4px',
                        'fontFamily': "'DM Sans', sans-serif",
                    }),
                    html.Span("Document / Event", style={
                        'fontSize': '13px', 'fontWeight': '700', 'color': C['magenta'],
                        'fontFamily': "'Playfair Display', serif",
                    }),
                ], style={
                    'flex': '1', 'background': f'{C["magenta"]}08',
                    'border': f'1px solid {C["magenta"]}22',
                    'borderRadius': '12px', 'padding': '14px 16px',
                }),
                html.Div([
                    html.Span("Total items", style={
                        'fontSize': '9px', 'fontWeight': '800', 'color': C['ink_soft'],
                        'textTransform': 'uppercase', 'letterSpacing': '0.12em',
                        'display': 'block', 'marginBottom': '4px',
                        'fontFamily': "'DM Sans', sans-serif",
                    }),
                    html.Span("430 posts", style={
                        'fontSize': '13px', 'fontWeight': '700', 'color': C['tangerine'],
                        'fontFamily': "'Playfair Display', serif",
                    }),
                ], style={
                    'flex': '1', 'background': f'{C["tangerine"]}08',
                    'border': f'1px solid {C["tangerine"]}22',
                    'borderRadius': '12px', 'padding': '14px 16px',
                    'marginLeft': '10px',
                }),
                html.Div([
                    html.Span("What they represent", style={
                        'fontSize': '9px', 'fontWeight': '800', 'color': C['ink_soft'],
                        'textTransform': 'uppercase', 'letterSpacing': '0.12em',
                        'display': 'block', 'marginBottom': '4px',
                        'fontFamily': "'DM Sans', sans-serif",
                    }),
                    html.Span("Consumer opinions on Algerian products", style={
                        'fontSize': '12px', 'fontWeight': '700', 'color': C['teal'],
                        'fontFamily': "'Playfair Display', serif",
                    }),
                ], style={
                    'flex': '2', 'background': f'{C["teal"]}08',
                    'border': f'1px solid {C["teal"]}22',
                    'borderRadius': '12px', 'padding': '14px 16px',
                    'marginLeft': '10px',
                }),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '0'}),
        ], style={
            'background': 'rgba(255,255,255,0.60)',
            'border': f'1.5px solid {C["border_med"]}',
            'borderRadius': '16px', 'padding': '20px 22px',
            'marginBottom': '16px',
        }),

    ], style={'marginBottom': '24px'})

    # ── 2. Attribute Table ───────────────────────────────────────────────────
    TYPE_BADGE = {
        'Categorical — Nominal':     (C['magenta'],   '#FFF0F5'),
        'Categorical — Ordinal':     (C['tangerine'],  '#FFF5EE'),
        'Quantitative — Discrete':   (C['teal'],       '#F0FAFA'),
        'Quantitative — Continuous': (C['violet'],     '#F5F0FF'),
        'Temporal':                  (C['sky'],         '#F0F8FF'),
    }

    def type_badge(label):
        color, bg = TYPE_BADGE.get(label, (C['ink_soft'], '#F5F5F5'))
        return html.Span(label, style={
            'background': bg, 'color': color,
            'border': f'1.5px solid {color}44',
            'borderRadius': '20px', 'padding': '3px 10px',
            'fontSize': '10px', 'fontWeight': '700',
            'fontFamily': "'DM Sans', sans-serif",
            'whiteSpace': 'nowrap',
        })

    # Each tuple: (attribute_name, what it describes about the item, type, justification)
    ATTRS = [
        ("title",
         "The headline text of the post: how the item is identified/named by its author.",
         "Categorical (Nominal)",
         "A free-text string label. No inherent order or magnitude between titles; each is a unique, "
         "unordered descriptor for that post."),
        ("category",
         "The product domain the post belongs to: appliances, food, electronics, or cosmetics.",
         "Categorical (Nominal)",
         "4 distinct, unordered groups assigned during scraping via keyword matching. There is no "
         "ranking between categories: 'food' is not greater or lesser than 'electronics'."),
        ("subreddit",
         "The community where the post was published: r/AlgeriaRates or r/Algeria.",
         "Categorical (Nominal)",
         "Two unordered source labels. Neither subreddit is 'higher' than the other; they are "
         "qualitatively different collection contexts."),
        ("score",
         "The net upvote count the post received from Reddit users — a measure of popularity.",
         "Quantitative (Discrete)",
         "Counts are whole integers (0, 1, 2, …). Discrete because partial votes are impossible. "
         "Meaningful zero and ratio comparisons: a score of 20 is twice as popular as 10."),
        ("comments",
         "The number of replies the post received — a measure of community engagement depth.",
         "Quantitative (Discrete)",
         "Integer comment counts (≥ 0). Discrete; you cannot have half a comment. Higher values "
         "indicate posts that sparked more conversation."),
        ("text_length",
         "The character count of the post body: how long/detailed the review is.",
         "Quantitative (Continuous)",
         "Treated as continuous: character count can take any non-negative integer value with no "
         "natural gap between values. Longer posts signal more detailed opinions."),
        ("date",
         "The timestamp of when the post was published on Reddit.",
         "Temporal",
         "A time-ordered attribute parsed into month and day-of-week to expose activity patterns. "
         "Temporal data has inherent order and direction (earlier → later)."),
        ("clean_content",
         "The cleaned body text of the post after stopword removal and normalisation.",
         "Categorical (Nominal)",
         "Free-text; each post has a unique string value with no ordinal relationship to other posts. "
         "Used as the source for keyword frequency counting."),
    ]

    header_style = {
        'fontSize': '10px', 'fontWeight': '800', 'color': C['ink_soft'],
        'textTransform': 'uppercase', 'letterSpacing': '0.12em',
        'padding': '10px 14px', 'borderBottom': f'1px solid {C["border_med"]}',
        'fontFamily': "'DM Sans', sans-serif",
        'background': 'rgba(232,24,109,0.04)',
    }
    cell_style = {
        'padding': '11px 14px',
        'fontSize': '11.5px', 'color': C['ink'],
        'borderBottom': f'1px solid {C["border"]}',
        'fontFamily': "'DM Sans', sans-serif",
        'verticalAlign': 'top',
    }
    mono_style = {**cell_style, 'fontFamily': "'Courier New', monospace",
                  'fontWeight': '700', 'color': C['magenta'], 'fontSize': '12px'}

    attr_table = html.Div([
        html.Table([
            html.Thead(html.Tr([
                html.Th("Attribute",              style={**header_style, 'width': '12%'}),
                html.Th("What it describes",      style={**header_style, 'width': '28%'}),
                html.Th("Type",                   style={**header_style, 'width': '22%'}),
                html.Th("Justification",          style={**header_style, 'width': '38%'}),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(attr,  style=mono_style),
                    html.Td(desc,  style={**cell_style, 'fontStyle': 'italic', 'color': C['ink_mid']}),
                    html.Td(type_badge(typ), style={**cell_style, 'paddingTop': '9px'}),
                    html.Td(just,  style={**cell_style, 'color': C['ink_mid'], 'lineHeight': '1.65'}),
                ], style={'background': 'rgba(255,255,255,0)' if i % 2 == 0 else 'rgba(232,24,109,0.025)'})
                for i, (attr, desc, typ, just) in enumerate(ATTRS)
            ]),
        ], style={'width': '100%', 'borderCollapse': 'collapse'}),
    ], style={
        'background': 'rgba(255,255,255,0.60)',
        'border': f'1.5px solid {C["border_med"]}',
        'borderRadius': '16px', 'overflow': 'hidden',
        'marginBottom': '24px',
    })

    # ── 2. Visual Channel Mapping ────────────────────────────────────────────
    MAPPINGS = [
        ("🎨", "Color",    C['magenta'],
         "category → hue",
         "Category is nominal: hue is the ideal channel for distinguishing unordered groups "
         "without implying rank. Each of the 4 categories has a fixed, perceptually distinct color "
         "(magenta, orange, teal, orchid) applied consistently across every chart."),
        ("📍", "Position X/Y", C['tangerine'],
         "score, comments → axes",
         "Score (upvotes) and comments are quantitative: position along a spatial axis is the "
         "most accurate perceptual channel for magnitude comparisons. Log-scale X in the scatter "
         "chart handles the heavy right-skew of Reddit scores."),
        ("⭕", "Size",     C['teal'],
         "text_length → bubble area",
         "Text length is a continuous quantitative variable. Area encodes a third dimension in the "
         "scatter chart (log-scaled to prevent extreme outliers from dominating). Larger bubbles "
         "immediately signal longer, more detailed posts."),
        ("🌡️", "Color Intensity", C['violet'],
         "post count → heatmap luminance",
         "In the Day × Month heatmap, post count is encoded through a sequential pink→dark-rose "
         "color scale. Intensity (luminance) is the natural channel for ordered magnitudes on a grid."),
        ("📏", "Bar Length", C['sky'],
         "avg score, frequency → bar length",
         "Bar length encodes quantitative magnitude for the Score by Category and Top Keywords charts. "
         "Length is among the most accurately perceived channels for 1D comparisons."),
        ("📈", "Line + Markers", C['orchid'],
         "time → position along X, trend → line slope",
         "The timeline maps months to ordered X positions (temporal → spatial order) and uses "
         "spline interpolation to reveal trends. Markers highlight discrete monthly values."),
    ]

    def channel_card(icon, channel, accent, mapping, rationale):
        return html.Div([
            html.Div([
                html.Div(icon, style={
                    'fontSize': '22px', 'width': '44px', 'height': '44px',
                    'background': f'{accent}15', 'border': f'1.5px solid {accent}33',
                    'borderRadius': '12px', 'display': 'flex',
                    'alignItems': 'center', 'justifyContent': 'center',
                    'marginRight': '12px', 'flexShrink': '0',
                }),
                html.Div([
                    html.Div(channel, style={
                        'fontSize': '13px', 'fontWeight': '800', 'color': C['ink'],
                        'fontFamily': "'DM Sans', sans-serif",
                    }),
                    html.Div(mapping, style={
                        'fontSize': '11px', 'color': accent, 'fontWeight': '700',
                        'fontFamily': "'Courier New', monospace", 'marginTop': '2px',
                    }),
                ]),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
            html.Div(rationale, style={
                'fontSize': '11.5px', 'color': C['ink_mid'], 'lineHeight': '1.65',
                'fontFamily': "'DM Sans', sans-serif",
            }),
            html.Div(style={
                'height': '2px', 'background': f'linear-gradient(90deg, {accent}55, transparent)',
                'borderRadius': '2px', 'marginTop': '12px',
            }),
        ], style={
            'background': 'rgba(255,255,255,0.65)',
            'backdropFilter': 'blur(10px)',
            'border': '1.5px solid rgba(255,255,255,0.9)',
            'borderLeft': f'3px solid {accent}',
            'borderRadius': '16px',
            'padding': '18px',
            'boxShadow': '0 4px 14px rgba(0,0,0,0.04)',
        })

    channel_grid = dbc.Row([
        dbc.Col(channel_card(*m), md=4, style={'marginBottom': '12px'})
        for m in MAPPINGS
    ], style={'marginBottom': '24px'})

    # ── 3. Layout Justification ──────────────────────────────────────────────
    def layout_item(num, title, desc, accent):
        return html.Div([
            html.Div(num, style={
                'fontSize': '11px', 'fontWeight': '900', 'color': accent,
                'fontFamily': "'Playfair Display', serif",
                'width': '28px', 'height': '28px',
                'background': f'{accent}15', 'border': f'1.5px solid {accent}44',
                'borderRadius': '8px',
                'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
                'flexShrink': '0', 'marginRight': '14px',
            }),
            html.Div([
                html.Div(title, style={
                    'fontSize': '12px', 'fontWeight': '700', 'color': C['ink'],
                    'fontFamily': "'DM Sans', sans-serif", 'marginBottom': '4px',
                }),
                html.Div(desc, style={
                    'fontSize': '11.5px', 'color': C['ink_mid'], 'lineHeight': '1.65',
                    'fontFamily': "'DM Sans', sans-serif",
                }),
            ]),
        ], style={'display': 'flex', 'alignItems': 'flex-start', 'marginBottom': '16px'})

    LAYOUT_REASONS = [
        ("1", "Sticky header: orientation at all times",
         "The navigation bar stays fixed at the top so users always know the project context while scrolling through long chart sections.",
         C['magenta']),
        ("2", "About panel first: context before data",
         "Presenting the dataset description, pipeline steps, and key findings before any chart primes the reader with domain knowledge needed to interpret visualizations correctly.",
         C['tangerine']),
        ("3", "Data documentation second: academic transparency",
         "Attribute types and visual channel mappings are placed immediately after the intro so the instructor can verify design decisions before engaging with the interactive charts.",
         C['teal']),
        ("4", "Filter bar above all charts: global scope",
         "A single filter bar (category checklist, score slider, subreddit dropdown) controls every chart simultaneously. Placing it above all sections avoids duplicated controls and makes the causal relationship between filter and chart update obvious.",
         C['violet']),
        ("5", "3-column KPI row: scannable overview",
         "Four headline metrics (posts, avg score, avg comments, avg length) are shown as large-number cards in a responsive grid. This provides an immediate numerical summary before the user dives into charts.",
         C['orchid']),
        ("6", "Section grouping by analytical task",
         "Charts are grouped into thematic rows: Distribution & Scores → Activity Over Time → Text Analysis → Patterns & ML. This mirrors a natural analytical narrative: 'what is the data?' → 'when?' → 'what do people say?' → 'can a model learn it?'",
         C['sky']),
        ("7", "Wide + narrow column pairing",
         "Each row pairs a wider chart (7/12 or 6/12 columns) with a complementary narrower one. This creates visual rhythm and allows related charts to share horizontal space without competing for attention.",
         C['coral']),
    ]

    layout_panel = html.Div([
        html.Div([layout_item(*r) for r in LAYOUT_REASONS]),
    ], style={
        'background': 'rgba(255,255,255,0.60)',
        'border': f'1.5px solid {C["border_med"]}',
        'borderRadius': '16px',
        'padding': '22px 24px',
    })

    # ── Assemble the full documentation block ────────────────────────────────
    return html.Div([
        html.Div("✦  ABOUT THIS PROJECT", style={
            'fontSize': '10px', 'fontWeight': '800', 'color': C['magenta'],
            'letterSpacing': '0.22em', 'marginBottom': '6px',
            'fontFamily': "'DM Sans', sans-serif",
        }),
        html.Div([
            html.Span("Data ", style={
                'fontFamily': "'Playfair Display', serif",
                'fontSize': '26px', 'fontWeight': '900', 'fontStyle': 'italic',
                'color': C['magenta'],
            }),
            html.Span("Documentation", style={
                'fontFamily': "'Playfair Display', serif",
                'fontSize': '26px', 'fontWeight': '900', 'color': C['ink'],
            }),
        ], style={'marginBottom': '6px'}),
        html.Div(
            "Attribute classification, visual encoding choices, and layout design rationale.",
            style={
                'fontSize': '12px', 'color': C['ink_mid'],
                'fontFamily': "'DM Sans', sans-serif", 'marginBottom': '24px',
            }
        ),

        section_label("✦  ITEMS & ATTRIBUTES"),
        items_block,
        attr_table,

        section_label("✦  DATA → VISUAL CHANNEL MAPPING"),
        channel_grid,

        section_label("✦  DASHBOARD LAYOUT JUSTIFICATION"),
        layout_panel,

    ], style={
        'background': 'rgba(255,255,255,0.50)',
        'backdropFilter': 'blur(20px)',
        'WebkitBackdropFilter': 'blur(20px)',
        'border': '2px solid rgba(255,255,255,0.85)',
        'borderRadius': '24px',
        'padding': '30px 28px',
        'marginBottom': '24px',
        'boxShadow': '0 8px 40px rgba(232,24,109,0.09), inset 0 1px 0 rgba(255,255,255,0.8)',
        'position': 'relative',
        'overflow': 'hidden',
    })


# ─── App ─────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700;1,900&family=DM+Sans:wght@300;400;500;600;700;800&display=swap",
    ],
    title="DZ Products · Reddit Analysis",
)

app.index_string = '''<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  :root {
    --bg:      #FDE8EF;
    --bg2:     #FAD4E0;
    --magenta: #E8186D;
    --ink:     #2A1120;
    --ink-mid: #7A3A55;
    --ink-soft:#B07090;
    --border:  rgba(220,100,140,0.20);
  }

  body {
    background: var(--bg) !important;
    margin: 0;
    background-image:
      radial-gradient(ellipse 800px 600px at 10% 0%,   rgba(232,24,109,0.12) 0%, transparent 70%),
      radial-gradient(ellipse 600px 500px at 90% 20%,  rgba(217,70,239,0.10) 0%, transparent 70%),
      radial-gradient(ellipse 700px 500px at 50% 100%, rgba(255,140,66,0.09) 0%, transparent 70%),
      radial-gradient(ellipse 500px 400px at 80% 80%,  rgba(20,184,166,0.08) 0%, transparent 70%);
    background-attachment: fixed;
  }

  * { box-sizing: border-box; }

  h1, h2, h3, h4, h5, h6 { font-family: 'Playfair Display', serif !important; }
  p, span, div, label, input { font-family: 'DM Sans', sans-serif !important; }

  /* ── Dropdown ─────────────────────────────────────────── */
  .Select-control {
    background: rgba(255,255,255,0.80) !important;
    border: 1.5px solid rgba(232,24,109,0.25) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(8px);
  }
  .Select-menu-outer {
    background: rgba(255,240,246,0.97) !important;
    border: 1.5px solid rgba(232,24,109,0.25) !important;
    border-radius: 0 0 12px 12px !important;
    z-index: 9999 !important;
    position: absolute !important;
  }
  .Select-option { color: var(--ink) !important; background: transparent !important; }
  .Select-option:hover,
  .Select-option.is-focused { background: rgba(232,24,109,0.08) !important; }
  .Select-value-label, .Select-placeholder { color: var(--ink-mid) !important; }

  .filter-bar { position: relative; z-index: 200; }

  /* Slider */
  .rc-slider-track { background: linear-gradient(90deg, #E8186D, #D946EF) !important; }
  .rc-slider-handle { border-color: #E8186D !important; background: white !important; box-shadow: 0 0 0 3px rgba(232,24,109,0.25) !important; }
  .rc-slider-rail { background: rgba(232,24,109,0.15) !important; }
  .rc-slider-mark-text { color: var(--ink-soft) !important; font-size: 10px !important; }

  .dash-checklist label { color: var(--ink) !important; }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg2); }
  ::-webkit-scrollbar-thumb { background: rgba(232,24,109,0.35); border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: rgba(232,24,109,0.55); }

  .chart-card { transition: transform 0.2s ease, box-shadow 0.2s ease; }
  .chart-card:hover { transform: translateY(-3px); box-shadow: 0 12px 40px rgba(232,24,109,0.14) !important; }

  .section-label {
    font-size: 10px; font-weight: 800; letter-spacing: 0.22em;
    color: var(--ink-soft); text-transform: uppercase;
    margin-bottom: 14px; display: flex; align-items: center; gap: 10px;
  }
  .section-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(232,24,109,0.25), transparent);
  }

  /* ── Doc table hover ──────────────────────────────────── */
  tbody tr:hover td { background: rgba(232,24,109,0.04) !important; }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>'''

# ─── Layout ──────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={'minHeight': '100vh'},
    children=[

        # ── HEADER ──────────────────────────────────────────────────────────
        html.Div([
            dbc.Container(fluid=True, children=[
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Div([reddit_icon(28)], style={
                                'width': '52px', 'height': '52px', 'borderRadius': '16px',
                                'background': 'rgba(255,255,255,0.80)',
                                'backdropFilter': 'blur(8px)',
                                'border': '1.5px solid rgba(255,255,255,0.95)',
                                'display': 'flex', 'alignItems': 'center',
                                'justifyContent': 'center',
                                'marginRight': '16px', 'flexShrink': '0',
                                'boxShadow': '0 4px 16px rgba(232,24,109,0.15)',
                            }),
                            html.Div([
                                html.Div([
                                    html.Span("DZ", style={
                                        'fontFamily': "'Playfair Display', serif",
                                        'fontWeight': '900', 'fontStyle': 'italic',
                                        'fontSize': '22px', 'color': C['magenta'],
                                    }),
                                    html.Span(" Products Analysis", style={
                                        'fontFamily': "'Playfair Display', serif",
                                        'fontWeight': '700', 'fontSize': '22px',
                                        'color': C['ink'],
                                    }),
                                ]),
                                html.Div([
                                    reddit_icon(12),
                                    html.Span(
                                        "Reddit consumer reviews · r/algeriarates & r/algeria · 2026",
                                        style={'fontSize': '11px', 'color': C['ink_soft'], 'marginLeft': '5px'},
                                    ),
                                ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '4px'}),
                            ]),
                        ], style={'display': 'flex', 'alignItems': 'center'}),
                    ], md=7),
                    dbc.Col([
                        html.Div([
                            html.Div("430 posts · 4 categories · 2 subreddits", style={
                                'color': C['ink_soft'], 'fontSize': '12px', 'marginBottom': '5px',
                            }),
                            html.Div([
                                html.Span("✦  made by  ", style={'color': C['ink_soft'], 'fontSize': '11px'}),
                                html.Span("Chadli Sarah Nada", style={
                                    'color': C['magenta'], 'fontSize': '12px',
                                    'fontWeight': '700', 'fontFamily': "'Playfair Display', serif",
                                    'fontStyle': 'italic',
                                }),
                                html.Span("  &  ", style={'color': C['ink_soft'], 'fontSize': '11px'}),
                                html.Span("Tachache Aya", style={
                                    'color': C['magenta'], 'fontSize': '12px',
                                    'fontWeight': '700', 'fontFamily': "'Playfair Display', serif",
                                    'fontStyle': 'italic',
                                }),
                            ]),
                        ], style={'textAlign': 'right'}),
                    ], md=5),
                ], align='center'),
            ]),
        ], style={
            'background': 'rgba(255,255,255,0.72)',
            'backdropFilter': 'blur(20px)',
            'WebkitBackdropFilter': 'blur(20px)',
            'borderBottom': '1.5px solid rgba(255,255,255,0.9)',
            'padding': '16px 24px',
            'marginBottom': '24px',
            'boxShadow': '0 4px 24px rgba(232,24,109,0.08)',
            'position': 'sticky', 'top': '0', 'zIndex': '100',
        }),

        dbc.Container(fluid=True, style={'maxWidth': '1380px', 'padding': '0 20px'}, children=[

            # 1. Project intro (unchanged)
            dataset_intro(),

            # 2. ── NEW: Data documentation ──────────────────────────────
            data_documentation(),

            # ── FILTER BAR ──────────────────────────────────────────────────
            html.Div([
                html.Div("✦  FILTER & EXPLORE", className="section-label"),
                dbc.Row([
                    dbc.Col([
                        html.Div("Categories", style={
                            'fontSize': '10px', 'fontWeight': '700',
                            'color': C['ink_soft'], 'textTransform': 'uppercase',
                            'letterSpacing': '0.14em', 'marginBottom': '10px',
                        }),
                        dcc.Checklist(
                            id='cat-filter',
                            options=[{
                                'label': html.Span([
                                    html.Span("●", style={
                                        'color': CAT_COLORS.get(c,'#ccc'),
                                        'marginRight': '5px', 'fontSize': '13px',
                                    }),
                                    html.Span(c.capitalize(), style={
                                        'fontWeight': '600', 'color': C['ink'], 'fontSize': '12px',
                                    }),
                                ]),
                                'value': c,
                            } for c in CATEGORIES],
                            value=CATEGORIES, inline=True,
                            labelStyle={
                                'marginRight': '22px', 'cursor': 'pointer',
                                'display': 'inline-flex', 'alignItems': 'center',
                            },
                        ),
                    ], md=6),
                    dbc.Col([
                        html.Div("Min Score", style={
                            'fontSize': '10px', 'fontWeight': '700',
                            'color': C['ink_soft'], 'textTransform': 'uppercase',
                            'letterSpacing': '0.14em', 'marginBottom': '10px',
                        }),
                        dcc.Slider(
                            id='score-slider', min=0, max=100, step=5, value=0,
                            marks={0: '0', 25: '25', 50: '50', 75: '75', 100: '100+'},
                            tooltip={'placement': 'bottom', 'always_visible': False},
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Div("Subreddit", style={
                            'fontSize': '10px', 'fontWeight': '700',
                            'color': C['ink_soft'], 'textTransform': 'uppercase',
                            'letterSpacing': '0.14em', 'marginBottom': '10px',
                        }),
                        dcc.Dropdown(
                            id='sub-filter',
                            options=[{'label': f'r/{s}', 'value': s} for s in df['subreddit'].unique()],
                            value=None, placeholder='All subreddits',
                        ),
                    ], md=2),
                ], align='start'),
            ], className="filter-bar", style={
                'background': 'rgba(255,255,255,0.68)',
                'backdropFilter': 'blur(14px)',
                'WebkitBackdropFilter': 'blur(14px)',
                'border': '1.5px solid rgba(255,255,255,0.92)',
                'borderRadius': '20px',
                'padding': '20px 24px',
                'marginBottom': '20px',
                'boxShadow': '0 4px 20px rgba(232,24,109,0.07)',
            }),

            html.Div("✦  OVERVIEW METRICS", className="section-label"),
            html.Div(id='kpi-row'),

            html.Div("✦  DISTRIBUTION & SCORES", className="section-label"),
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id='fig-donut',     config={'displayModeBar': False}, style={'height': '340px'}), style=CARD, className="chart-card"), md=4),
                dbc.Col(html.Div(dcc.Graph(id='fig-score-bar', config={'displayModeBar': False}, style={'height': '340px'}), style=CARD, className="chart-card"), md=4),
                dbc.Col(html.Div(dcc.Graph(id='fig-top-posts', config={'displayModeBar': False}, style={'height': '340px'}), style=CARD, className="chart-card"), md=4),
            ]),

            html.Div("✦  ACTIVITY OVER TIME", className="section-label"),
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id='fig-timeline', config={'displayModeBar': False}, style={'height': '320px'}), style=CARD, className="chart-card"), md=7),
                dbc.Col(html.Div(dcc.Graph(id='fig-scatter',  config={'displayModeBar': False}, style={'height': '320px'}), style=CARD, className="chart-card"), md=5),
            ]),

            html.Div("✦  TEXT ANALYSIS", className="section-label"),
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id='fig-keywords', config={'displayModeBar': False}, style={'height': '320px'}), style=CARD, className="chart-card"), md=6),
                dbc.Col(html.Div(dcc.Graph(id='fig-violin',   config={'displayModeBar': False}, style={'height': '320px'}), style=CARD, className="chart-card"), md=6),
            ]),

            html.Div("✦  PATTERNS & MACHINE LEARNING", className="section-label"),
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id='fig-heatmap', config={'displayModeBar': False}, style={'height': '320px'}), style=CARD, className="chart-card"), md=7),
                dbc.Col(html.Div(dcc.Graph(id='fig-knn',     config={'displayModeBar': False}, style={'height': '320px'}), style=CARD, className="chart-card"), md=5),
            ]),

            # ── FOOTER ─────────────────────────────────────────────────────
            html.Div([
                html.Div("✦", style={'color': C['magenta'], 'fontSize': '18px', 'marginBottom': '8px'}),
                html.Div([
                    html.Span("DZ Products Analysis", style={
                        'fontFamily': "'Playfair Display', serif",
                        'fontStyle': 'italic', 'fontWeight': '700',
                        'color': C['ink'], 'fontSize': '14px',
                    }),
                    html.Span("  ·  Built with Dash & Plotly", style={'color': C['ink_soft'], 'fontSize': '11px'}),
                ]),
                html.Div([
                    html.Span("Chadli Sarah Nada  &  Tachache Aya", style={
                        'fontFamily': "'Playfair Display', serif",
                        'fontStyle': 'italic', 'color': C['magenta'],
                        'fontSize': '12px', 'fontWeight': '700',
                    }),
                ], style={'marginTop': '4px'}),
            ], style={'textAlign': 'center', 'padding': '28px 0 40px'}),
        ]),
    ],
)


# ─── Callbacks ────────────────────────────────────────────────────────────────
def get_filtered(cats, sub, min_score):
    cats = cats or CATEGORIES
    mask = df['category'].isin(cats) & (df['score'] >= min_score)
    if sub:
        mask = mask & (df['subreddit'] == sub)
    return df[mask]

@app.callback(
    Output('kpi-row', 'children'),
    Input('cat-filter', 'value'),
    Input('sub-filter', 'value'),
    Input('score-slider', 'value'),
)
def update_kpis(cats, sub, min_score):
    d = get_filtered(cats, sub, min_score)
    n            = len(d)
    avg_score    = round(d['score'].mean(), 1)    if n else 0
    avg_comments = round(d['comments'].mean(), 1) if n else 0
    avg_len      = int(d['text_length'].mean())   if n else 0
    return dbc.Row([
        kpi_card("Total Posts",     str(n),            "📄", C['magenta']),
        kpi_card("Avg Score",       str(avg_score),    "⬆️",  C['tangerine']),
        kpi_card("Avg Comments",    str(avg_comments), "💬", C['teal']),
        kpi_card("Avg Text Length", str(avg_len),      "📝", C['violet']),
    ], style={'marginBottom': '10px'})

@app.callback(
    Output('fig-donut',     'figure'),
    Output('fig-score-bar', 'figure'),
    Output('fig-top-posts', 'figure'),
    Output('fig-timeline',  'figure'),
    Output('fig-scatter',   'figure'),
    Output('fig-keywords',  'figure'),
    Output('fig-violin',    'figure'),
    Output('fig-heatmap',   'figure'),
    Output('fig-knn',       'figure'),
    Input('cat-filter',   'value'),
    Input('sub-filter',   'value'),
    Input('score-slider', 'value'),
)
def update_all_charts(cats, sub, min_score):
    d = get_filtered(cats, sub, min_score)
    return (
        fig_donut(d), fig_score_bar(d), fig_top_posts(d),
        fig_timeline(d), fig_scatter(d), fig_keywords(d),
        fig_violin(d), fig_heatmap(d), fig_knn(),
    )

if __name__ == '__main__':
    app.run(debug=True, port=8050)
