import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc


def cria_graficos_ecommerce(df):
    # Gráfico 1: Histograma de Preços
    fig_hist = px.histogram(
        df,
        x='Preço',
        nbins=30,
        title='Distribuição de Preços dos Produtos',
        color_discrete_sequence=['#2ca02c']
    )
    fig_hist.update_layout(xaxis_title='Preço (R$)', yaxis_title='Frequência')

    # Gráfico 2: Dispersão (Nº de Avaliações vs Qtd Vendidos)
    fig_scatter = px.scatter(
        df,
        x='N_Avaliações',
        y='Qtd_Vendidos_Cod',
        color='Gênero',
        hover_data=['Marca'],
        title='Relação entre Volume de Avaliações e Código de Vendas'
    )
    fig_scatter.update_layout(xaxis_title='Número de Avaliações', yaxis_title='Código de Quantidade Vendida')

    # Gráfico 3: Mapa de Calor (Correlação entre Métricas Numéricas)
    cols_numericas = ['Preço', 'Nota', 'N_Avaliações', 'Desconto', 'Qtd_Vendidos_Cod']
    cols_existentes = [col for col in cols_numericas if col in df.columns]
    corr_matrix = df[cols_existentes].corr()
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='Viridis',
        title='Mapa de Calor de Correlação Linear'
    )

    # Gráfico 4: Gráfico de Barras (Top 10 Marcas com Mais Produtos)
    top_marcas = df['Marca'].value_counts().head(10).reset_index()
    top_marcas.columns = ['Marca', 'Quantidade']
    fig_bar = px.bar(
        top_marcas,
        x='Marca',
        y='Quantidade',
        color='Quantidade',
        color_continuous_scale='Blues',
        title='Top 10 Marcas com Maior Sortimento no E-commerce'
    )
    fig_bar.update_layout(xaxis_title='Marca', yaxis_title='Quantidade de Produtos')

    # Gráfico 5: Gráfico de Pizza / Donut (Distribuição por Gênero)
    df_genero = df['Gênero'].value_counts().reset_index()
    df_genero.columns = ['Gênero', 'Total']
    fig_pie = px.pie(
        df_genero,
        names='Gênero',
        values='Total',
        hole=0.3,
        title='Participação por Categoria de Gênero',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    # Gráfico 6: Densidade das Notas dos Produtos
    fig_densidade = px.histogram(
        df,
        x='Nota',
        histnorm='probability density',
        title='Distribuição de Satisfação (Notas dos Produtos)',
        color_discrete_sequence=['#9467bd']
    )
    fig_densidade.update_layout(xaxis_title='Nota (0 a 5)', yaxis_title='Densidade')

    # Gráfico 7: Gráfico de Regressão Linear com trendline='ols' (Opção 1)
    fig_reg = px.scatter(
        df.dropna(subset=['Preço', 'Nota']),
        x='Preço',
        y='Nota',
        opacity=0.5,
        trendline='ols',
        trendline_color_override='red',
        title='Regressão Linear: Relação entre Preço e Nota'
    )
    fig_reg.update_layout(xaxis_title='Preço (R$)', yaxis_title='Nota')

    return fig_hist, fig_scatter, fig_heatmap, fig_bar, fig_pie, fig_densidade, fig_reg


def cria_app(df):
    app = Dash(__name__)

    f_hist, f_scatter, f_heat, f_bar, f_pie, f_dens, f_reg = cria_graficos_ecommerce(df)

    app.layout = html.Div(
        style={'fontFamily': 'Arial, sans-serif', 'margin': '25px', 'backgroundColor': '#f8f9fa'},
        children=[
            html.H1(
                "Painel Executivo de E-commerce — Análise de Desempenho",
                style={'textAlign': 'center', 'color': '#2c3e50'}
            ),
            html.P(
                "Visualização interativa das métricas de vendas, avaliações, preços e catálogo.",
                style={'textAlign': 'center', 'color': '#7f8c8d'}
            ),
            html.Hr(),

            dcc.Graph(figure=f_hist),
            dcc.Graph(figure=f_scatter),
            dcc.Graph(figure=f_heat),
            dcc.Graph(figure=f_bar),
            dcc.Graph(figure=f_pie),
            dcc.Graph(figure=f_dens),
            dcc.Graph(figure=f_reg)
        ]
    )

    return app


# Leitura dos dados
df = pd.read_csv('../raw/ecommerce_estatistica.csv')

if __name__ == '__main__':
    app = cria_app(df)
    app.run(debug=True, port=8050)