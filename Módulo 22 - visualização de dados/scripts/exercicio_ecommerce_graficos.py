import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configuração visual padrão para gráficos limpos
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Leitura e Diagnóstico do DataFrame
df = pd.read_csv('../raw/ecommerce_estatistica.csv')
print("--- Primeiras Linhas da Base ---")
print(df.head().to_string())

print("\n--- Informações Gerais e Tipos ---")
print(df.info())

print("\n--- Resumo Estatístico ---")
print(df.describe())

# ==========================================
# 1. GRÁFICO DE HISTOGRAMA
# ==========================================
plt.figure(figsize=(10, 6))
plt.hist(df['Preço'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
plt.title('Distribuição de Frequência dos Preços dos Produtos', fontsize=14, fontweight='bold')
plt.xlabel('Preço (R$)', fontsize=12)
plt.ylabel('Frequência (Quantidade de Produtos)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==========================================
# 2. GRÁFICO DE DISPERSÃO
# ==========================================
plt.figure(figsize=(10, 6))
plt.scatter(df['N_Avaliações'], df['Qtd_Vendidos_Cod'], color='#e67e22', alpha=0.6, edgecolors='w', s=50)
plt.title('Relação entre Número de Avaliações e Volume de Vendas', fontsize=14, fontweight='bold')
plt.xlabel('Número de Avaliações Recebidas', fontsize=12)
plt.ylabel('Volume de Vendas (Codificado)', fontsize=12)
plt.tight_layout()
plt.show()

# ==========================================
# 3. MAPA DE CALOR (HEATMAP)
# ==========================================
plt.figure(figsize=(10, 8))
# Seleciona apenas as variáveis numéricas para matriz de correlação
colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
matriz_corr = df[colunas_numericas].corr()

sns.heatmap(matriz_corr, annot=True, fmt='.2f', cmap='Blues', linewidths=0.5)
plt.title('Mapa de Calor da Correlação entre Variáveis do E-commerce', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ==========================================
# 4. GRÁFICO DE BARRAS
# ==========================================
plt.figure(figsize=(10, 6))
top_marcas = df['Marca'].value_counts().head(10)
top_marcas.plot(kind='bar', color='#2ecc71', edgecolor='black')
plt.title('Top 10 Marcas com Maior Quantidade de Produtos Cadastrados', fontsize=14, fontweight='bold')
plt.xlabel('Marca do Produto', fontsize=12)
plt.ylabel('Quantidade de Itens no Catálogo', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ==========================================
# 5. GRÁFICO DE PIZZA
# ==========================================
plt.figure(figsize=(8, 8))
# Agrupa as 4 principais categorias e resume as demais em 'Outros' para evitar poluição
top_categorias = df['Gênero'].value_counts().head(4)
plt.pie(
    top_categorias.values,
    labels=top_categorias.index,
    autopct='%.1f%%',
    startangle=90,
    colors=['#5dade2', '#f4d03f', '#58d68d', '#ec7063'],
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
plt.title('Participação dos Principais Segmentos no Portfólio', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ==========================================
# 6. GRÁFICO DE DENSIDADE (KDE)
# ==========================================
plt.figure(figsize=(10, 6))
sns.kdeplot(df['Nota'], fill=True, color='#9b59b6', alpha=0.6, linewidth=2)
plt.title('Curva de Densidade da Nota de Avaliação dos Clientes', fontsize=14, fontweight='bold')
plt.xlabel('Nota Média de Avaliação (1.0 a 5.0)', fontsize=12)
plt.ylabel('Densidade de Probabilidade', fontsize=12)
plt.tight_layout()
plt.show()

# ==========================================
# 7. GRÁFICO DE REGRESSÃO
# ==========================================
plt.figure(figsize=(10, 6))
sns.regplot(
    x='Nota',
    y='Preço',
    data=df,
    color='#e74c3c',
    scatter_kws={'alpha': 0.4, 'color': '#34495e'},
    line_kws={'linewidth': 2}
)
plt.title('Regressão Linear: Relação entre Nota e Preço Praticado', fontsize=14, fontweight='bold')
plt.xlabel('Nota de Avaliação', fontsize=12)
plt.ylabel('Preço (R$)', fontsize=12)
plt.tight_layout()
plt.show()