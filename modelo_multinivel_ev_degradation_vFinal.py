# ==============================================================================
# PROJETO: MODELAGEM HIERÁRQUICA MULTINÍVEL - DEGRADAÇÃO DE BATERIAS DE VE (SoH)
# ==============================================================================

# In[1.1 - Importanto Pacotes e definindo Setup Padrão]
# ------------------------------------------------------------------------------
#   REPRODUCIBILITY & SETUP (Configuração Inicial e Sementes)
# ------------------------------------------------------------------------------
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Modelagem Estatística
import statsmodels.formula.api as smf

# Machine Learning & Diagnósticos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Remove limite de colunas e restrição de largura
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Fixando sementes para reproducibilidade
np.random.seed(42)
random.seed(42)

# Estilo visual padronizado
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# In[1.2 - Data Load]
# ------------------------------------------------------------------------------
#  DATA LOAD & EARLY CLEANING
# ------------------------------------------------------------------------------

# Carregamento do arquivo local
df = pd.read_csv("ev_battery_degradation_v1.csv")

# Remoção antecipada do Vehicle_ID (evita vazamento ou custo desnecessário de memória)
if 'Vehicle_ID' in df.columns:
    df = df.drop('Vehicle_ID', axis=1)

# Anonimização de Car_Model para grupos genéricos (A, B, C...)
if 'Car_Model' in df.columns:
    car_models = df['Car_Model'].unique()
    model_mapping = {model: chr(65 + i) for i, model in enumerate(car_models)}
    df['Car_Model_Generic'] = df['Car_Model'].map(model_mapping)
    df = df.drop('Car_Model', axis=1)

# In[1.3 - Perfil dos Dados]
# ------------------------------------------------------------------------------
#  DATA PROFILE SECTION
# ------------------------------------------------------------------------------
print("=" * 60)
print("                    DATA PROFILE SECTION                    ")
print("=" * 60)
print(f"Dataset Shape: {df.shape}")
print("\n--- Missing Values Count ---")
print(df.isnull().sum())
print("\n--- Data Types ---")
print(df.dtypes)
print("\n--- Data Stats ---")
print(df.describe())
print("=" * 60)

# In[1.4 - Definição do Projeto]
# ------------------------------------------------------------------------------
#  RESPONSE & GROUPING SPECIFICATION
# ------------------------------------------------------------------------------
print("""
DEFINIÇÃO DO ESCOPO DE MODELAGEM:
1. Target Contínuo: 'SoH_Percent' (State of Health da bateria, [0, 100%])
   - Modelado via Linear Mixed-Effects Model (smf.mixedlm)
     
2. Variável de Agrupamento Hierárquico (Nível 2): 'Car_Model_Generic'
   - Captura a heterogeneidade contextual (fabricação, BMS e arquitetura do veículo).
""")

# In[2.1 - EDA]
# ------------------------------------------------------------------------------
#  EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------------------------
print("\nGerando gráficos de Análise Exploratória (EDA)...")

# A) Distribuição da variável resposta SoH_Percent
plt.figure(figsize=(10, 5))
sns.histplot(df['SoH_Percent'], kde=True, bins=30, color='teal')
plt.title('Distribuição de SoH_Percent', fontsize=14)
plt.xlabel('SoH (%)')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()

# B) Scatter plots com retas de regressão por modelo de veículo
g = sns.lmplot(
    data=df,
    x='Total_Charging_Cycles',
    y='SoH_Percent',
    hue='Car_Model_Generic',
    palette='tab10',
    height=5,
    aspect=1.6,
    scatter_kws={'alpha': 0.3},
    legend=True
)

# Ajusta o título no FacetGrid
g.figure.suptitle('Regressão: Ciclos de Carga vs. SoH% por Modelo de Veículo', y=1.03)
# Força os marcadores da legenda a ficarem com opacidade 1.0 (visíveis)
if g._legend:
    for handle in g._legend.legend_handles:
        handle.set_alpha(1.0)
plt.show()

# C) Heatmap de correlação das variáveis numéricas
numeric_cols_raw = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(10, 7))
sns.heatmap(df[numeric_cols_raw].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlação das Variáveis Numéricas', fontsize=14)
plt.tight_layout()
plt.show()

# D) Countplots das variáveis categóricas (Battery_Status e Driving_Style)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(data=df, x='Battery_Status', ax=axes[0], palette='crest')
axes[0].set_title('Contagem por Battery Status')

sns.countplot(data=df, x='Driving_Style', ax=axes[1], palette='viridis')
axes[1].set_title('Contagem por Driving Style')
plt.tight_layout()
plt.show()

# In[2.2 - Estrutura dos Grupos e Diagnóstico de Nível 2]
# ------------------------------------------------------------------------------
# AVALIAÇÃO DA ESTRUTURA HIERÁRQUICA E ICC PRELIMINAR
# ------------------------------------------------------------------------------

print("\n--- [Nível 2] Contagem e Balanceamento por Modelo de Veículo ---")
cluster_counts = df['Car_Model_Generic'].value_counts()
print(cluster_counts)

# Gráfico de frequência das unidades de nível 2
plt.figure(figsize=(10, 4))
sns.countplot(data=df, y='Car_Model_Generic', order=cluster_counts.index, palette='crest')
plt.title('Distribuição de Amostras por Car_Model_Generic (Nível 2)')
plt.xlabel('Número de Veículos/Amostras')
plt.ylabel('Modelo')
plt.tight_layout()
plt.show()

# Cálculo do ICC PRELIMINAR (ANOVA unidirecional via OLS), só para a triagem
# inicial de In[2.2]: decidir, ainda na base completa e antes da divisão
# treino/teste, se vale a pena seguir com a modelagem multinível
# ICC = MSB - MSW / (MSB + (k - 1) * MSW)
mod_icc = ols('SoH_Percent ~ C(Car_Model_Generic)', data=df).fit()
anova_table = sm.stats.anova_lm(mod_icc, typ=1)

msb = anova_table.loc['C(Car_Model_Generic)', 'mean_sq']
msw = anova_table.loc['Residual', 'mean_sq']
k = len(df) / df['Car_Model_Generic'].nunique()  # tamanho médio aproximado do grupo

icc_preliminar_anova = (msb - msw) / (msb + (k - 1) * msw)
print(f"\nICC preliminar (ANOVA, base completa, n={len(df)}): {icc_preliminar_anova:.4f}")
print("Nota: ICC > 0.05 já justifica seguir com a abordagem multinível.")
print("Este valor é só um diagnóstico de triagem (estimador diferente, base completa,")
print("sem controle por preditores). O ICC OFICIAL do artigo é o do Modelo Nulo HLM2,")
print("estimado por máxima verossimilhança no conjunto de treino (In[4.1] adiante).")


# In[2.3 - Heterogeneidade de Interceptos e Inclinações]
# ------------------------------------------------------------------------------
# COMPARAÇÃO VISUAL DE INTERCEPTOS E INCLINAÇÕES ENTRE GRUPOS
# ------------------------------------------------------------------------------

# A) Variação do Intercepto (SoH médio por grupo)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.boxplot(data=df, x='Car_Model_Generic', y='SoH_Percent', ax=axes[0], palette='Set2')
axes[0].set_title('SoH% por Modelo de Veículo (Variação de Intercepto)')
axes[0].tick_params(axis='x', rotation=45)

sns.boxplot(data=df, x='Battery_Type', y='SoH_Percent', ax=axes[1], palette='Pastel1')
axes[1].set_title('SoH% por Tipo de Bateria')
plt.tight_layout()
plt.show()

# B) Variação de Inclinação (Random Slopes): FacetGrid por Car_Model_Generic
# Permite avaliar se a reta de degradação com ciclos de carga muda de inclinação por modelo
g = sns.lmplot(
    data=df,
    x='Total_Charging_Cycles',
    y='SoH_Percent',
    col='Car_Model_Generic',
    col_wrap=4,
    height=3.5,
    scatter_kws={'alpha': 0.25, 's': 15},
    line_kws={'color': 'darkred'}
)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Degradação (Ciclos vs. SoH%) por Modelo: Inspeção de Random Slopes')
plt.show()


# In[2.4 - Variáveis Categóricas e Testes de Hipótese]
# ------------------------------------------------------------------------------
# TESTES DE DIFERENÇA DE MÉDIAS/MEDIANAS PARA CATEGÓRICAS
# ------------------------------------------------------------------------------

# Verificação de relação entre Driving_Style e SoH_Percent
print("\n--- Estatísticas de SoH_Percent por Driving_Style ---")
print(df.groupby('Driving_Style')['SoH_Percent'].agg(['count', 'mean', 'std', 'median']))

# Teste de Kruskal-Wallis (não-paramétrico) para Driving_Style
driving_groups = [group['SoH_Percent'].values for _, group in df.groupby('Driving_Style')]
kw_stat, kw_p = stats.kruskal(*driving_groups)
print(f"\nTeste Kruskal-Wallis (Driving_Style vs. SoH): Estatística={kw_stat:.3f}, p-valor={kw_p:.4e}")

# Análise de Contingência / Possível vazamento: Battery_Status vs SoH_Percent
print("\n--- Resumo de SoH_Percent por Battery_Status ---")
print(df.groupby('Battery_Status')['SoH_Percent'].agg(['min', 'max', 'mean']))


# In[2.5 - Diagnóstico de Multicolinearidade (VIF)]
# ------------------------------------------------------------------------------
# AVALIAÇÃO DE COLINEARIDADE ENTRE AS VARIÁVEIS EXPLICATIVAS CONTÍNUAS
# ------------------------------------------------------------------------------

continuous_predictors = [
    'Battery_Capacity_kWh',
    'Vehicle_Age_Months',
    'Total_Charging_Cycles',
    'Avg_Temperature_C',
    'Fast_Charge_Ratio',
    'Avg_Discharge_Rate_C',
    'Internal_Resistance_Ohm'
]

# Preparação para VIF (removendo NaN se houver e adicionando constante)
vif_df = df[continuous_predictors].dropna()
X_vif = sm.add_constant(vif_df)

vif_data = pd.DataFrame()
vif_data["Variável"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

print("\n--- Variance Inflation Factor (VIF) ---")
print(vif_data[vif_data["Variável"] != 'const'].sort_values(by='VIF', ascending=False))
print("Nota: Valores de VIF > 5 (ou 10) indicam problemas de multicolinearidade.")

# In[2.6 - Feature Engineering e Reavaliação do VIF]
# ------------------------------------------------------------------------------
# CRIAÇÃO DA TAXA DE CICLOS POR MÊS E NOVO TESTE DE VIF
# ------------------------------------------------------------------------------

# 1. Criação da variável derivada (com proteção para eventual divisão por zero)
df['Cycles_Per_Month'] = np.where(
    df['Vehicle_Age_Months'] > 0,
    df['Total_Charging_Cycles'] / df['Vehicle_Age_Months'],
    0.0
)

print("--- Estatísticas Descritivas de Cycles_Per_Month ---")
print(df['Cycles_Per_Month'].describe().round(3))


# 2. Definição do novo conjunto de preditores contínuos
# Substituímos 'Total_Charging_Cycles' por 'Cycles_Per_Month'
# Removemos 'Internal_Resistance_Ohm' por ser uma variável endógena 
# simultânea da degradação
# Removemos 'Battery_Capacity_kWh' (e 'Battery_Type', descartado na Data Prep) porque
# ambas são constantes dentro de cada Car_Model_Generic: são variáveis de nível 2 e há
# apenas 5 grupos, o que torna o efeito estimável só por 5 pontos (limitação do estudo)
continuous_predictors_updated = [
    'Vehicle_Age_Months',
    'Cycles_Per_Month',
    'Avg_Temperature_C',
    'Fast_Charge_Ratio',
    'Avg_Discharge_Rate_C'
]

# 3. Cálculo do novo VIF
vif_df_updated = df[continuous_predictors_updated].dropna()
X_vif_updated = sm.add_constant(vif_df_updated)

vif_data_updated = pd.DataFrame()
vif_data_updated["Variável"] = X_vif_updated.columns
vif_data_updated["VIF"] = [
    variance_inflation_factor(X_vif_updated.values, i) 
    for i in range(X_vif_updated.shape[1])
]

print("\n--- Novo Variance Inflation Factor (VIF) com Cycles_Per_Month ---")
print(vif_data_updated[vif_data_updated["Variável"] != 'const'].sort_values(by='VIF', ascending=False))

# In[3.1 - Data Prep]
# ------------------------------------------------------------------------------
#  DATA PREPARATION: ENCODING, STRATIFIED SPLIT & SCALING (SEM DATA LEAKAGE)
# ------------------------------------------------------------------------------

df_prep = df.copy()

# 1. Descartar variáveis redundantes ou que geram data leakage
#    (Battery_Status é um limiar do próprio SoH; Battery_Type é constante por modelo de veículo)
df_prep.drop(columns=['Battery_Status', 'Battery_Type'], inplace=True, errors='ignore')

# 2. Preditores contínuos validados no VIF
numeric_predictors = continuous_predictors_updated

# 3. One-Hot Encoding apenas para Driving_Style (drop_first=True mantém uma classe base)
df_prep = pd.get_dummies(
    df_prep, 
    columns=['Driving_Style'], 
    drop_first=True, 
    dtype=int
)

# 4. Garantir Car_Model_Generic como categoria (Cluster Nível 2)
df_prep['Car_Model_Generic'] = df_prep['Car_Model_Generic'].astype('category')

print("Data Prep concluído com sucesso!")
print(f"Dimensões do DataFrame final: {df_prep.shape}")
print(f"Colunas disponíveis: {df_prep.columns.tolist()}")


# 5. Divisão estratificada pelo grupo de Nível 2 (Car_Model_Generic)
train_idx, test_idx = train_test_split(
    df_prep.index, 
    test_size=0.20, 
    random_state=42, 
    stratify=df_prep['Car_Model_Generic']
)

df_train = df_prep.loc[train_idx].copy().reset_index(drop=True)
df_test = df_prep.loc[test_idx].copy().reset_index(drop=True)

print(f"Tamanho Treino: {df_train.shape[0]} amostras | Tamanho Teste: {df_test.shape[0]} amostras")

# 6. Verificação de integridade da estratificação (proporções relativas)
print("\n--- Proporção de Car_Model_Generic (Treino vs Teste) ---")
check_split = pd.DataFrame({
    'Treino (%)': df_train['Car_Model_Generic'].value_counts(normalize=True) * 100,
    'Teste (%)': df_test['Car_Model_Generic'].value_counts(normalize=True) * 100
})
print(check_split.round(2))

# 7. Padronização (Z-Score): fit no treino e transform em ambos
scaler = StandardScaler()
df_train[numeric_predictors] = scaler.fit_transform(df_train[numeric_predictors])
df_test[numeric_predictors] = scaler.transform(df_test[numeric_predictors])

print("\nPadronização concluída com sucesso.")

# In[4.1 - Modelo Nulo: HLM2 vs. OLS e Teste de Razão de Verossimilhança]
# ------------------------------------------------------------------------------
#  MODELO NULO (UNCONDITIONAL MEANS MODEL) E BENCHMARK OLS NULO
# ------------------------------------------------------------------------------
print("""
ESPECIFICAÇÃO MATEMÁTICA DO MODELO NULO HLM2:
  Nível 1 (Veículo i no Modelo j):
    SoH_Percent[i,j] = β0[j] + ε[i,j],   ε[i,j] ~ Normal(0, σ²)

  Nível 2 (Intercepto Aleatório):
    β0[j] = γ00 + u0[j],                 u0[j] ~ Normal(0, τ00²)

  Modelo Combinado:
    SoH_Percent[i,j] = γ00 + u0[j] + ε[i,j]

MODELO BENCHMARK OLS NULO:
    SoH_Percent[i] = β0 + ε[i],          ε[i] ~ Normal(0, σ²)
""")

# ------------------------------------------------------------------------------
# 1. Ajuste dos Modelos Nulos (Fórmula: apenas intercepto '~ 1')
# ------------------------------------------------------------------------------
formula_nulo = "SoH_Percent ~ 1"

# A) Modelo OLS Nulo (ignora a estrutura hierárquica)
modelo_ols_nulo = smf.ols(formula=formula_nulo, data=df_train).fit()

# B) Modelo HLM2 Nulo (reml=False para permitir comparação direta de logLik via ML)
modelo_nulo_hlm2 = smf.mixedlm(
    formula=formula_nulo,
    data=df_train,
    groups=df_train['Car_Model_Generic']
).fit(reml=False)

print("=" * 80)
print("SUMÁRIO DO MODELO NULO HLM2 (EFEITOS MISTOS):")
print(modelo_nulo_hlm2.summary())
print("=" * 80)

# ------------------------------------------------------------------------------
# 2. Análise da Significância do Efeito Aleatório de Intercepto e ICC do Modelo
# ------------------------------------------------------------------------------
tau2 = float(modelo_nulo_hlm2.cov_re.iloc[0, 0])  # Variância entre grupos (Group Var)
sigma2 = float(modelo_nulo_hlm2.scale)             # Variância residual intra-grupo (Scale)
icc_oficial = tau2 / (tau2 + sigma2)               # ICC OFICIAL do artigo (ver nota abaixo)

print("\n--- DECOMPOSIÇÃO DE VARIÂNCIA DO HLM2 NULO ---")
print(f"Variância Entre Grupos (tau00² - Car_Model_Generic): {tau2:.4f}")
print(f"Variância Residual Intra-Grupo (sigma²):             {sigma2:.4f}")
print(f"ICC OFICIAL (Modelo Nulo HLM2, ML, treino n={len(df_train)}): {icc_oficial:.4f} ({icc_oficial*100:.2f}%)")
print("Este é o ICC citado no restante do artigo (Resumo, Resultados e Discussão).")
print(f"(O ICC preliminar de In[2.2], por ANOVA na base completa, foi {icc_preliminar_anova:.4f};")
print(" a diferença vem da amostra - base completa vs. treino - e do estimador - ANOVA vs. ML.)")

def teste_wald_variancia(modelo, nome_modelo):
    # Teste Z de Wald para os componentes de variância/covariância aleatória de um
    # modelo MixedLM (H0: o componente é igual a zero; H1: é diferente de zero).
    # Importante: "modelo.tvalues"/"modelo.pvalues" do statsmodels usam uma
    # parametrização interna (variância dividida pela escala residual) que NÃO
    # corresponde ao par Coef./Std.Err. impresso em modelo.summary() para as
    # linhas de variância. Por isso, o Z é recalculado aqui na MESMA escala do
    # summary(): a variância/covariância em "modelo.cov_re" dividida pelo
    # erro-padrão correspondente em "modelo.bse_re".
    bse_re = modelo.bse_re          # erros-padrão dos componentes, na escala do summary()
    cov_re = modelo.cov_re          # matriz de variâncias (diagonal) e covariâncias (fora da diagonal)
    linhas = []                     # acumula uma linha de resultado por componente
    for nome, se in bse_re.items():
        # "bse_re" nomeia cada componente como "<Var> Var" (diagonal) ou
        # "<A> x <B> Cov" (fora da diagonal); o nome localiza o valor
        # correspondente em "cov_re" sem depender da ordem das linhas
        if nome.endswith(" Cov"):
            a, b = nome[:-4].split(" x ")   # separa os dois efeitos aleatórios da covariância
            estimativa = cov_re.loc[a, b]   # covariância entre os dois efeitos
        else:
            a = nome[:-4]                   # remove o sufixo " Var" para obter o nome do efeito
            estimativa = cov_re.loc[a, a]   # variância do efeito (elemento da diagonal)
        z = estimativa / se                              # estatística Z de Wald
        p_valor = 2 * (1 - stats.norm.cdf(abs(z)))        # p-valor bicaudal (Normal padrão)
        linhas.append({
            'Componente': nome, 'Estimativa': estimativa, 'EP': se, 'Z': z, 'p-valor': p_valor
        })
    df_wald = pd.DataFrame(linhas)
    print(f"\n--- TESTE Z DE WALD: COMPONENTES DE VARIÂNCIA ALEATÓRIA ({nome_modelo}) ---")
    print(df_wald.to_string(index=False))
    for _, linha in df_wald.iterrows():
        # H0 é rejeitada (efeito aleatório estatisticamente significante) quando p <= 5%
        veredito = "significante a 5%" if linha['p-valor'] <= 0.05 else "NÃO significante a 5%"
        print(f"  H0: {linha['Componente']} = 0  ->  Z = {linha['Z']:.3f}, p-valor = {linha['p-valor']:.4e} ({veredito})")
    return df_wald

# Aplicação ao Modelo Nulo: testa se a variância entre modelos de veículo (tau00²) é zero
wald_nulo = teste_wald_variancia(modelo_nulo_hlm2, "Modelo Nulo")

# ------------------------------------------------------------------------------
# 3. Tabela Comparativa: OLS Nulo vs. HLM2 Nulo
# ------------------------------------------------------------------------------
df_comparacao = pd.DataFrame({
    'Métrica': ['Log-Likelihood (LLF)', 'AIC', 'BIC', 'Nº Parâmetros', 'Graus de Liberdade Residual'],
    'OLS Nulo': [
        modelo_ols_nulo.llf, 
        modelo_ols_nulo.aic, 
        modelo_ols_nulo.bic, 
        modelo_ols_nulo.df_model + 1,  # intercepto
        modelo_ols_nulo.df_resid
    ],
    'HLM2 Nulo': [
        modelo_nulo_hlm2.llf, 
        modelo_nulo_hlm2.aic, 
        modelo_nulo_hlm2.bic, 
        len(modelo_nulo_hlm2.params) + 1,  # fixos + variâncias
        modelo_nulo_hlm2.df_resid
    ]
})
print("\n--- COMPARAÇÃO ENTRE OLS NULO E HLM2 NULO ---")
print(df_comparacao.to_string(index=False))

# ------------------------------------------------------------------------------
# 4. Gráfico de Comparação Visual das Log-Likelihoods
# ------------------------------------------------------------------------------
df_llf = pd.DataFrame({
    'modelo': ['OLS Nulo', 'HLM2 Nulo'],
    'loglik': [modelo_ols_nulo.llf, modelo_nulo_hlm2.llf]
})

fig, ax = plt.subplots(figsize=(12, 6))

c = ['dimgray', 'darkslategray']

ax1 = ax.barh(df_llf.modelo, df_llf.loglik, color=c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=28, fmt='%.2f')
ax.set_ylabel("Modelo Proposto", fontsize=20)
ax.set_xlabel("LogLik", fontsize=20)
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='x', labelsize=16)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# Funções auxiliares: p-valor de LRT quando o parâmetro testado é uma
# variância (ou envolve uma), que sob H0 fica na FRONTEIRA do espaço de
# parâmetros (variância não pode ser negativa). Nesse caso, a estatística de
# teste NÃO segue a qui-quadrado usual; usar qui-quadrado "pura" (com todos os
# graus de liberdade) sempre dá um p-valor conservador (maior que o correto)
# ------------------------------------------------------------------------------
def p_valor_lrt_variancia_unica(lr_stat):
    # Teste que adiciona UM único componente de variância (ex.: intercepto
    # aleatório sobre um modelo sem nenhum efeito aleatório). Sob H0, a
    # distribuição assintótica correta é uma mistura 50/50 entre um ponto de
    # massa em zero e qui-quadrado(1) (Self & Liang, 1987); como o ponto de
    # massa em zero nunca "sobrevive" além de zero, a sobrevivência da mistura
    # é simplesmente metade da sobrevivência de qui-quadrado(1)
    return 0.5 * stats.chi2.sf(lr_stat, df=1)

def p_valor_lrt_inclinacao_aleatoria(lr_stat):
    # Teste que adiciona uma inclinação aleatória (variância + covariância) a
    # um modelo que já tem intercepto aleatório. Sob H0 (variância da
    # inclinação = 0), um dos dois graus de liberdade corresponde a uma
    # variância na fronteira do espaço de parâmetros, e o outro, a uma
    # covariância livre (sem fronteira); a distribuição assintótica correta é
    # uma mistura 50/50 entre qui-quadrado(1) e qui-quadrado(2), não
    # qui-quadrado(2) "puro" (Stram & Lee, 1994)
    return 0.5 * stats.chi2.sf(lr_stat, df=1) + 0.5 * stats.chi2.sf(lr_stat, df=2)

# ------------------------------------------------------------------------------
# 5. Execução do Teste de Razão de Verossimilhança (Likelihood Ratio Test)
# ------------------------------------------------------------------------------
def lrtest(modelos):
    # Desempacota os dois modelos passados na lista
    modelo_1, modelo_2 = modelos[0], modelos[-1]

    llk_1 = modelo_1.llf
    llk_2 = modelo_2.llf

    # Estatística de teste: -2 * (LL_restrito - LL_completo)
    LR_statistic = -2 * (llk_1 - llk_2)
    p_val_ingenuo = stats.chi2.sf(LR_statistic, df=1)   # qui-quadrado(1) pura, ignora a fronteira
    p_val = p_valor_lrt_variancia_unica(LR_statistic)   # p-valor correto (mistura 50/50)

    print("\nLikelihood Ratio Test:")
    print(f"-2.(LL0-LLm): {round(LR_statistic, 2)}")
    print(f"p-value (qui-quadrado(1) pura, ingênuo): {p_val_ingenuo:.4e}")
    print(f"p-value (mistura 0.5*qui-quadrado(0) + 0.5*qui-quadrado(1), correto): {p_val:.4e}")
    print("")
    print("==================Result========================")
    if p_val <= 0.05:
        print("H1: Different models, favoring the one with the highest Log-Likelihood")
        print("    -> A variância aleatória entre modelos é estatisticamente significante.")
        print("    -> O modelo multinível supera estatisticamente o OLS tradicional.")
    else:
        print("H0: Models with log-likelihoods that are not statistically different at 95% confidence level")
        print("    -> Não há evidência suficiente para justificar a modelagem multinível frente ao OLS.")
    print("=" * 48)

# Execução do teste comparando OLS nulo com HLM2 nulo
lrtest([modelo_ols_nulo, modelo_nulo_hlm2])

# In[4.2 - Random Intercept Model]
# ------------------------------------------------------------------------------
# 2. HLM2 COM INTERCEPTOS ALEATÓRIOS E PREDITORES FIXOS (NÍVEL 1)
# ------------------------------------------------------------------------------
print("""
ESPECIFICAÇÃO MATEMÁTICA DO MODELO (RANDOM INTERCEPT):
  Nível 1 (Veículo i no Modelo j):
    SoH_Percent[i,j] = β0[j] + β1*Cycles_Per_Month[i,j] + β2*Vehicle_Age_Months[i,j] +
                       β3*Avg_Temperature_C[i,j] + β4*Fast_Charge_Ratio[i,j] +
                       β5*Avg_Discharge_Rate_C[i,j] +
                       Σ βk*Driving_Style_Dummy[k,i,j] + ε[i,j]
    com ε[i,j] ~ Normal(0, σ²)

  Nível 2 (Intercepto Aleatório entre Modelos j):
    β0[j] = γ00 + u0[j]
    com u0[j] ~ Normal(0, τ00²)
""")

# 1. Identificar automaticamente as colunas de dummies de Driving_Style geradas no prep
driving_dummies = [c for c in df_train.columns if c.startswith('Driving_Style_')]

# 2. Lista completa de preditores fixos (numéricos + dummies)
fixed_predictors = numeric_predictors + driving_dummies
formula_ri = "SoH_Percent ~ " + " + ".join(fixed_predictors)

print("Fórmula ajustada para o Modelo Random Intercept:")
print(formula_ri)

# 3. Ajuste do modelo via MixedLM (reml=False para permitir comparação de logLik/AIC via ML)
modelo_ri_hlm2 = smf.mixedlm(
    formula=formula_ri,
    data=df_train,
    groups=df_train['Car_Model_Generic']
).fit(reml=False)

print("\n" + "=" * 80)
print("SUMÁRIO DO MODELO RANDOM INTERCEPT (HLM2):")
print(modelo_ri_hlm2.summary())
print("=" * 80)

# ------------------------------------------------------------------------------
# 4. Decomposição de Variância e R² Proporcional (Snijders & Bosker)
# ------------------------------------------------------------------------------
# Variâncias do Modelo Nulo (obtidas no In[4.1])
tau2_nulo = float(modelo_nulo_hlm2.cov_re.iloc[0, 0])
sigma2_nulo = float(modelo_nulo_hlm2.scale)

# Variâncias do Modelo Random Intercept
tau2_ri = float(modelo_ri_hlm2.cov_re.iloc[0, 0])
sigma2_ri = float(modelo_ri_hlm2.scale)

# Redução proporcional da variância total (R² Multinível de Snijders & Bosker)
r2_snijders = 1 - ((sigma2_ri + tau2_ri) / (sigma2_nulo + tau2_nulo))

# Redução da variância residual intra-grupo (Nível 1)
r2_nivel_1 = 1 - (sigma2_ri / sigma2_nulo)

print("\n--- DECOMPOSIÇÃO E REDUÇÃO DE VARIÂNCIA ---")
print(f"Variância Residual Intra-Grupo Nulo (sigma²):     {sigma2_nulo:.4f}")
print(f"Variância Residual Intra-Grupo Atual (sigma²):    {sigma2_ri:.4f}")
print(f"R² Nível 1 (Redução da variância residual):       {r2_nivel_1:.4f} ({r2_nivel_1*100:.2f}%)")
print(f"R² Total Multinível (Snijders & Bosker):          {r2_snijders:.4f} ({r2_snijders*100:.2f}%)")

# ------------------------------------------------------------------------------
# 5. Comparativo de Ajuste: Modelo Nulo vs. Random Intercept
# ------------------------------------------------------------------------------
df_comp_ri = pd.DataFrame({
    'Métrica': ['Log-Likelihood', 'AIC', 'BIC'],
    'HLM2 Nulo': [modelo_nulo_hlm2.llf, modelo_nulo_hlm2.aic, modelo_nulo_hlm2.bic],
    'HLM2 Random Intercept': [modelo_ri_hlm2.llf, modelo_ri_hlm2.aic, modelo_ri_hlm2.bic]
})
print("\n--- COMPARAÇÃO DE CRITÉRIOS DE AJUSTE ---")
print(df_comp_ri.to_string(index=False))

# ------------------------------------------------------------------------------
# 6. Teste de Razão de Verossimilhança: Nulo vs. Random Intercept
# ------------------------------------------------------------------------------
# Grau de liberdade = número de preditores adicionados
# Teste sobre EFEITOS FIXOS (não sobre variância): ambos os modelos têm a mesma
# estrutura de efeito aleatório (só o intercepto), então não há parâmetro na
# fronteira do espaço paramétrico aqui, e a qui-quadrado(df_diff) usual é correta
df_diff = len(fixed_predictors)
lr_stat_ri = -2 * (modelo_nulo_hlm2.llf - modelo_ri_hlm2.llf)
p_val_ri = stats.chi2.sf(lr_stat_ri, df=df_diff)

print("\n--- TESTE LR (NULO vs. RANDOM INTERCEPT) ---")
print(f"-2*(LL_nulo - LL_ri): {lr_stat_ri:.2f} (gl = {df_diff})")
print(f"p-valor: {p_val_ri:.4e}")
if p_val_ri <= 0.05:
    print("Resultado: A inclusão dos preditores fixos melhora o ajuste de forma estatisticamente significante.")

# In[4.3 - Eliminação Parcimoniosa (Remoção de Avg_Discharge_Rate_C)]
# ------------------------------------------------------------------------------
# AJUSTE DO MODELO REDUZIDO E VALIDAÇÃO VIA CRITÉRIOS DE INFORMAÇÃO E TESTE LR
# ------------------------------------------------------------------------------
# 1. Nova lista de preditores sem a variável não significante
fixed_predictors_red = [p for p in fixed_predictors if p != 'Avg_Discharge_Rate_C']
formula_ri_red = "SoH_Percent ~ " + " + ".join(fixed_predictors_red)

# 2. Ajuste do modelo reduzido
modelo_ri_reduzido = smf.mixedlm(
    formula=formula_ri_red,
    data=df_train,
    groups=df_train['Car_Model_Generic']
).fit(reml=False)

# 3. Comparação de AIC e BIC
df_comp_parsimonia = pd.DataFrame({
    'Métrica': ['Log-Likelihood', 'AIC', 'BIC'],
    'Modelo Completo': [modelo_ri_hlm2.llf, modelo_ri_hlm2.aic, modelo_ri_hlm2.bic],
    'Modelo Reduzido (Sem Discharge)': [modelo_ri_reduzido.llf, modelo_ri_reduzido.aic, modelo_ri_reduzido.bic]
})
print("--- COMPARAÇÃO: MODELO COMPLETO vs. REDUZIDO ---")
print(df_comp_parsimonia.to_string(index=False))

# 4. Teste de Razão de Verossimilhança (1 grau de liberdade)
# Teste sobre UM EFEITO FIXO (Avg_Discharge_Rate_C), não sobre variância: sem
# fronteira no espaço de parâmetros, então a qui-quadrado(1) usual é correta
lr_stat_red = -2 * (modelo_ri_reduzido.llf - modelo_ri_hlm2.llf)
p_val_red = stats.chi2.sf(lr_stat_red, df=1)

print(f"\nTeste LR entre Completo e Reduzido: estatística={lr_stat_red:.3f}, p-valor={p_val_red:.4f}")
if p_val_red > 0.05:
    print("Conclusão: A remoção de Avg_Discharge_Rate_C NÃO prejudicou o ajuste.")
    print("Recomendação: Adotar o modelo reduzido como definitivo.")
else:
    print("Conclusão: A variável contribui significantemente para o ajuste.")

# Aplicação ao Modelo de Nível 1 definitivo (reduzido): testa se a variância
# entre modelos de veículo (tau00²), já controlando pelos preditores fixos, é zero
wald_ri = teste_wald_variancia(modelo_ri_reduzido, "Modelo de Nível 1 (Reduzido)")



# In[4.4 - Random Slopes Model (Convergência Robusta via REML + Powell)]
# ------------------------------------------------------------------------------
# HLM2 COM INTERCEPTOS E INCLINAÇÕES ALEATÓRIAS SEM INSTABILIDADE NUMÉRICA
# ------------------------------------------------------------------------------

print("""
ESPECIFICAÇÃO MATEMÁTICA DO MODELO COM RANDOM SLOPES:
  Nível 1 (Veículo i no Modelo j):
    SoH_Percent[i,j] = β0[j] + β1[j]*Cycles_Per_Month[i,j] + Σ βk*Xk[i,j] + ε[i,j]

  Nível 2 (Variação de Intercepto e Inclinação por Car_Model_Generic):
    β0[j] = γ00 + u0[j]   (Intercepto médio da frota + desvio do modelo j)
    β1[j] = γ10 + u1[j]   (Taxa média de degradação por ciclo + sensibilidade do modelo j)

  Estrutura de Covariância dos Efeitos Aleatórios:
    [u0[j], u1[j]]' ~ Normal(0, [ [tau00², tau01], [tau01, tau10²] ])
""")

print("Ajustando Modelo com Random Slopes via REML + Powell...")

# 1. Reestimar o modelo de Random Intercept com REML para comparação direta via LR
modelo_ri_reml = smf.mixedlm(
    formula=formula_ri_red,
    data=df_train,
    groups=df_train['Car_Model_Generic']
).fit(reml=True, method='powell')

# 2. Ajuste do Random Slopes com REML e otimizador Powell
modelo_rs_hlm2 = smf.mixedlm(
    formula=formula_ri_red,
    data=df_train,
    groups=df_train['Car_Model_Generic'],
    re_formula="~ Cycles_Per_Month"
).fit(reml=True, method='powell')

# ------------------------------------------------------------------------------
# Diagnóstico do ajuste quase singular (ConvergenceWarning: "Hessian matrix...
# not positive definite"). Com apenas 5 grupos de nível 2, a matriz de
# covariância 2x2 (intercepto x inclinação) tem 3 parâmetros livres estimados
# a partir de essencialmente 5 pontos; a correlação entre os dois efeitos
# aleatórios costuma ficar mal identificada e é empurrada para a fronteira do
# espaço de matrizes válidas (correlação perto de 1 em valor absoluto). Isso NÃO
# invalida as demais estimativas do modelo, mas é registrado aqui como
# diagnóstico e reportado no artigo como limitação (ver HISTORICO.md)
autovalores_cov_re = np.linalg.eigvalsh(modelo_rs_hlm2.cov_re.values)   # devem ser >= 0 numa matriz de covariância válida
var_intercepto = modelo_rs_hlm2.cov_re.iloc[0, 0]                       # tau00² (variância do intercepto aleatório)
var_inclinacao = modelo_rs_hlm2.cov_re.iloc[1, 1]                       # tau10² (variância da inclinação aleatória)
cov_intercepto_inclinacao = modelo_rs_hlm2.cov_re.iloc[0, 1]            # tau01 (covariância entre os dois)
correlacao_intercepto_inclinacao = cov_intercepto_inclinacao / np.sqrt(var_intercepto * var_inclinacao)

print("\n--- DIAGNÓSTICO: AJUSTE QUASE SINGULAR DO RANDOM SLOPES ---")
print(f"Autovalores de cov_re (2x2): {autovalores_cov_re}")
print(f"Correlação implícita entre intercepto e inclinação aleatórios: {correlacao_intercepto_inclinacao:.8f}")
print("Nota: com apenas 5 grupos de nível 2, a covariância entre os dois efeitos")
print("aleatórios é mal identificada e o otimizador a empurra para a fronteira do")
print("espaço de matrizes de covariância válidas (correlação perto de 1 em valor")
print("absoluto, menor autovalor perto de zero) - fenômeno conhecido na literatura")
print("de modelos mistos como 'ajuste quase singular' (singular fit). Testado com")
print("quatro otimizadores (Powell, LBFGS, CG, BFGS): todos convergem para o mesmo")
print("ponto quase degenerado, confirmando que não é um artefato do otimizador.")

print("\n" + "=" * 80)
print("SUMÁRIO DO MODELO RANDOM SLOPES CONVERGIDO (HLM2):")
print(modelo_rs_hlm2.summary())
print("=" * 80)

# Aplicação ao Modelo de Nível 2 (random slopes): testa se a variância do
# intercepto, a variância da inclinação de Cycles_Per_Month e a covariância
# entre elas são iguais a zero, cada uma isoladamente
wald_rs = teste_wald_variancia(modelo_rs_hlm2, "Modelo de Nível 2 (Random Slopes)")
print("Nota: com apenas 5 grupos de nível 2, o teste Z de Wald tem baixo poder")
print("para componentes de variância; o teste de razão de verossimilhança abaixo")
print("(gl = 2) é a evidência estatisticamente mais robusta para este modelo.")

# ------------------------------------------------------------------------------
# 3. Teste de Razão de Verossimilhança: Random Intercept vs. Random Slopes (REML)
# ------------------------------------------------------------------------------
# Diferença de 2 graus de liberdade: variância do slope (tau10²) e covariância (tau01)
lr_stat_rs = -2 * (modelo_ri_reml.llf - modelo_rs_hlm2.llf)
p_val_rs_ingenuo = stats.chi2.sf(lr_stat_rs, df=2)          # qui-quadrado(2) pura, ignora a fronteira
p_val_rs = p_valor_lrt_inclinacao_aleatoria(lr_stat_rs)      # p-valor correto (mistura 50/50)

print("\n--- TESTE LR (RANDOM INTERCEPT vs. RANDOM SLOPES - REML) ---")
print(f"-2*(LL_ri - LL_rs): {lr_stat_rs:.2f} (gl = 2)")
print(f"p-valor (qui-quadrado(2) pura, ingênuo): {p_val_rs_ingenuo:.4e}")
print(f"p-valor (mistura 0.5*qui-quadrado(1) + 0.5*qui-quadrado(2), correto): {p_val_rs:.4e}")
if p_val_rs <= 0.05:
    print("Resultado: A inclinação aleatória em Cycles_Per_Month é ESTATISTICAMENTE SIGNIFICANTE.")
    print("           -> O modelo com Random Slopes supera o modelo de apenas Intercepto Aleatório.")
else:
    print("Resultado: O modelo mais simples (apenas Random Intercept) é preferível.")

# ------------------------------------------------------------------------------
# 4. Extração dos Efeitos Aleatórios por Modelo (BLUPs)
# ------------------------------------------------------------------------------
re_dict = modelo_rs_hlm2.random_effects
df_blups = pd.DataFrame(re_dict).T
df_blups.columns = ['u0_Intercepto', 'u1_Slope_Cycles']

print("\n--- EFEITOS ALEATÓRIOS ESTIMADOS POR MODELO (BLUPs) ---")
print(df_blups.round(4))

# In[4.5 - Benchmark Global: Comparativo dos 5 Modelos e Log-Likelihood]
# ------------------------------------------------------------------------------
# AJUSTE DO OLS TRADICIONAL E GRÁFICO COMPARATIVO COMPLETO (DO NULO AO HLM2 FINAL)
# ------------------------------------------------------------------------------

# 1. Ajuste do OLS Tradicional (Pooled OLS) com as mesmas covariáveis do HLM2
modelo_ols_tradicional = smf.ols(
    formula=formula_ri_red,
    data=df_train
).fit()

# 2. Reajuste do Random Slopes por Máxima Verossimilhança (ML), apenas para comparações
#    O LogLik do REML (modelo_rs_hlm2) não é comparável ao LogLik por ML dos demais modelos,
#    pois o REML calcula a verossimilhança dos resíduos e não dos dados. As estimativas
#    finais continuam sendo as do modelo REML (modelo_rs_hlm2)
modelo_rs_ml = smf.mixedlm(
    formula=formula_ri_red,                       # mesmos efeitos fixos do random intercept reduzido
    data=df_train,                                # mesmo conjunto de treino
    groups=df_train['Car_Model_Generic'],         # agrupamento por modelo de veículo (nível 2)
    re_formula="~ Cycles_Per_Month"               # intercepto e inclinação aleatórios em ciclos
).fit(reml=False, method='powell')                # ML com Powell, como no ajuste REML

print("=" * 80)
print("BENCHMARK GLOBAL DE AJUSTE (LOG-LIKELIHOOD, TODOS POR ML):")
print(f"1. OLS Nulo:                 {modelo_ols_nulo.llf:.2f}")
print(f"2. HLM2 Nulo:                {modelo_nulo_hlm2.llf:.2f}")
print(f"3. OLS Tradicional:          {modelo_ols_tradicional.llf:.2f}")
print(f"4. HLM2 Nível 1 (Intercept): {modelo_ri_reduzido.llf:.2f}")
print(f"5. HLM2 Final (Random Slope):{modelo_rs_ml.llf:.2f}")
print(f"   (LogLik REML do HLM2 Final, não comparável: {modelo_rs_hlm2.llf:.2f})")
print("=" * 80)

# ------------------------------------------------------------------------------
# 3. Testes de Razão de Verossimilhança da Progressão (sempre ML contra ML)
# ------------------------------------------------------------------------------
# A) OLS Tradicional vs. HLM2 Nível 1 (Justifica intercepto aleatório com covariáveis)
#    1 componente de variância novo (intercepto aleatório): usar a mistura 50/50
lr_stat_cov = -2 * (modelo_ols_tradicional.llf - modelo_ri_reduzido.llf)
p_val_cov = p_valor_lrt_variancia_unica(lr_stat_cov)

# B) HLM2 Nível 1 vs. HLM2 Final (Justifica a inclinação aleatória em ciclos), ambos por ML
#    2 graus de liberdade novos (variância do slope + covariância): usar a mistura chi2(1)/chi2(2)
lr_stat_final = -2 * (modelo_ri_reduzido.llf - modelo_rs_ml.llf)
p_val_final = p_valor_lrt_inclinacao_aleatoria(lr_stat_final)

print("\n--- TESTES DE RAZÃO DE VEROSSIMILHANÇA (LR TESTS) ---")
print("(p-valores corrigidos para a fronteira do espaço de parâmetros: ver In[4.1] e In[4.4])")
print(f"A) OLS Tradicional vs. HLM2 Nível 1: LR = {lr_stat_cov:.2f} (gl=1) | p-valor = {p_val_cov:.4e}")
print(f"B) HLM2 Nível 1 vs. HLM2 Final:      LR = {lr_stat_final:.2f} (gl=2) | p-valor = {p_val_final:.4e}")

# ------------------------------------------------------------------------------
# 4. Tabela Consolidada com os 5 Modelos Estimados (LogLik por ML)
# ------------------------------------------------------------------------------
df_llf_global = pd.DataFrame({
    'modelo': [
        'OLS Nulo', 
        'HLM2 Nulo', 
        'OLS Tradicional', 
        'HLM2 Nível 1',
        'HLM2 Final'
    ],
    'loglik': [
        modelo_ols_nulo.llf, 
        modelo_nulo_hlm2.llf, 
        modelo_ols_tradicional.llf, 
        modelo_ri_reduzido.llf,
        modelo_rs_ml.llf                          # versão ML, comparável aos demais
    ]
})

# ------------------------------------------------------------------------------
# 5. Gráfico no Padrão do Projeto Anterior (5 Barras Horizontais)
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))

# Paleta alternando tons de cinza/ardósia (idêntica ao padrão da sua referência)
cores = ['dimgray', 'darkslategray', 'dimgray', 'darkslategray', 'dimgray']

barras = ax.barh(df_llf_global['modelo'], df_llf_global['loglik'], color=cores, height=0.65)
ax.bar_label(barras, label_type='center', color='white', fontsize=26, fmt='%.1f')

ax.set_ylabel("Modelo Proposto", fontsize=20)
ax.set_xlabel("LogLik", fontsize=20)
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.grid(axis='x', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# In[5.1 - Diagnóstico dos Resíduos (Modelo Random Slopes)]
# ------------------------------------------------------------------------------
# AVALIAÇÃO DOS PRESSUPOSTOS: NORMALIDADE E HOMOCEDASTICIDADE
# ------------------------------------------------------------------------------

# 1. Extração dos resíduos condicionais (Nível 1) e valores ajustados do modelo final
residuos_l1 = modelo_rs_hlm2.resid
valores_ajustados = modelo_rs_hlm2.fittedvalues

# 2. Testes Estatísticos de Normalidade dos Resíduos
jb_stat, jb_pval = stats.jarque_bera(residuos_l1)
da_stat, da_pval = stats.normaltest(residuos_l1)

print("--- TESTES DE NORMALIDADE DOS RESÍDUOS (NÍVEL 1) ---")
print(f"Jarque-Bera:        Estatística = {jb_stat:.3f} | p-valor = {jb_pval:.4e}")
print(f"D'Agostino-Pearson: Estatística = {da_stat:.3f} | p-valor = {da_pval:.4e}")
print(f"Assimetria (Skewness): {stats.skew(residuos_l1):.4f}")
print(f"Curtose (Kurtosis):    {stats.kurtosis(residuos_l1):.4f}")

# 3. Painel Gráfico de Diagnóstico (Desempacotando eixos explicitamente)
fig, (ax_homo, ax_qq, ax_hist) = plt.subplots(1, 3, figsize=(18, 5))

# A) Homocedasticidade: Resíduos vs. Valores Ajustados
ax_homo.scatter(valores_ajustados, residuos_l1, alpha=0.25, color='darkslategray', s=15)
ax_homo.axhline(0, color='red', linestyle='--', linewidth=1.2)
ax_homo.set_title('Homocedasticidade: Resíduos vs. Ajustados (Random Slopes)', fontsize=12)
ax_homo.set_xlabel('Valores Ajustados (SoH% Previsto)', fontsize=11)
ax_homo.set_ylabel('Resíduos Condicionais (ε)', fontsize=11)
ax_homo.grid(True, linestyle=':', alpha=0.6)

# B) Q-Q Plot dos Resíduos de Nível 1
sm.qqplot(residuos_l1, line='45', fit=True, ax=ax_qq, color='darkslategray', alpha=0.3)
ax_qq.set_title('Q-Q Plot: Normalidade dos Resíduos (Nível 1)', fontsize=12)
ax_qq.grid(True, linestyle=':', alpha=0.6)

# C) Histograma e Densidade dos Resíduos
ax_hist.hist(residuos_l1, bins=40, density=True, alpha=0.6, color='teal', edgecolor='black')
x_vals = np.linspace(residuos_l1.min(), residuos_l1.max(), 100)
p_vals = stats.norm.pdf(x_vals, residuos_l1.mean(), residuos_l1.std())
ax_hist.plot(x_vals, p_vals, 'r-', linewidth=1.5, label='Normal Teórica')
ax_hist.set_title('Distribuição dos Resíduos vs. Normal', fontsize=12)
ax_hist.set_xlabel('Resíduos Condicionais (ε)', fontsize=11)
ax_hist.set_ylabel('Densidade', fontsize=11)
ax_hist.legend()
ax_hist.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# 4. Inspeção dos Efeitos Aleatórios Completos (u0j e u1j por modelo de veículo)
re_df = pd.DataFrame(modelo_rs_hlm2.random_effects).T
re_df.columns = ['u0_Intercepto', 'u1_Slope_Cycles']
print("\n--- EFEITOS ALEATÓRIOS ESTIMADOS (BLUPs) POR MODELO ---")
print(re_df.round(4))

# In[5.2 - Avaliação no Conjunto de Teste (Out-of-Sample - Random Slopes)]
# ------------------------------------------------------------------------------
# PREDIÇÕES MARGINAIS E CONDICIONAIS (u0j + u1j) NO DF_TEST E MÉTRICAS
# ------------------------------------------------------------------------------

# 1. Predição Marginal no Teste (Efeitos Fixos populacionais: X * beta)
y_pred_marginal = modelo_rs_hlm2.predict(df_test)

# 2. Predição Condicional Completa: Efeitos Fixos + Random Intercept (u0) + Random Slope (u1 * Ciclos)
u0_map = re_df['u0_Intercepto'].to_dict()
u1_map = re_df['u1_Slope_Cycles'].to_dict()

# Mapeia u0 e u1 para as linhas do df_test conforme o modelo do veículo
car_models_str = df_test['Car_Model_Generic'].astype(str)
u0_test = car_models_str.map(u0_map).astype(float)
u1_test = car_models_str.map(u1_map).astype(float)

# Equação condicional de efeitos mistos
y_pred_condicional = y_pred_marginal + u0_test + (u1_test * df_test['Cycles_Per_Month'])

y_test_real = df_test['SoH_Percent']

# 3. Cálculo das Métricas de Avaliação Fora da Amostra
metricas = {
    'Métrica': ['R² (Coef. Determinação)', 'RMSE (Erro Médio Quadrático)', 'MAE (Erro Médio Absoluto)'],
    'Predição Marginal (Populacional)': [
        r2_score(y_test_real, y_pred_marginal),
        np.sqrt(mean_squared_error(y_test_real, y_pred_marginal)),
        mean_absolute_error(y_test_real, y_pred_marginal)
    ],
    'Predição Condicional (Com u0j e u1j)': [
        r2_score(y_test_real, y_pred_condicional),
        np.sqrt(mean_squared_error(y_test_real, y_pred_condicional)),
        mean_absolute_error(y_test_real, y_pred_condicional)
    ]
}

df_metricas = pd.DataFrame(metricas)
print("--- DESEMPENHO NO CONJUNTO DE TESTE (MODELO RANDOM SLOPES) ---")
print(df_metricas.round(4).to_string(index=False))

# 4. Gráfico de Calibração: Real vs. Previsto no Teste
fig, (ax_marg, ax_cond) = plt.subplots(1, 2, figsize=(14, 6))

# A) Predição Marginal
ax_marg.scatter(y_test_real, y_pred_marginal, alpha=0.3, color='dimgray', s=15)
ax_marg.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--', lw=2)
ax_marg.set_title(f'Predição Marginal (Populacional)\nR² = {r2_score(y_test_real, y_pred_marginal):.4f}', fontsize=12)
ax_marg.set_xlabel('SoH Real (%)', fontsize=11)
ax_marg.set_ylabel('SoH Previsto (%)', fontsize=11)
ax_marg.grid(True, linestyle=':', alpha=0.6)

# B) Predição Condicional (Interceptos e Slopes Aleatórios)
ax_cond.scatter(y_test_real, y_pred_condicional, alpha=0.3, color='teal', s=15)
ax_cond.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--', lw=2)
ax_cond.set_title(f'Predição Condicional (Com Random Slopes)\nR² = {r2_score(y_test_real, y_pred_condicional):.4f}', fontsize=12)
ax_cond.set_xlabel('SoH Real (%)', fontsize=11)
ax_cond.set_ylabel('SoH Previsto (%)', fontsize=11)
ax_cond.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# In[6.1 - Forest Plot: Impacto dos Preditores no SoH (Modelo Random Slopes)]
# ------------------------------------------------------------------------------
# VISUALIZAÇÃO DE IMPACTO (FOREST PLOT COM INTERVALOS DE CONFIANÇA DE 95%)
# ------------------------------------------------------------------------------

# 1. Extração dos coeficientes e intervalos de confiança (95%) do modelo final
ci = modelo_rs_hlm2.conf_int()
ci.columns = ['ci_lower', 'ci_upper']

df_forest = pd.DataFrame({
    'Beta': modelo_rs_hlm2.params,
    'ci_lower': ci['ci_lower'],
    'ci_upper': ci['ci_upper']
})

# 2. Filtrar Intercepto e TODOS os parâmetros de variância/covariância aleatória
# O regex remove automaticamente 'Group Var', 'Cov' e 'Cycles_Per_Month Var'
plot_coefs = df_forest.loc[
    (~df_forest.index.isin(['Intercept'])) & 
    (~df_forest.index.str.contains('Var|Cov', regex=True))
].sort_values(by='Beta')

# 3. Construção do Forest Plot
fig, ax = plt.subplots(figsize=(11, 6))

y_pos = np.arange(len(plot_coefs))
x_vals = plot_coefs['Beta']
err_lower = x_vals - plot_coefs['ci_lower']
err_upper = plot_coefs['ci_upper'] - x_vals

# Cores: vermelho para variáveis que aceleram a perda de SoH, verde para as que preservam
cores = ['crimson' if b < 0 else 'forestgreen' for b in x_vals]

# Barras de erro (Intervalo de Confiança a 95%)
ax.errorbar(
    x_vals, 
    y_pos, 
    xerr=[err_lower, err_upper], 
    fmt='none', 
    ecolor='gray', 
    elinewidth=2, 
    capsize=4,
    zorder=2
)

# Pontos centrais dos Betas
ax.scatter(x_vals, y_pos, color=cores, s=90, zorder=3)

# Rótulos com os valores numéricos de Beta ao lado de cada ponto
for y, b in zip(y_pos, x_vals):
    ha_pos = 'right' if b < 0 else 'left'
    deslocamento = -0.06 if b < 0 else 0.06
    ax.text(
        b + deslocamento, 
        y, 
        f"{b:.3f}", 
        va='center', 
        ha=ha_pos, 
        fontsize=10, 
        fontweight='bold',
        color='black'
    )

# Linha de referência nula (Beta = 0)
ax.axvline(0, color='black', linestyle='--', linewidth=1.2)

# Formatação dos Eixos e Títulos
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_coefs.index, fontsize=11)
# Margem vertical ajustada para evitar cortes
ax.set_ylim(-0.5, len(plot_coefs) + 0.6)

ax.set_title('Impacto dos Preditores no SoH% (Forest Plot - Modelo Random Slopes HLM2)', fontsize=13)
ax.set_xlabel('Variação Marginal no SoH (%) por Desvio Padrão do Preditor Padronizado', fontsize=11)
ax.grid(axis='x', linestyle=':', alpha=0.6)

# Anotações explicativas com espaçamento adequado no topo
ax.text(
    x_vals.min() * 0.95, 
    len(plot_coefs) - 0.2, 
    '← Acelera a degradação', 
    color='crimson', 
    fontsize=11, 
    fontweight='bold', 
    ha='left'
)
ax.text(
    x_vals.max() * 0.45, 
    len(plot_coefs) - 0.2, 
    'Preserva a bateria →', 
    color='forestgreen', 
    fontsize=11, 
    fontweight='bold', 
    ha='left'
)

plt.tight_layout()
plt.show()