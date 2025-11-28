import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale

# -----------------------------
# 🇧🇷 Configurar normas brasileiras
# -----------------------------
try:
    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
except:
    locale.setlocale(locale.LC_ALL, '')

# Função para formatar números no padrão BR
def format_number_br(x):
    return locale.format_string("%.0f", x, grouping=True)

# Função para formatar datas
def format_date_br(date):
    return pd.to_datetime(date).strftime('%d/%m/%Y')


# -----------------------------
# 1. Carregar dados
# -----------------------------
file_path = r"D:\Dataset covidPE 2020-2024 (NOVO)\Codigo do Projeto\covidPE_2020_2024_tratado_FINAL.csv"
df = pd.read_csv(file_path, parse_dates=['data'], dayfirst=True)

print("Dimensão:", df.shape)
print(df.head())


# -----------------------------
# 2. Converter formato das datas (BR)
# -----------------------------
df['data_br'] = df['data'].dt.strftime('%d/%m/%Y')


# -----------------------------
# 3. Estatísticas básicas
# -----------------------------
print("\n📌 Estatísticas descritivas (casos):")
print(df['casos_novos'].describe())

print("\n📌 Estatísticas descritivas (óbitos):")
print(df['obitos_novos'].describe())


# -----------------------------
# 4. Casos totais por município
# -----------------------------
casos_municipio = df.groupby('municipio')['casos_novos'].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 7))
casos_municipio.head(15).plot(kind='bar', color="#1f77b4")

plt.title("15 Municípios com Mais Casos Acumulados — PE (2020–2024)", fontsize=16, weight='bold')
plt.ylabel("Total de Casos", fontsize=12)
plt.xlabel("Município", fontsize=12)
plt.xticks(rotation=45, ha='right')

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: format_number_br(x)))
plt.tight_layout()
plt.show()


# -----------------------------
# 5. Evolução diária dos casos em Pernambuco
# -----------------------------
casos_diarios = df.groupby('data')['casos_novos'].sum()

plt.figure(figsize=(14, 6))
plt.plot(casos_diarios.index, casos_diarios.values, linewidth=2, color="#d62728")

plt.title("Evolução Diária dos Casos de COVID-19 — Pernambuco", fontsize=16, weight='bold')
plt.ylabel("Casos por Dia", fontsize=12)
plt.xlabel("Data", fontsize=12)

# Formatando datas no eixo X
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(10))
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: format_date_br(casos_diarios.index[int(x)])) if len(casos_diarios) > 1 else "")

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: format_number_br(x)))

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# -----------------------------
# 6. Óbitos por Município (Top 15)
# -----------------------------
obitos_municipio = df.groupby('municipio')['obitos_novos'].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 7))
obitos_municipio.head(15).plot(kind='bar', color="#2ca02c")

plt.title("15 Municípios com Mais Óbitos Acumulados — PE (2020–2024)", fontsize=16, weight='bold')
plt.ylabel("Total de Óbitos", fontsize=12)
plt.xlabel("Município", fontsize=12)
plt.xticks(rotation=45, ha='right')

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: format_number_br(x)))

plt.tight_layout()
plt.show()
