
---

# 📊 COVID-19 EM PERNAMBUCO

## Dados Observados e Modelos Epidemiológicos (SIR / SEIR / SEIRD / SEIRV)

Este repositório apresenta um **dashboard interativo em Streamlit** para análise da evolução da **COVID-19 no estado de Pernambuco**, integrando **dados epidemiológicos reais (2020–2025)** com **modelos matemáticos compartimentais** amplamente utilizados em epidemiologia.

O sistema permite visualizar **dados observados**, **projeções epidemiológicas** e a **comparação entre dados reais e modelos**, cobrindo **todo o período disponível nos arquivos**, sem cortes temporais implícitos.
Site: https://epidemiologia20252-hsjay4nhebwduqnnkvnzks.streamlit.app/

---

## 🎯 Objetivos do Projeto

* Analisar a evolução temporal da COVID-19 em Pernambuco
* Aplicar modelos epidemiológicos compartimentais:

  * **SIR**
  * **SEIR**
  * **SEIRD**
  * **SEIRV**
* Comparar dados observados com simulações epidemiológicas
* Fornecer uma ferramenta visual clara para apoio a estudos acadêmicos
* Garantir reprodutibilidade, transparência e rigor metodológico

---

## 🗂️ Estrutura do Repositório

```text
├── app.py                         # Aplicação Streamlit
├── covid_pe_seir_ready.parquet    # Dados epidemiológicos observados (PE)
├── cache.parquet                  # Resultados dos modelos epidemiológicos
├── requirements.txt               # Dependências do projeto
└── README.md                      # Documentação do projeto
```

---

## 📁 Descrição dos Arquivos de Dados

### 🔹 `covid_pe_seir_ready.parquet`

Base de dados **pré-processada**, contendo apenas **registros do estado de Pernambuco (PE)**.

Principais variáveis:

* `date` – Data do registro
* `municipio` – Município de Pernambuco
* `new_cases` – Casos novos diários
* `cum_cases` – Casos acumulados
* `I_est` – Estimativa de infectantes
* `population` – População estimada

---

### 🔹 `cache.parquet`

Arquivo de **cache computacional**, contendo os resultados pré-calculados dos modelos epidemiológicos.

Principais variáveis:

* `date` – Data da simulação
* `municipio` – Município
* `modelo` – Tipo de modelo (`SIR`, `SEIR`, `SEIRD`, `SEIRV`)
* `S`, `E`, `I`, `R`, `D`, `V` – Compartimentos epidemiológicos

Este arquivo é utilizado para:

* Acelerar o carregamento do app
* Evitar reprocessamento pesado no Streamlit
* Garantir consistência entre execuções

---

## 🧮 Modelos Epidemiológicos Implementados

| Modelo    | Descrição                              |
| --------- | -------------------------------------- |
| **SIR**   | Suscetíveis – Infectados – Recuperados |
| **SEIR**  | Inclui período de incubação (Expostos) |
| **SEIRD** | Inclui óbitos                          |
| **SEIRV** | Inclui vacinação                       |

Os modelos seguem formulações clássicas da literatura epidemiológica, com parâmetros estimados previamente e armazenados no arquivo de cache.

---

## 📊 Funcionalidades do Dashboard

* Seleção de **município** (ou todo o estado)
* Seleção de **modelo epidemiológico**
* Visualização de:

  * Casos diários
  * Casos acumulados
  * Estimativa de infectantes
  * Evolução dos compartimentos epidemiológicos
  * Proporção da população por compartimento
  * Comparação **Observado × Modelo**
* **Período completo automático** (todo o intervalo disponível nos arquivos)
* Datas no **formato brasileiro (DD/MM/AAAA)**

---

## 🖥️ Como Executar Localmente

### 1️⃣ Criar ambiente virtual (opcional, recomendado)

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 2️⃣ Instalar dependências

```bash
pip install -r requirements.txt
```

### 3️⃣ Executar o aplicativo

```bash
streamlit run app.py
```

O app estará disponível em:

```
http://localhost:8501
```

---

## 📦 Dependências Principais

* Python ≥ 3.9
* Streamlit
* Pandas
* Plotly
* PyArrow / FastParquet

---

## 🧠 Considerações Metodológicas

* O período analisado corresponde **integralmente aos dados disponíveis nos arquivos**
* Não há cortes temporais implícitos
* Todos os municípios pertencem exclusivamente ao estado de Pernambuco
* O uso de cache garante reprodutibilidade e desempenho

---

## 📜 Licença

Este projeto é disponibilizado para **fins acadêmicos e educacionais**.

---

## 👨‍🔬 Autor / Orientação

Projeto desenvolvido para fins acadêmico da UFRPE da matéria **Modelagem Computacional_Epidemiologia**, com foco na análise da COVID-19 no estado de Pernambuco.

---
