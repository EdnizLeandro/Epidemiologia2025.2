---

# 📊 COVID-19 EM PERNAMBUCO

## Dados Observados e Modelos Epidemiológicos (SIR / SEIR / SEIRD / SEIRV)

Este repositório apresenta um **dashboard interativo desenvolvido em Streamlit** para a análise da evolução da COVID-19 no **estado de Pernambuco**, integrando **dados epidemiológicos reais (2020–2025)** com **modelos matemáticos compartimentais** amplamente utilizados em epidemiologia.

O sistema permite visualizar **dados observados**, **simulações epidemiológicas** e a **comparação entre dados reais e modelos**, cobrindo **todo o período disponível nos arquivos**, sem cortes temporais implícitos.

🌐 **Aplicação online:**    https://shre.ink/qWBW

---

## 📚 Fontes de Dados

Os dados utilizados neste projeto foram obtidos a partir de bases oficiais do Ministério da Saúde do Brasil:

* [https://covid.saude.gov.br/](https://covid.saude.gov.br/)
* [https://opendatasus.saude.gov.br/dataset/?tags=covid-19](https://opendatasus.saude.gov.br/dataset/?tags=covid-19)


---

## 🎯 Objetivos do Projeto

* Analisar a **evolução temporal da COVID-19 em Pernambuco**
* Aplicar e comparar **modelos epidemiológicos compartimentais**:

  * **SIR**
  * **SEIR**
  * **SEIRD**
  * **SEIRV**
* Comparar **dados observados** com **simulações epidemiológicas**
* Fornecer uma **ferramenta visual clara e interativa** para apoio a estudos acadêmicos

---

## 🗂️ Estrutura do Repositório

```
├── app.py                         # Aplicação Streamlit
├── covid_pe_seir_ready.parquet    # Dados epidemiológicos observados (PE)
├── cache.parquet                  # Resultados dos modelos epidemiológicos
├── requirements.txt               # Dependências do projeto
└── README.md                      # Documentação do projeto
```

---

## 📁 Descrição dos Arquivos de Dados

### 🔹 `covid_pe_seir_ready.parquet`

Base de dados **pré-processada**, contendo exclusivamente registros do **estado de Pernambuco**.

**Principais variáveis:**

* `date` - Data do registro
* `municipio` - Município de Pernambuco
* `new_cases` - Casos novos diários (incidência)
* `cum_cases` - Casos acumulados
* `I_est` - Estimativa de indivíduos infectantes
* `population` - População estimada
* `I` - Pessoas infecciosas

---

### 🔹 `cache.parquet`

Arquivo de **cache computacional**, contendo os **resultados pré-calculados dos modelos epidemiológicos**.

**Principais variáveis:**

* `date` – Data da simulação
* `municipio` – Município
* `modelo` – Tipo de modelo (`SIR`, `SEIR`, `SEIRD`, `SEIRV`)
* `S`, `E`, `I`, `R`, `D`, `V` – Compartimentos epidemiológicos

Este arquivo é utilizado para:

* 🚀 Acelerar o carregamento do aplicativo
* 🧮 Evitar reprocessamentos computacionais pesados no Streamlit
* 🔁 Garantir consistência e reprodutibilidade entre execuções

---

## 🧮 Modelos Epidemiológicos Implementados

| Modelo    | Descrição                              |
| --------- | -------------------------------------- |
| **SIR**   | Suscetíveis – Infectados – Recuperados |
| **SEIR**  | Inclui período de incubação (Expostos) |
| **SEIRD** | Inclui óbitos                          |
| **SEIRV** | Inclui vacinação                       |

Os modelos seguem **formulações clássicas da literatura epidemiológica**, com parâmetros estimados previamente e armazenados no arquivo de cache.

---

## 📊 Funcionalidades do Dashboard

* Seleção de **município** ou **estado inteiro**
* Seleção de **modelo epidemiológico**
* Visualização de:

  * Casos diários
  * Casos acumulados
  * Estimativa de infectantes
  * Evolução dos compartimentos epidemiológicos
  * Proporção da população por compartimento
  * Comparação **Observado × Modelo**
* Exibição automática de **todo o período disponível nos arquivos**
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
👉 [http://localhost:8501](http://localhost:8501)

---

## 📦 Dependências Principais

* Python ≥ 3.9
* Streamlit
* Pandas
* Plotly
* PyArrow / FastParquet

---

## 🧠 Considerações Metodológicas

* O período analisado corresponde **integralmente aos dados disponíveis**
* Não há **cortes temporais implícitos**
* Todos os municípios pertencem exclusivamente ao **estado de Pernambuco**
* A utilização de **cache computacional** garante desempenho e reprodutibilidade
* A comparação entre dados reais e modelos é realizada de forma **conceitualmente consistente**

---

## 📜 Licença

Este projeto é disponibilizado **exclusivamente para fins acadêmicos e educacionais**.

---

## 👨‍🔬 Autor / Orientação

Projeto desenvolvido para fins **acadêmicos** na **Universidade Federal Rural de Pernambuco (UFRPE)**,
na disciplina **Modelagem Computacional em Epidemiologia**,
com foco na análise da **COVID-19 no estado de Pernambuco**.

---
