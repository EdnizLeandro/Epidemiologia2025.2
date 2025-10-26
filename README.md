# 🧬 Epidemiologia2025.2

**Epidemiologia2025.2** implementa um modelo híbrido que combina o **SEIR** (Suscetíveis, Expostos, Infectados, Recuperados) **para simular a propagação **temporal e espacial** da covid19 em pernambuco no príodo de 27/03/2020 há 30/08/2025.  
O projeto também utiliza a **Entropia de Shannon** para quantificar a **desordem espacial**, analisando a dinâmica e complexidade dos surtos epidêmicos.

---

## 📖 Descrição do Projeto

Este projeto propõe uma abordagem integrada para estudar a disseminação de doenças com período de latência, unindo:
- A **dinâmica temporal** do modelo **SEIR**, que descreve a transição entre estados epidemiológicos;
- A **dinâmica espacial** modelada por um **Autômato Celular**, que simula a interação local entre indivíduos em uma grade bidimensional;
- A **Entropia da Informação (Shannon)**, que mede o grau de desordem ou incerteza na distribuição espacial dos estados de saúde.

Essa combinação permite compreender tanto a evolução da epidemia ao longo do tempo quanto os padrões espaciais emergentes de contágio.

---

## 🎯 Objetivos

- Simular a **evolução temporal e espacial** de uma epidemia.  
- Analisar o impacto de parâmetros como taxa de infecção, incubação e recuperação.  
- Quantificar a **entropia da informação** como medida de desordem espacial.  
- Gerar visualizações dinâmicas e estatísticas da propagação da doença.

---

## 🧠 Modelos Utilizados

### 🔹 Modelo SEIR
Extensão do modelo SIR, com a classe **Expostos (E)** representando indivíduos infectados, mas ainda não contagiosos.  
Adequado para doenças com **período de incubação**, como **sarampo**, **catapora** e **COVID-19**.

### 🔹 Autômato Celular (AC)
Representa a população em uma **grade bidimensional**, onde cada célula está em um dos estados:
- `S` — Suscetível  
- `E` — Exposto  
- `I` — Infectado  
- `R` — Recuperado  

As transições de estado dependem dos vizinhos, permitindo observar **padrões espaciais complexos** de disseminação.

### 🔹 Entropia de Shannon
Utilizada para medir a **desordem espacial** do sistema.  
Valores altos → alta incerteza e diversidade de estados;  
Valores baixos → maior ordem e estabilidade.

---

## ⚙️ Tecnologias Utilizadas

- **Python 3.11+**
- **NumPy**, **Pandas**
- **Matplotlib**, **Seaborn**
- **SciPy**
- **Jupyter Notebook**



