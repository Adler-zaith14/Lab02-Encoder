# LAB P1-02: Construindo o Transformer Encoder
**Disciplina:** Tópicos em Inteligência Artificial – 2026.1
**Professor:** Prof. Dimmy Magalhães
**Instituição:** iCEV - Instituto de Ensino Superior
**Autor:** Adler Castro Alves

---

## Objetivo

Implementação from scratch do Forward Pass de um Encoder completo baseado no paper *"Attention Is All You Need"* (Vaswani et al., 2017), utilizando apenas NumPy e Pandas. A entrada é a frase `"o banco bloqueou o cartao"` e a saída é o tensor de representações densas **Z** após passar por **N=6 camadas** empilhadas do Encoder.

```
Output = LayerNorm(x + Sublayer(x))
```

---

## Como Executar

> Recomendado rodar no **Google Colab** — basta executar as células em ordem.

```bash
pip install numpy pandas
```

---

## Estrutura do Projeto

```
├── encoder.ipynb       # Notebook Google Colab
├── encoder.py          # Implementação principal
├── requirements.txt    # Dependências
└── README.md
```

---

## Implementação

###  Passo 1 — Preparação dos Dados

Vocabulário de 4 palavras construído com Pandas mapeando `palavra → ID`. A frase de entrada é tokenizada, convertida em índices e embarcada em uma tabela inicializada com `np.random.randn(vocab_size, d_model)`.

| Hiperparâmetro | Valor |
|---|---|
| `d_model` | 64 |
| `d_ff` | 256 (`d_model × 4`) |
| `n_camadas` | 6 |
| Shape de entrada X | `(1, 5, 64)` |

---

###  Passo 2 — Funções Matemáticas

**Scaled Dot-Product Attention**

Implementa a atenção single-head da seção 3.1 do paper:

```
Attention(Q, K, V) = softmax(QKᵀ / √dk) · V
```

Onde `Q`, `K`, `V` são projeções lineares de `X` via matrizes de pesos `Wq`, `Wk`, `Wv`.

---

**Layer Normalization**

Normalização aplicada no eixo de features com epsilon para estabilidade numérica:

```
LayerNorm(x) = (x − μ) / √(σ² + ε)
```

---

**Feed-Forward Network**

Expansão linear seguida de ReLU e contração de volta à dimensão `d_model`:

```
FFN(x) = max(0, xW1 + b1) W2 + b2
```

---

###  Passo 3 — Inicialização dos Pesos

Os pesos `Wq`, `Wk`, `Wv`, `W1`, `W2` são inicializados uma única vez por camada e armazenados em uma lista de dicionários antes do forward pass:

```python
camadas = []
for _ in range(n_camadas):
    camadas.append({
        "Wq": np.random.randn(d_model, d_model),
        "Wk": np.random.randn(d_model, d_model),
        "Wv": np.random.randn(d_model, d_model),
        "W1": np.random.randn(d_model, d_ff),
        "b1": np.zeros(d_ff),
        "W2": np.random.randn(d_ff, d_model),
        "b2": np.zeros(d_model),
    })
```

---

###  Passo 4 — Forward Pass (6 Camadas)

Loop de 6 camadas seguindo o fluxo do paper:

```python
X_att   = scaled_dot_product_attention(X, Wq, Wk, Wv)
X_norm1 = layer_norm(X + X_att)
X_ffn   = ffn(X_norm1, W1, b1, W2, b2)
X_out   = layer_norm(X_norm1 + X_ffn)
X       = X_out
```

O tensor entra na camada 1 com shape `(1, 5, 64)` e sai da camada 6 com o mesmo shape — apenas os valores são transformados a cada camada.

---



**Anexo Google Colab:**

[(https://colab.research.google.com/drive/1yqm5C1OA3fZewDO0c0k4pOxpgNwU8Q2G?usp=sharing)]


**Referência:**  
* GOODFELLOW, Ian; BENGIO, Yoshua; COURVILLE, Aaron. Deep Learning. [S. l.]: MIT Press, 2016..
 * JURAFSKY, Daniel; MARTIN, James H. Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models. 3. ed. draft. [S. l.]: Stanford University/University of Colorado at Boulder, 2026..
 * RASCHKA, Sebastian. Build a Large Language Model (From Scratch). 1. ed. [S. l.]: Manning (MEAP), 2021..
 * UNIVERSIDADE FEDERAL DO PIAUÍ. Estágio Curricular Supervisionado - Fábrica de Software I: normas para o estágio supervisionado. Teresina: UFPI, 2026..
 * VASWANI, Ashish et al. Atenção é tudo o que você precisa. Tradução de Machine Translated by Google. [S. l.]: Google Brain/Google Research, 2017..
