# LAB P1-02: Construindo o Transformer Encoder "From Scratch"
**Disciplina:** Tópicos em Inteligência Artificial – 2026.1
**Professor:** Prof. Dimmy Magalhães
**Instituição:** iCEV - Instituto de Ensino Superior
**Autor:** Adler Castro Alves

---

## Objetivo
Implementação from scratch do Forward Pass de um bloco Encoder completo, baseado no paper "Attention Is All You Need" (Vaswani et al., 2017), utilizando apenas NumPy e Pandas. A entrada é uma frase simples e a saída é a representação densa Z após passar por N=6 camadas do Encoder.

```
Output = LayerNorm(x + Sublayer(x))
```

---

## Como Executar
```bash
pip install -r requirements.txt
python encoder.py
```

---

## Estrutura
```
├── encoder.py          # Implementação principal
├── requirements.txt    # Dependências
└── README.md
```

---

## O que foi implementado

**Passo 1 — Preparação dos dados**

Criei um DataFrame no pandas mapeando palavras para IDs, converti a frase de entrada em lista de IDs e inicializei a tabela de embeddings com `np.random.randn(vocab_size, d_model)`. O tensor X final ficou com shape `(1, 5, 64)` — batch, tokens, d_model.

**Passo 2 — Motor matemático**

- `self_attention`: projeta X em Q, K, V via matrizes de pesos aleatórias, calcula o produto escalar `QK^T`, divide por `√d_k` e aplica softmax escrita na mão com `np.exp`
- `layer_norm`: calcula média e variância no último eixo e normaliza com epsilon `1e-6` pra não dividir por zero
- `ffn`: expansão linear com W1 + ReLU (`np.maximum`) + contração com W2, voltando pra dimensão d_model

**Passo 3 — Empilhando as camadas**

Loop de 6 camadas seguindo o fluxo exato do paper:
```
X_att   = self_attention(X)
X_norm1 = layer_norm(X + X_att)
X_ffn   = ffn(X_norm1)
X_out   = layer_norm(X_norm1 + X_ffn)
X       = X_out
```

**Validação de sanidade:** o tensor entra na camada 1 com `(1, 5, 64)` e sai da camada 6 com o mesmo shape — só os valores mudam.

---


**Anexo Google Colab:**

[(https://colab.research.google.com/drive/1yqm5C1OA3fZewDO0c0k4pOxpgNwU8Q2G?usp=sharing)]


**Referência:**  
* GOODFELLOW, Ian; BENGIO, Yoshua; COURVILLE, Aaron. Deep Learning. [S. l.]: MIT Press, 2016..
 * JURAFSKY, Daniel; MARTIN, James H. Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models. 3. ed. draft. [S. l.]: Stanford University/University of Colorado at Boulder, 2026..
 * RASCHKA, Sebastian. Build a Large Language Model (From Scratch). 1. ed. [S. l.]: Manning (MEAP), 2021..
 * UNIVERSIDADE FEDERAL DO PIAUÍ. Estágio Curricular Supervisionado - Fábrica de Software I: normas para o estágio supervisionado. Teresina: UFPI, 2026..
 * VASWANI, Ashish et al. Atenção é tudo o que você precisa. Tradução de Machine Translated by Google. [S. l.]: Google Brain/Google Research, 2017..
