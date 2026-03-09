# LAB P1-02: Encoder do Transformer
**Disciplina:** Tópicos em Inteligência Artificial  
**Instituição:** ICEV  
**Autor:** Adler Castro Alves  

---

## Objetivo
Implementação from scratch do Encoder do Transformer conforme o paper "Attention Is All You Need" (Vaswani et al., 2017), utilizando apenas NumPy e Pandas. O laboratório cobre a pipeline completa de embeddings, self-attention e feed forward network empilhados em 6 camadas com residual connections e layer normalization.

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
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

## Normalização por √d_k
O produto escalar QK^T cresce proporcionalmente à dimensão d_k. Sem o fator de escala, os valores ficam grandes demais e o softmax satura — os gradientes ficam próximos de zero e o aprendizado trava. Dividir por √d_k mantém a variância dos scores estável independentemente da dimensão escolhida.

```python
scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(dk)
```

---

## Exemplo de Input e Output

```python
frase = "o banco bloqueou o cartao"
tokens = [vocab[p] for p in frase.split()]

emb_table = np.random.randn(len(vocab), d_model)
X = emb_table[tokens][np.newaxis, :]

for i in range(n_camadas):
    X = layer_norm(X + self_attention(X, d_model))
    X = layer_norm(X + ffn(X))
```

**Input:**
```
X: (1, 5, 64)  — batch, tokens, d_model
```

**Output:**
```
Z: (1, 5, 64)  — mesma forma, vetores enriquecidos com contexto
```

---

## Arquitetura

O Encoder empilha 6 blocos idênticos. Cada bloco aplica:

1. **Self-Attention** — cada token atende a todos os outros da frase simultaneamente
2. **Add & Norm** — conexão residual + layer normalization
3. **Feed Forward** — duas camadas lineares com ReLU, processa cada token individualmente
4. **Add & Norm** — nova conexão residual + layer normalization

Ao final das 6 camadas, cada vetor de token carrega contexto de toda a sequência.

**Anexo Google Colab:**

[(https://colab.research.google.com/drive/1yqm5C1OA3fZewDO0c0k4pOxpgNwU8Q2G?usp=sharing)]


**Referência:**  
* GOODFELLOW, Ian; BENGIO, Yoshua; COURVILLE, Aaron. Deep Learning. [S. l.]: MIT Press, 2016..
 * JURAFSKY, Daniel; MARTIN, James H. Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models. 3. ed. draft. [S. l.]: Stanford University/University of Colorado at Boulder, 2026..
 * RASCHKA, Sebastian. Build a Large Language Model (From Scratch). 1. ed. [S. l.]: Manning (MEAP), 2021..
 * UNIVERSIDADE FEDERAL DO PIAUÍ. Estágio Curricular Supervisionado - Fábrica de Software I: normas para o estágio supervisionado. Teresina: UFPI, 2026..
 * VASWANI, Ashish et al. Atenção é tudo o que você precisa. Tradução de Machine Translated by Google. [S. l.]: Google Brain/Google Research, 2017..
