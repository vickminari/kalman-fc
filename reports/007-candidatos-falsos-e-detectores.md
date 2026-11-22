# 007 — Candidatos falsos: comparação de detectores (blob, LoG, LSS) e score combinado

**Data:** 21/11/2026

**Código:**

- `src/motion_cue.py`, `src/blob_detector.py`, `src/lss.py`, `src/scoring.py`;
- `experiments/candidate_features.py`;
- `--scorer` em `src/main.py`.

**Testes:** `tests/test_candidate_cues.py`

## Contexto

Depois de [005](005-gating-e-trilha.md), a trilha ficava na bola em só 30% dos quadros visíveis, abaixo do melhor candidato do detector (38%). O detector HSV entregava ~6 falsos por quadro (pés, folhas e papéis, a base da rede).

Duas perguntas:

1. **Outro detector seria melhor?** Foram sugeridos o detector de blobs e o descritor de auto-similaridade local de Shechtman & Irani (2007), "Matching Local Self-Similarities across Images and Videos", que o professor mencionou.
2. **Que informação separa a bola dos falsos?**

## Método

Experimento `python -m experiments.candidate_features videos/campo_lateral.MOV`, sobre os **810 quadros marcados** do vídeo lateral (510 com a bola visível).

- **Rótulo:** um candidato é "bola" se estiver a ≤ 10 px da marcação.
- **Divisão temporal:** treino = quadros < 2012 (primeira metade das marcações), teste = quadros ≥ 2012. Tudo o que é ajustado (molde LSS, prior de tamanho, pesos) usa só o treino; as métricas são do teste.
- **AUC** = probabilidade de um candidato-bola ter o atributo "mais de bola" que um falso (0,5 = inútil, 1 = perfeito).

**Geradores de candidatos comparados:**

| Gerador | Ideia |
|---|---|
| **HSV** (atual, [002](002-detector.md)) | Máscara branca ∩ campo, morfologia, forma, anel de grama |
| **LoG em escala** | Máximos de −σ²∇²G_σ sobre um mapa contínuo de brancura, σ ∈ {1; 1,4; 2; 2,8; 4; 5,6}; mesmo teste de anel |
| **SimpleBlobDetector** (OpenCV) | Multi-limiar sobre o mapa de brancura + filtros de área, circularidade, convexidade e inércia; mesmo teste de anel |

**Atributos novos por candidato:**

- **Movimento com a câmera compensada:** compara o patch do candidato com o patch do **mesmo ponto do mundo** 3 quadros antes (localizado pelas homografias da estabilização), com tolerância de ±2 px. Objeto parado dá movimento baixo; bola rolando, alto.
- **LSS** (Shechtman & Irani): SSD do patch central 5×5 contra a vizinhança, em L*a*b*, com S = exp(−SSD / max(var_ruído, var_auto)), grade log-polar 12 ângulos × 3 raios, máximo por posição e normalização para [0, 1].
  - **Adaptações:** calculado **só nos candidatos**, com a região **escalada pelo diâmetro** do candidato (raio = 1,6 d, reamostrada para 41×41).
  - **Molde:** média dos descritores das bolas do treino.
  - **Atributo:** −L1 até o molde.
- **Prior de tamanho (perspectiva):** log d = a + b·y, ajustado nas bolas do treino (a = 0,886, b = 0,00971); atributo = −|log d − previsto|.

## Resultados

**1. Geradores** (vídeo inteiro / teste):

| Gerador | Bola entre os candidatos | Melhor = bola | Falsos/quadro | ms/quadro |
|---|---|---|---|---|
| **HSV** | **69,2%** / 66,7% | 38,2% / 29,5% | 5,99 / 6,61 | **3,5** |
| LoG em escala | 64,9% / 63,3% | 0,6% / 0,4% | 36,7 / 36,8 | 26,4 |
| SimpleBlobDetector | 25,1% / 26,6% | 18,8% / 16,5% | 0,31 / 0,45 | 5,0 |

- **LoG:** acha quase tantas bolas, mas com 6× mais falsos e 7× mais lento. A resposta do LoG não serve como score (texturas claras da grama também dão resposta).
- **SimpleBlobDetector:** com os filtros usados, perde 3/4 das bolas.
- **O gerador HSV fica.**

**2. Poder de separação de cada atributo** (AUC no teste, candidatos HSV):

| Atributo | AUC |
|---|---|
| Prior de tamanho | **0,940** |
| LSS (similaridade ao molde) | **0,922** |
| Movimento compensado | **0,913** |
| Score do detector | 0,803 |
| Circularidade | 0,798 |
| Grama no anel | 0,606 |
| Brancura | 0,596 |

**3. Score combinado:** regressão logística ajustada no treino, padronizada, com L2. Retirando uma pista de cada vez, no teste:

| Atributos | AUC | Melhor = bola | Recall no limiar* | Falsos/quadro no limiar* |
|---|---|---|---|---|
| Só os do detector | 0,815 | 32,9% | 60,8% | 4,46 |
| + movimento | 0,916 | 48,5% | 64,6% | 2,71 |
| + LSS | 0,930 | 47,3% | 62,9% | 1,58 |
| + tamanho | 0,970 | 59,9% | 60,8% | 0,75 |
| + movimento + tamanho | 0,983 | 64,1% | 60,3% | 0,20 |
| + LSS + tamanho | 0,990 | 65,0% | 62,4% | 0,17 |
| + movimento + LSS | 0,952 | 52,7% | 59,9% | 0,95 |
| **Todos** | **0,991** | **65,4%** | 60,8% | **0,14** |

\*Limiar do score combinado que mantinha 95% das bolas do **treino**. O recall máximo possível no teste é 66,7%: é o quanto o HSV acha.

**Pesos do modelo final** (atributos padronizados): tamanho 3,57 · LSS 1,78 · score 1,74 · movimento 0,64 · grama no anel −1,44 · circularidade −0,34 · brancura −0,27 · viés −7,49. Os pesos negativos dos atributos do detector aparecem porque eles são correlacionados com o score, que já os contém.

**4. Efeito na trilha** (`python -m src.main videos/campo_lateral.MOV --scorer data/models/ball_scorer_campo_lateral.json`):

| Configuração | Parte | Estimativa na bola | Medição na bola | Aceitas com a bola invisível | Erro da predição no escuro (mediana) |
|---|---|---|---|---|---|
| Sem score combinado (base) | treino | 43,2% | 36,6% | 110 | 242 px |
| Sem score combinado (base) | **teste** | **14,8%** | 13,9% | 99 | 256 px |
| Score combinado, só reordenando | treino | 28,6% | 23,1% | 103 | 151 px |
| Score combinado, só reordenando | teste | 34,6% | 32,1% | 120 | 158 px |
| **Score combinado + filtro no limiar** | treino | 82,1% | 66,7% | 6 | 5 px |
| **Score combinado + filtro no limiar** | **teste** | **63,7%** | 52,3% | **5** | **38 px** |

- **Velocidade** (rodando sozinho, 60 s do vídeo): 77 fps com o score combinado, contra 116 fps sem. Continua bem acima dos 15 fps exigidos.
- **Estados** com o filtro: RASTREANDO 36,9%, BUSCANDO 32,6% (a bola fora do quadro agora fica corretamente "sem trilha"), NO_ESCURO 26,5%.

## Decisão

1. **Manter o detector HSV como gerador de candidatos.** LoG e SimpleBlobDetector foram testados e descartados: o LoG por excesso de falsos e custo, o SimpleBlob por baixo recall.
2. **Acrescentar uma etapa de score combinado** (`src/scoring.py`) entre o detector e a trilha, com:
   - movimento compensado;
   - LSS (Shechtman & Irani);
   - prior de tamanho por perspectiva;
   - os atributos do detector.
   Ela é ativada por `--scorer data/models/ball_scorer_<vídeo>.json`. Candidatos abaixo do limiar são descartados, e a probabilidade vira o score da trilha.
3. **O modelo é gerado** por `experiments/candidate_features.py` (treino = primeira metade das marcações) e salvo em `data/models/`.

## Respostas às perguntas

- **Detector de blobs:** o LoG em escala é o "blob" clássico (Lindeberg; base do DoG/SIFT). Como **gerador de candidatos**, não superou o HSV nesta cena: a bola branca sobre a grama já é bem resolvida pela cor, e o problema eram os objetos brancos parecidos.
- **Auto-similaridade local:** como **verificador** dos candidatos, foi uma das pistas mais fortes (AUC 0,92), e somou ao prior de tamanho (0,970 → 0,990). O uso original do artigo (casar um molde por um conjunto denso de descritores na imagem inteira) seria caro e pensado para objetos maiores e com estrutura interna. Aplicado só nos candidatos e normalizado pela escala, custa ~0,5 ms por candidato.

## Limitações e riscos

- **Mesmo vídeo para ajuste e avaliação.** O molde LSS, o prior de tamanho e os pesos foram ajustados na primeira metade do **mesmo vídeo** e avaliados na segunda. A câmera é a mesma, então a generalização para **outra posição de câmera não foi medida**.
  - **O prior de tamanho depende da geometria:** no vídeo do centro, o modelo do lateral pode estar errado. Rodar o centro com o modelo do lateral gera saída (82 fps, 56 trilhas), mas **sem marcações não dá para avaliar**.
  - **Para o centro:** marcar e reajustar, ou derivar o tamanho esperado da homografia do campo (bola de 0,22 m projetada no chão), o que elimina o ajuste por vídeo.
- **Recall continua limitado pelo HSV** (~67–69%): bola colada no pé, encostada na parede branca, grande e borrada perto da câmera (exemplo: quadro 3850, em que a trilha foi parar na mão de um jogador). A pele clara sob a luz também gera candidatos "brancos".
- **Classificador aprendido:** é um classificador linear de 7 pesos sobre atributos feitos à mão, explicável no relatório e fora do filtro de Kalman (que continua implementado do zero). Não é rede neural nem rastreador pronto, então respeita o enunciado. Mesmo assim, vale citar no relatório como etapa do "sensor".
- **Ruído dos números:** com ~250 quadros com a bola visível no teste, diferenças de poucos pontos não são significativas.

## Revisão (21/11) — vídeo do centro e generalização entre vídeos

Com as marcações do centro (727 quadros, 599 com a bola visível; commit `d6fb94d`), o experimento foi repetido no vídeo do centro (treino = quadros < 1820, teste ≥ 1820). O modelo resultante foi salvo em `data/models/ball_scorer_campo_centro.json`.

**No próprio vídeo do centro** (teste): o padrão se repete.

| | Valor |
|---|---|
| Gerador HSV: recall | 67,7% |
| Gerador HSV: falsos/quadro | 3,42 |
| LoG: recall / falsos por quadro | 49,9% / 11,3 |
| SimpleBlob: recall | 11,9% |
| AUC: tamanho | 0,976 |
| AUC: movimento | 0,898 |
| AUC: LSS | 0,828 (mais fraco que no lateral, 0,922) |
| AUC: score do detector | 0,661 |
| Melhor = bola: score do detector → combinado | 28,3% → **66,3%** |
| Falsos/quadro no limiar | **0,10** |

- **Retirando uma pista de cada vez:** o tamanho continua a pista mais forte (sozinho, 64,6% de "melhor = bola"). O movimento ajuda mais que o LSS nesse vídeo: base + movimento + tamanho dá 66,0%; base + LSS + tamanho, 63,5%.
- **Prior de tamanho ajustado:** log d = 1,698 + 0,00628·y. No lateral era 0,886 + 0,00971·y: a geometria é outra, como esperado.

**Trilha, cruzando modelos e vídeos:**

| Vídeo | Modelo | Na bola (vídeo todo) | **Na bola (teste)** | Aceitas com a bola invisível (todo) | Predição no escuro, mediana (todo) |
|---|---|---|---|---|---|
| Centro | sem score | 35,1% | 29,7% | 87 | 155 px |
| Centro | do lateral (nunca viu o centro) | 50,4% | 53,3% | 25 | 118 px |
| Centro | do centro | 69,6% | **66,3%** | 9 | 44 px |
| Lateral | sem score | 30,0% | 14,8% | 209 | 252 px |
| Lateral | do centro (nunca viu o lateral) | 67,8% | **62,9%** | 20 | 18 px |
| Lateral | do lateral | 73,5% | 63,7% | 11 | 12 px |

**Conclusões:**

- **Nos dois sentidos, o modelo de outro vídeo é muito melhor que não usar score.**
- **O modelo do centro generaliza quase perfeitamente para o lateral** (62,9% contra 63,7% do modelo próprio). O do lateral perde ~13 pontos no centro. Hipótese: no centro o modelo dá mais peso ao movimento (1,59 contra 0,64) e menos ao score do detector, que varia mais entre câmeras.
- **O prior de tamanho é a peça mais dependente da câmera.** Derivá-lo da homografia do campo deve remover a maior parte dessa dependência.

**Decisão:** até existir um modelo ajustado nos vídeos definitivos, usar o **modelo do centro como padrão genérico** para vídeos sem marcações; para cada vídeo marcado, o modelo próprio. Próximo passo natural: um modelo **conjunto**, treinado na primeira metade dos dois vídeos.
