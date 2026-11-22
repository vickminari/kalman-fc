# 014 — E2: três σa (teimoso / intermediário / nervoso) e ruído de processo adaptativo

**Data:** 22/11/2026

**Código:**
- `experiments/e2_sigma_a.py` (novo);
- `src/tracker.py`: `TrackerParams.maneuver_*`, `BallTracker._process_noise`, e padrões novos;
- `src/main.py` (`--no-maneuver`, parâmetros no `track.json`).

**Testes:** `tests/test_tracker.py` (4 novos: manobra liga e desliga, resposta ao chute e o compromisso teimoso × nervoso). 50 no total.

## Contexto

- **O E2 é obrigatório:** três valores de Σd (teimoso / intermediário / nervoso), com erro, jitter e atraso.
- **O [010](010-diagnostico-do-kalman-no-video.md) mostrou o problema do σa = 4000:** o filtro fica no lado "peixinho-dourado" (K = 0,90), quase copia o detector, e a predição no escuro segue uma velocidade ruidosa.
- **O [005](005-gating-e-trilha.md) pedia para refazer a varredura de σa com o score combinado** (a última foi feita antes dele, com trocas de alvo mascarando o efeito).
- **A extensão sugerida no plano é o ruído de processo adaptativo** (detecção de manobra): suave na rolagem, ágil nos chutes.
- **σm fica em 1,5 px** (revisão do [004](004-filtro-de-kalman.md)): é o ruído efetivo. Por isso a varredura mexe só em σa.

## Método

**`experiments/e2_sigma_a.py`** roda a trilha sobre os candidatos em cache (como o `replay_tracker`, [013](013-limites-fisicos-no-f2.md)) para cada configuração e mede:

- **K da posição em regime;**
- **"na bola"** e a régua do [011](011-regua-de-diagnostico-da-trilha.md);
- **erro contra a marcação** da medição crua e da estimativa, nos mesmos quadros;
- **tremor** (2ª diferença) da medição e da estimativa;
- **atraso:** o deslocamento s que minimiza a mediana de |estimativa(f + s) − marcação(f)|, com a bola andando (> 2 px/quadro) e a trilha nela;
- **resposta a um chute sintético** (sem ruído): bola a 100 px/s chutada para +200 e +500 px/s; quadros até a velocidade estimada chegar a 90% do salto, e quadros no escuro. Mais o tremor sintético com ruído de 0,3 px (o σm medido).

**Gráficos** em `outputs/experiments/<vídeo>/`:

- `e2_F1_trecho_<quadro>.png` e `e2_F2_trecho_<quadro>.png`: trajetória crua × filtrada no trecho com a maior mudança de velocidade da bola segundo as marcações;
- `e2_F1_chute_sintetico.png`: resposta ao chute.

**Ruído de processo adaptativo (manobra)**, em `BallTracker`:

- **Quando dispara:** se a medição aceita tem Mahalanobis² d² > `maneuver_d2`, a bola não fez o que o modelo previa (chute, quique, desvio).
- **O que muda:** a próxima predição usa Σd × `maneuver_factor`. O `KalmanFilter` é o mesmo; só a matriz `Sd` daquela predição muda (`kf.Sd = ...` antes de `predict()`).
- **Efeito:** P⁻ cresce, então naquele passo a região de busca é maior e o ganho K também, e a velocidade se ajusta depressa. No passo seguinte o Σd volta ao normal.
- **Variante testada:** `maneuver_in_dark` aumenta Σd também nas predições sem medição. Só a região de busca cresce mais depressa; a predição x⁻ = D x não muda.

Comandos:

```bash
python -m experiments.e2_sigma_a videos/campo_lateral.MOV --sigma-a 300 1000 4000 --adaptive 2000:3:8
python -m experiments.e2_sigma_a videos/campo_lateral.MOV --metric --sigma-a 10 40 150 --adaptive 40:3:16
```

(idem para `campo_centro`; tabelas em `outputs/experiments/<vídeo>/e2_F1.csv` e `e2_F2.csv`, detalhes em `.json`)

## Evidência

### E2 em F1 (px), σm = 1,5 px

| | Teimoso σa = 300 | Intermediário 1000 | Nervoso 4000 (padrão antigo) | **Adaptativo 2000, d² > 3, Σd × 8 (padrão novo)** |
|---|---|---|---|---|
| K da posição | 0,48 | 0,70 | 0,90 | 0,81 (≈ 0,95 na manobra) |
| **Lateral:** na bola, teste / todo | 54,9 / 64,7% | 64,1 / 72,5% | 63,7 / 73,5% | **65,0 / 74,3%** |
| **Centro:** na bola, teste / todo | 56,7 / 54,9% | 59,8 / 60,3% | 66,3 / 69,6% | **67,7 / 69,4%** |
| Tremor medição → estimativa, lateral | 0,61 → **0,28** px | 0,66 → 0,44 | 0,70 → 0,60 | 0,70 → 0,56 |
| Tremor medição → estimativa, centro | 0,93 → **0,42** px | 0,99 → 0,64 | 1,02 → 0,88 | 1,01 → 0,77 |
| Saltos > 20 px (lateral / centro) | 77 / 37 | 84 / 35 | 117 / 79 | 110 / **28** |
| Erro da predição no escuro, teste (lat / cen) | 61 / 32 px | 30 / 57 px | 38 / 47 px | **23 / 26 px** |
| Trilhas (lat / cen) | 76 / 65 | 65 / 63 | 59 / 51 | 58 / 53 |
| Escuro recuperado de 10–30 quadros (lat / cen) | 16 / 22 | 12 / 12 | 12 / 10 | 13 / 12 |
| Atraso (quadros; lat / cen) | 0,19 / 0,09 | 0,19 / 0,13 | 0,14 / 0,09 | 0,15 / 0,09 |
| Chute sintético +200 px/s: quadros até 90% | **perde a bola** | 3 | 1 | 2 |
| Chute sintético +500 px/s | perde | perde | 3 (2 no escuro) | perde |
| Tremor sintético (ruído 0,3 px) | 0,25 px | 0,43 | 0,62 | 0,54 |

### E2 em F2 (metros), limite de 20 m/s

| | Teimoso σa = 10 m/s² | Intermediário 40 | Nervoso 150 (padrão antigo) | **Adaptativo 40, d² > 3, Σd × 16 (padrão novo)** |
|---|---|---|---|---|
| K da posição (lateral) | 0,36–0,67 | 0,59–0,88 | 0,81–0,98 | 0,59–0,88 |
| **Lateral:** na bola, teste / todo | 65,0 / 70,8% | 68,4 / 74,1% | 66,2 / 71,6% | **68,8 / 75,1%** |
| **Centro:** na bola, teste / todo | 60,9 / 61,1% | 70,0 / 73,3% | 68,0 / 72,3% | **71,1 / 74,0%** |
| Tremor da estimativa (lat / cen) | 0,32 / 0,50 px | 0,50 / 0,75 | 0,64 / 0,92 | 0,51 / 0,76 |
| Saltos > 20 px (lat / cen) | 85 / 92 | 126 / 76 | 187 / 110 | 144 / 85 |
| Erro da predição no escuro, teste (lat / cen) | 17 / 37 px | 25 / 23 px | 29 / 44 px | **19 / 22 px** |
| Escuro recuperado de 10–30 quadros (lat / cen) | 22 / 23 | 17 / 14 | 12 / 10 | 15 / 12 |

### Outras configurações testadas (F1, na bola no teste, lateral / centro)

| Configuração | Lateral | Centro | Observação |
|---|---|---|---|
| σa = 2000 fixo | 65,4% | 65,4% | já tem a maior parte do ganho; erro no escuro no centro 44 px |
| σa = 3000 fixo | 61,6% | 63,5% | **pior que 2000 e que 4000:** mostra o tamanho do ruído (trocas de alvo) |
| Adaptativo 1000, d² > 3, × 16 | 62,9% | 64,6% | tremor de intermediário (0,46 / 0,65), mas perde chutes maiores |
| Adaptativo 1000, × 16, também no escuro | 62,4% | 65,4% | mais saltos (133 / 82): a região cresce e aceita retomadas longe |
| Adaptativo 1500, d² > 3, × 10 | 65,0% | 66,0% | intermediário entre 1000 e 2000 |
| Adaptativo 2000, d² > 5, × 8 | 65,0% | 65,7% | limiar maior: dispara menos; no centro, erro no escuro 42 px |
| Adaptativo 2000, d² > 3, × 4 | 65,0% | 67,1% | fator menor, quase igual |
| Adaptativo 2000, × 8, 2 quadros | 65,0% | 67,7% | igual ao de 1 quadro |

F2: σa = 75 fixo deu 65,8 / 67,1% e muito mais saltos (201 / 183); o adaptativo 75, × 8 deu 67,5 / 67,7%. **O ótimo em metros está perto de 40 m/s², não em 150** (a conversão de 4000 px/s² por 0,04 m/px herdou o lado nervoso).

## Leitura (elefante × peixinho-dourado)

- **O compromisso aparece inteiro.**
  - **Teimoso** (K ≈ 0,5): corta o tremor da estimativa à metade (0,61 → 0,28 px), mas um chute de +200 px/s (6,7 px/quadro) já tira a bola da região de busca. Nos vídeos, perde 10 pontos de "na bola" e abre 17 trilhas a mais.
  - **Nervoso** (K = 0,9): segue chutes de até +500 px/s, mas quase não suaviza, e a velocidade ruidosa leva a predição no escuro para longe.
- **O "atraso" do E2 aparece como perda, não como deslocamento.** O modelo de velocidade constante não atrasa com a bola em velocidade constante: o atraso medido é ~0,1–0,2 quadro em todas as configurações. No chute, ou a medição cabe na região de busca e o filtro alcança a nova velocidade em 1–3 quadros, ou não cabe e a trilha fica no escuro. **Cada σa define o maior chute que o filtro segue** (com σm = 1,5 px e gate de 99%):

  | σa (px/s²) | 300 | 1000 | 2000 | 4000 |
  |---|---|---|---|---|
  | Maior salto de velocidade seguido | < 200 px/s | 200–300 | 300–500 | ≥ 500 |

- **O erro contra a marcação não separa as configurações** (~2,4 px na lateral, ~3 px no centro, igual para medição e estimativa): a marcação manual tem a mesma ordem de erro. A suavização se mede pelo tremor.
- **A manobra ajuda pouco no "na bola" e bastante no escuro.** Com σa de base menor, a velocidade estimada é menos ruidosa, e a predição no escuro anda menos errado (erro mediano no escuro: 38 → 23 px no lateral, 47 → 26 px no centro). Com d² > 3, a manobra dispara em ~22% das medições normais (χ² com 2 graus de liberdade): na prática, o filtro alterna entre σa = 2000 e ≈ 5700.
- **A manobra não aumenta o maior chute seguido.** Ela precisa de uma medição *aceita* para disparar. Se o chute tira a bola da região de busca, quem resolve é o crescimento de P no escuro, limitado pelo teto de 60 px do [005](005-gating-e-trilha.md). Aumentar Σd também no escuro foi testado: a região cresce mais depressa, mas aceita retomadas longe e aumenta os saltos.
- **Ruído do experimento:** σa = 3000 saiu pior que 2000 e que 4000 no lateral (61,6%). Diferenças de 2–3 pontos são trocas de alvo e não tendência. **A decisão usa a consistência** (os dois vídeos, F1 e F2, e as várias métricas), não um ponto isolado.

## Decisão

**Padrões novos:**

- **F1:** σa = **2000 px/s²** com manobra (d² > 3, Σd × 8, 1 quadro). Era 4000 fixo.
- **F2:** σa = **40 m/s²** com manobra (d² > 3, Σd × 16, 1 quadro). Era 150 fixo; o limite de 20 m/s do [013](013-limites-fisicos-no-f2.md) continua.

**Por quê:** nos dois vídeos, em F1 e F2, o padrão novo fica igual ou melhor que o antigo em "na bola" (teste e vídeo todo; a única exceção é o vídeo todo do centro F1, 69,4 contra 69,6%), tremor, saltos e erro da predição no escuro.

**`--no-maneuver`** no `src.main` desliga a manobra. O E2 do relatório usa os três σa fixos (manobra desligada) e o adaptativo como quarta coluna.

## Alternativas descartadas

- **σa = 4000 (padrão antigo):** mais tremor, mais saltos e a pior predição no escuro, sem ganho de acerto.
- **σa baixo (300–1000) fixo:** suaviza, mas perde a bola nos chutes.
- **σa = 2000 fixo:** quase tão bom quanto o adaptativo. Fica como a escolha "sem extensão" se a dupla preferir não defender o adaptativo.
- **Manobra também no escuro:** mais saltos.
- **σm medido (0,3 px) com σa menor:** mesmo K, região de busca menor e pior (revisão do [004](004-filtro-de-kalman.md)).

## Consequências

- **As métricas de referência mudam** (atualizadas no [006](006-estado-atual-e-proximos-passos.md), com o pipeline completo).
- **O relatório tem o E2 pronto:**
  - a tabela de três σa;
  - o gráfico do trecho real (`e2_F1_trecho_543.png`, lateral: no chute do quadro ~565 o teimoso e o intermediário seguem reto);
  - a resposta ao chute sintético;
  - a discussão "cada σa define o maior chute seguido".
- **Limitação:** o maior chute seguido pelo padrão novo é menor que o do antigo (salto de 300–500 px/s contra ≥ 500 px/s). Nos vídeos de teste isso não custou acerto, mas um vídeo com chutes mais fortes perto da câmera pode pedir σa maior.
