# 013 — Limites físicos no filtro em metros (F2): velocidade máxima e `max_dark`

**Data:** 22/11/2026

**Código:** `experiments/replay_tracker.py` (novo), `src/tracker.py` (`TrackerParams.metric`, sem mudança de lógica)

**Testes:** `tests/test_replay.py` (2 testes)

## Contexto

O [011](011-regua-de-diagnostico-da-trilha.md) mostrou que o F2 tem quase o dobro de saltos no escuro do F1 (lateral: 157 contra 65; passo p90 no escuro de 27 px contra 16). No fundo do campo, o Σm em metros é grande na profundidade, o K vai a 0,98 nesse eixo, e a velocidade em m/s fica muito ruidosa. A predição no escuro segue essa velocidade.

Em px, o limite de velocidade e o amortecimento foram testados e descartados no [005](005-gating-e-trilha.md) (o amortecimento quebrava a travessia da oclusão). Em metros, um limite tem sentido físico: a bola de um chute amador forte não passa de ~30 m/s (108 km/h).

O `TrackerParams` já tinha `max_speed` (corta |v| depois de cada correção; no escuro a velocidade não muda, então a predição também respeita o limite) e `max_dark`. Faltava medir.

## Ferramenta: `experiments/replay_tracker.py`

Rodar o pipeline inteiro para cada combinação levaria ~2 min por vídeo. O experimento guarda, uma vez por vídeo, a estabilização, os candidatos e o score combinado (`outputs/experiments/<vídeo>/replay_cache.pkl`), e repete só a trilha, com a mesma lógica do `src/main.py`. Cada combinação passa pela régua do [011](011-regua-de-diagnostico-da-trilha.md).

**Conferência:** com os parâmetros padrão, o replay dá exatamente os números do `src.main` nos dois vídeos, em F1 e F2 (lateral: 63,7% / 64,6% na bola no teste, 117 / 221 saltos).

## Decisão

- **`TrackerParams.metric` passa a ter `max_speed = 20 m/s`** (72 km/h). O F1 (px) continua sem limite.
- **`max_dark` continua 20 quadros.**
- **`dark_damping` continua desligado** (1,0), como no [005](005-gating-e-trilha.md).

**Por que 20 m/s, e não 30:** a velocidade real da bola, medida pelas **marcações** (posições marcadas a cada ~5 quadros levadas ao campo; média em 1/6 s, só com os dois pontos dentro do campo):

| | Pares | Mediana | p90 | p95 | p99 | Máx. | > 15 m/s | > 20 m/s |
|---|---|---|---|---|---|---|---|---|
| Lateral | 459 | 1,1 m/s | 3,6 | 5,4 | 17,1 | 32,3 | 8 | 2 |
| Centro | 479 | 1,5 m/s | 4,0 | 5,6 | 17,8 | 30,4 | 7 | 4 |

20 m/s fica acima de 99% das velocidades reais. Os poucos intervalos acima disso são provavelmente bola no ar, fora do plano da homografia. Com 30 ou 40 m/s o limite quase nunca atua (resultados iguais ao sem limite). 15 m/s cortaria ~1,5% dos intervalos reais (chutes de verdade) e mostraria velocidades falsas no minimapa.

**Por que não reduzir `max_dark`:** o E3 pede atravessar oclusões de 10–30 quadros. Com `max_dark` = 10, os episódios no escuro de 10–30 quadros que terminam com a bola recuperada caem de 12 para 1 (lateral) e de 10 para 1 (centro). O erro mediano da predição no escuro cai muito (44 → 18 px), mas só porque os episódios longos deixam de existir. Com 15, caem para 8 e 4.

## Alternativas (testadas)

Grade em F2, σa = 150 m/s² (`python -m experiments.replay_tracker videos/<v>.MOV --metric --max-speed 1e9 15 20 30 40 --max-dark 10 15 20`). Tabelas em `outputs/experiments/<vídeo>/t2_F2_grid.csv` e `t2_F2_fine.csv`.

**Lateral F2:**

| max_speed (m/s) / max_dark | sem / 20 (antes) | 40 / 20 | 30 / 20 | **20 / 20** | 15 / 20 | 10 / 20 | 20 / 15 | 20 / 10 |
|---|---|---|---|---|---|---|---|---|
| Na bola (teste) | 64,6% | 64,6% | 64,6% | **66,2%** | 66,2% | 67,1% | 66,7% | 65,8% |
| Na bola (todo) | 70,2% | 70,2% | 70,2% | **71,6%** | 71,6% | 71,0% | 72,0% | 72,5% |
| Aceitas com a bola invisível | 13 | 13 | 13 | **13** | 14 | 14 | 14 | 14 |
| Erro no escuro, mediana (teste) | 44,1 px | 44,1 | 43,0 | **29,4** | 29,4 | 25,7 | 17,3 | 16,8 |
| Velocidade p95 | 61,6 km/h | 61,6 | 61,6 | **50,4** | 49,3 | 36,0 | 50,4 | 42,0 |
| Saltos > 20 px | 221 | 221 | 221 | **187** | 176 | 173 | 179 | 148 |
| … no escuro | 157 | 157 | 157 | **121** | 112 | 106 | 114 | 80 |
| Trilhas | 60 | 60 | 60 | **57** | 57 | 57 | 66 | 77 |
| Escuro recuperado de 10–30 quadros | 12 | 12 | 12 | **12** | 12 | 15 | 8 | 1 |

**Centro F2:**

| max_speed / max_dark | sem / 20 (antes) | 30 / 20 | **20 / 20** | 15 / 20 | 20 / 15 | 20 / 10 |
|---|---|---|---|---|---|---|
| Na bola (teste) | 68,0% | 68,0% | **68,0%** | 67,4% | 68,6% | 69,4% |
| Na bola (todo) | 72,1% | 72,1% | **72,3%** | 71,8% | 73,8% | 74,3% |
| Erro no escuro, mediana (teste) | 44,3 px | 44,3 | **44,3** | 36,5 | 23,3 | 13,6 |
| Velocidade p95 | 49,2 km/h | 49,2 | **49,2** | 37,3 | 43,5 | 37,7 |
| Saltos > 20 px | 110 | 110 | **110** | 87 | 90 | 70 |
| Escuro recuperado de 10–30 quadros | 10 | 10 | **10** | 10 | 4 | 1 |

## Evidência no pipeline completo

`python -m src.main videos/campo_lateral.MOV --scorer data/models/ball_scorer_campo_lateral.json --metric --save-video --tag lateral_F2_t2`, seguido de `python -m experiments.track_diagnostics lateral_F2 lateral_F2_t2`:

| Lateral F2 | Antes (`lateral_F2`) | Depois (`lateral_F2_t2`) |
|---|---|---|
| Na bola (teste) | 64,6% | **66,2%** |
| Na bola (vídeo todo) | 70,2% | **71,6%** |
| Aceitas com a bola invisível | 13 | 13 |
| Erro no escuro, mediana (teste) | 44,1 px | **29,4 px** |
| Velocidade p95 | 61,6 km/h | **50,4 km/h** |
| Saltos > 20 px (no escuro / retomada / nova / outro) | 221 (157 / 42 / 10 / 12) | **187** (121 / 46 / 8 / 12) |
| Passo no escuro, mediana / p90 | 3,5 / 27,0 px | 3,4 / 21,5 px |
| Aceitas na bola / com a bola visível | 301/320 | 308/327 |
| Trilhas | 60 | 57 |
| Pipeline (com vídeo e minimapa) | 42 fps | 68 fps (a velocidade varia com a carga da máquina; o limite não custa nada) |

Os números são idênticos aos do replay.

## Consequências

- **No lateral o limite ajuda** em todas as métricas da régua. **No centro, 20 m/s não muda nada**: a velocidade corrigida do centro quase nunca passa disso. As predições mais rápidas do lateral vêm de bolas no fundo do campo, onde 1 px vale muitos metros.
- **O limite não resolve a maior parte dos saltos no escuro** (121 continuam no lateral): a velocidade ruidosa está quase sempre abaixo de 20 m/s. O que sobra é o compromisso de σa (E2, Tarefa 3) e a visualização ([012](012-visualizacao-honesta-da-predicao.md)), que já não desenha a predição depois de 10 quadros.
- **O erro no escuro e o `max_dark` andam juntos:** reduzir `max_dark` melhora a métrica só por cortar os episódios longos. Para o E3, o número que importa é "episódios de 10–30 quadros recuperados".
- **O F1 (px) continua sem limite.** Em px um limite não tem sentido físico (a escala muda com a profundidade), como no [005](005-gating-e-trilha.md).
