# 006 — Estado atual, métricas de referência e próximos passos

**Atualizado em:** 22/11/2026, depois de [008](008-videos-de-teste-como-base.md), [009](009-geometria-do-campo-e-filtro-em-metros.md), [010](010-diagnostico-do-kalman-no-video.md) e da limpeza do filtro, tarefas 0–5 ([011](011-regua-de-diagnostico-da-trilha.md)–[015](015-falsos-positivos-de-pele-e-cor-da-bola.md), revisão do [004](004-filtro-de-kalman.md)). Commits: `3405e91` (pipeline), `b8a686e` (score combinado), `d6fb94d` (marcações do centro), `3dea9e4` (campo, F2 e minimapa), `cdb6610` (limpeza do rastreamento, [011](011-regua-de-diagnostico-da-trilha.md)–[015](015-falsos-positivos-de-pele-e-cor-da-bola.md)). Os reports entram no repositório a partir do commit seguinte.

## O que existe

| Peça | Arquivo | Situação |
|---|---|---|
| Leitura de vídeo/câmera | `src/video.py` | Pronta |
| Detector HSV (gerador de candidatos) | `src/detector.py` | ~69% de recall, ~6 falsos por quadro |
| Detectores de blob (comparação) | `src/blob_detector.py` | Testados e descartados como gerador ([007](007-candidatos-falsos-e-detectores.md)) |
| Movimento compensado | `src/motion_cue.py` | Pronto (AUC 0,91) |
| Auto-similaridade local (LSS) | `src/lss.py` | Pronto (AUC 0,92) |
| Score combinado | `src/scoring.py`, `data/models/ball_scorer_campo_{lateral,centro}.json` | Pronto, com a cor da bola (`cool`, [015](015-falsos-positivos-de-pele-e-cor-da-bola.md)). Modelo por vídeo (treino = 1ª metade); generaliza entre os dois vídeos |
| Estabilização (câmera na mão) | `src/stabilize.py` | Pronta e validada |
| Filtro de Kalman | `src/kalman.py` | Pronto e validado contra o OpenCV |
| Gating e trilha | `src/gating.py`, `src/tracker.py` | Prontos; σa reajustado com o score combinado (E2) e detecção de manobra ([014](014-e2-sigma-a-e-ruido-de-processo-adaptativo.md)) |
| Visualização | `src/viz.py`, `src/minimap.py` | Cores da aula, elipse, faixa "NO ESCURO", minimapa com km/h; rastro sólido só com medição, predição no escuro tracejada e esmaecida por até 10 quadros ([012](012-visualizacao-honesta-da-predicao.md)) |
| Geometria do campo | `src/field.py`, `tools/calibrate_field.py`, `data/calib/` | Lateral calibrado (2,28 px; h = 2,09 m); centro calibrado com priors ([009](009-geometria-do-campo-e-filtro-em-metros.md)) |
| Filtro em metros (F2) | `src/main.py --metric` | Pronto; limite físico de 20 m/s ([013](013-limites-fisicos-no-f2.md)); F2 ≥ F1 em acerto, com velocidade plausível |
| Pipeline | `src/main.py` (`--scorer`, `--metric`, `--field`) | Arquivo ou câmera |
| Experimentos | `experiments/sweep_tracker.py`, `experiments/candidate_features.py`, `experiments/track_diagnostics.py` (régua, [011](011-regua-de-diagnostico-da-trilha.md)), `experiments/replay_tracker.py` (trilha sobre candidatos em cache, [013](013-limites-fisicos-no-f2.md)), `experiments/sigma_m.py` ([004](004-filtro-de-kalman.md)), `experiments/e2_sigma_a.py` (E2, [014](014-e2-sigma-a-e-ruido-de-processo-adaptativo.md)), `experiments/skin_check.py` ([015](015-falsos-positivos-de-pele-e-cor-da-bola.md)) | Prontos |
| Testes | `tests/` | 59 passando |
| Verdade de campo | `data/annotations/` | **Os dois vídeos completos**: lateral 810 quadros (510 com a bola visível), centro 727 (599) |

Em relação ao plano de trabalho: **T03–T11, T17 (via calibração do Gabriel + f na cena), T18, T19, T20, T21, T22, T24 e T25** estão feitos.

**Faltam:**

- **Experimentos e entrega:** E1 e E3 no formato final (o E2 está pronto, [014](014-e2-sigma-a-e-ruido-de-processo-adaptativo.md)) e os entregáveis.

## Métricas de referência

**Histórico (commit `3dea9e4`, antes da limpeza; não remedido com os padrões atuais).** Trilha, % dos quadros com a bola visível em que a estimativa fica a ≤ 10 px da marcação, na **metade de teste** de cada vídeo (lateral: quadros ≥ 2012; centro: ≥ 1820). Parâmetros da trilha da época (σa = 4000, gate 0,99, max_dark 20). **As métricas atuais estão na seção seguinte.**

| Vídeo | Sem score | Modelo do outro vídeo | Modelo do próprio vídeo |
|---|---|---|---|
| Lateral | 14,8% | 62,9% (do centro) | **63,7%** |
| Centro | 29,7% | 53,3% (do lateral) | **66,3%** |

| Outras métricas | Lateral | Centro |
|---|---|---|
| Candidatos: recall do HSV (teste) | 66,7% | 67,7% |
| Candidatos: falsos/quadro sem → com score | 6,61 → 0,14 | 3,42 → 0,10 |
| Pipeline com score (sozinho) | 77 fps | ~80 fps |

Comando de referência:

```bash
python -m src.main videos/<vídeo>.MOV --scorer data/models/ball_scorer_<vídeo>.json
```

### Métricas de referência da trilha, F1 × F2 (régua do [011](011-regua-de-diagnostico-da-trilha.md)) — ATUAIS

**Padrões atuais** (22/11):

- **F1:** σa = 2000 px/s² com manobra (d² > 3 → Σd × 8);
- **F2:** σa = 40 m/s² com manobra (× 16) e `max_speed` = 20 m/s;
- **nos dois:** σm = 1,5 px, gate 0,99, `max_dark` 20;
- **score combinado com `cool`**, modelo do próprio vídeo.

Evolução na metade de teste ("na bola") e no vídeo todo:

| | Lateral F1 | Lateral F2 | Centro F1 | Centro F2 |
|---|---|---|---|---|
| Antes da limpeza (`3dea9e4`), teste / todo | 63,7 / 73,5% | 64,6 / 70,2% | 66,3 / 69,6% | 68,0 / 72,1% |
| + limite de 20 m/s no F2 ([013](013-limites-fisicos-no-f2.md)) | = | 66,2 / 71,6% | = | 68,0 / 72,3% |
| + σa novo e manobra ([014](014-e2-sigma-a-e-ruido-de-processo-adaptativo.md)) | 65,0 / 74,3% | 68,8 / 75,1% | 67,7 / 69,4% | 71,1 / 74,0% |
| **+ cor da bola no score ([015](015-falsos-positivos-de-pele-e-cor-da-bola.md)) = atual** | **71,7 / 77,8%** | **73,8 / 78,6%** | **68,3 / 73,6%** | **71,1 / 76,8%** |

Régua completa, atual (entre parênteses, antes da limpeza):

| | Lateral F1 | Lateral F2 | Centro F1 | Centro F2 |
|---|---|---|---|---|
| Aceitas com a bola invisível (todo) | 5 (11) | 5 (13) | 0 (9) | 0 (10) |
| Predição no escuro, erro mediano (teste) | 17 px (38) | 15 px (44) | 10 px (47) | 8 px (44) |
| Velocidade p95 | 26 km/h (48) | 23 km/h (62) | 28 km/h (96, implausível) | 27 km/h (49) |
| K da posição em regime (sem manobra) | 0,81 (0,90) | 0,58–0,88 (0,81–0,98) | 0,81 (0,90) | 0,64–0,89 (0,85–0,98) |
| Tremor medição → estimativa | 0,68 → 0,54 px (0,70 → 0,60) | 0,69 → 0,49 (0,70 → 0,63) | 1,00 → 0,77 (1,02 → 0,88) | 1,01 → 0,74 (1,03 → 0,92) |
| Saltos > 20 px | 46 (117) | 60 (221) | 17 (79) | 66 (110) |
| Aceitas na bola / com a bola visível | 329/336 (306/321) | 328/335 (301/320) | 375/378 (367/379) | 389/391 (382/391) |
| Trilhas iniciadas | 40 (59) | 37 (60) | 31 (51) | 23 (48) |
| Escuro recuperado de 10–30 quadros | 11 (12) | 13 (12) | 6 (10) | 8 (10) |

**Conferido no pipeline completo** (`python -m src.main ... [--metric] --save-video --tag <v>_F<n>_t4` e `python -m experiments.track_diagnostics lateral_F1_t4 lateral_F2_t4 centro_F1_t4 centro_F2_t4`): números idênticos aos do replay. **Pipeline com vídeo e minimapa: 76 fps (lateral) e 69–70 fps (centro).** Vídeos em `outputs/<v>_F<n>_t4/tracked.mp4`.

Comandos: `python -m src.main videos/<v>.MOV --scorer data/models/ball_scorer_<v>.json [--metric]` e depois `python -m experiments.track_diagnostics <tag>`; para varrer parâmetros sem reprocessar o vídeo, `python -m experiments.replay_tracker videos/<v>.MOV [--metric] --sigma-a ... --max-speed ... --max-dark ...`.

## Próximos passos (atualizado em 22/11, depois da limpeza do filtro: [011](011-regua-de-diagnostico-da-trilha.md)–[015](015-falsos-positivos-de-pele-e-cor-da-bola.md))

**Base:** vídeos de teste ([008](008-videos-de-teste-como-base.md)). Ordem proposta:

1. ~~**Limpar o comportamento do filtro**~~ ([010](010-diagnostico-do-kalman-no-video.md), bloco A; plano em `atividadeFinal/plano_limpeza_kalman.md`): feito.
   - régua ([011](011-regua-de-diagnostico-da-trilha.md));
   - visualização honesta ([012](012-visualizacao-honesta-da-predicao.md));
   - limite de 20 m/s no F2 ([013](013-limites-fisicos-no-f2.md));
   - σm medido (revisão do [004](004-filtro-de-kalman.md));
   - E2 e manobra ([014](014-e2-sigma-a-e-ruido-de-processo-adaptativo.md));
   - cor da bola no score ([015](015-falsos-positivos-de-pele-e-cor-da-bola.md)).
2. **Experimentos obrigatórios** nos vídeos de teste, com script, tabelas e gráficos:
   - **E1:** trajetória crua × filtrada. O erro contra a marcação não separa crua e filtrada (~2,4 px, a precisão da marcação); usar o tremor e os gráficos do `e2_sigma_a.py`;
   - ~~**E2:**~~ pronto ([014](014-e2-sigma-a-e-ruido-de-processo-adaptativo.md): tabela, gráficos, resposta ao chute);
   - **E3:** oclusões naturais de 10–30 quadros (6–13 recuperadas por vídeo e filtro; listar com `track_diagnostics`), mais oclusão simulada.
3. **Ferramenta interativa de marcos** (`tools/pick_field_points.py`, Tarefa 6 do plano, opcional); se der, medir a distância linha do gol → muro do fundo.
4. **Tamanho esperado da bola pela geometria** no score combinado; depois, modelo de score conjunto.
5. **Entregáveis:** relatório (fundamentação e metodologia a partir dos reports), vídeo de 1–3 min (com minimapa e oclusão), README final.
6. **Opcional:** gravação "limpa" de 1–2 min (tripé, bola colorida, campo limpo, bola rasteira, oclusões propositais), para E1–E3 "de livro" e para o vídeo de demonstração ([010](010-diagnostico-do-kalman-no-video.md), bloco B). Com outra bola, o atributo `cool` precisa ser redefinido ([015](015-falsos-positivos-de-pele-e-cor-da-bola.md)).

## Pendências de dados

- Nenhuma obrigatória.
- **Opcionais:**
  - distância da linha do gol ao muro do fundo e altura do travessão, para firmar a calibração do centro;
  - clipe curto com oclusão proposital, caso o professor exija.
