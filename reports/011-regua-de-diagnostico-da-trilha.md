# 011 — Régua de diagnóstico da trilha (`experiments/track_diagnostics.py`)

**Data:** 22/11/2026

**Código:** `experiments/track_diagnostics.py`, `src/kalman.py` (`steady_state_gain`)

**Testes:** `tests/test_diagnostics.py` (6 testes; 33 no total)

## Contexto

Os números do [010](010-diagnostico-do-kalman-no-video.md) saíram de um script avulso. Para a limpeza do filtro (`atividadeFinal/plano_limpeza_kalman.md`, tarefas 1–5), cada mudança precisa ser comparada com a mesma régua, nos dois vídeos e em F1 e F2.

## Decisão

Um experimento do repositório que lê um `outputs/<tag>/track.csv` já rodado (não roda o vídeo de novo) e mede:

- **métricas de referência** (as da tabela do [006](006-estado-atual-e-proximos-passos.md)): "na bola" no teste e no vídeo todo, aceitas com a bola invisível, erro mediano da predição no escuro, velocidade p95;
- **K em regime** para os parâmetros do `track.json`. No F2 o Σm em metros vem de J·Σm_px·Jᵀ no ponto mediano das medições, por isso o K tem dois valores (eixo lateral e eixo da profundidade);
- **distância estimativa → medição** e **tremor** (norma da 2ª diferença, só em trincas de quadros consecutivos com medição);
- **saltos** > 20 px da estimativa entre quadros consecutivos, classificados (a primeira regra que valer): trilha nova → sem medição (no escuro ou tentativa sem medição) → retomada (medição logo depois de quadros sem medição) → outro;
- **deslocamento por quadro no escuro** (mediana e p90);
- **associação:** aceitas na bola, fora da bola e com a bola invisível; bola visível sem medição; trilhas nascidas na bola;
- **episódios no escuro:** quantos terminam recuperando a medição e quantos perdem a trilha, e quantos dos recuperados duram 10–30 quadros (o que o E3 pede).

**Coordenadas:** saltos, tremor e deslocamentos ficam, por padrão, no **quadro de referência** (as colunas h00..h22 do CSV levam o quadro atual para a referência). Assim F1 e F2 ficam na mesma unidade (px a 640) e o balanço da câmera não vira movimento. `--coords cur` mede no quadro atual, como o script do 010.

**Teste** = quadros a partir da mediana dos marcados (lateral ≥ 2012, centro ≥ 1820), a mesma divisão do score combinado.

**K em regime** (`steady_state_gain`): repete predição + correção até o ganho parar de mudar. A covariância não depende das medições, então o K de regime é uma propriedade de (D, M, Σd, Σm). O teste confere a tabela do 010 e que o filtro real chega ao mesmo K depois de 300 passos.

## Evidência

Comando:

```bash
python -m experiments.track_diagnostics lateral_F1 lateral_F2 centro_F1 centro_F2 --out outputs/diag_baseline.csv
```

Rodado sobre as saídas de referência do commit `3dea9e4` (padrões: σa = 4000 px/s² ou 150 m/s², max_dark 20, gate 0,99, score do próprio vídeo).

| | Lateral F1 | Lateral F2 | Centro F1 | Centro F2 |
|---|---|---|---|---|
| Na bola (teste) | 63,7% | 64,6% | 66,3% | 68,0% |
| Na bola (vídeo todo) | 73,5% | 70,2% | 69,6% | 72,1% |
| Aceitas com a bola invisível (todo) | 11 | 13 | 9 | 10 |
| Predição no escuro, erro mediano (teste) | 38 px | 44 px | 47 px | 44 px |
| Velocidade p95 | 48 km/h | 62 km/h | 96 km/h | 49 km/h |
| K da posição em regime | 0,90 | 0,81–0,98 | 0,90 | 0,85–0,98 |
| Estimativa → medição (mediana) | 0,06 px | 0,05 px | 0,09 px | 0,08 px |
| Tremor medição / estimativa (mediana) | 0,70 / 0,60 px | 0,70 / 0,63 | 1,02 / 0,88 | 1,03 / 0,92 |
| **Saltos > 20 px** | **117** | **221** | **79** | **110** |
| … sem medição (no escuro) | 65 | 157 | 42 | 75 |
| … retomada longe | 36 | 42 | 22 | 24 |
| … trilha nova | 12 | 10 | 11 | 6 |
| … outro | 4 | 12 | 4 | 5 |
| Passo no escuro, mediana / p90 | 3,0 / 15,6 px | 3,5 / 27,0 px | 2,7 / 10,0 px | 2,7 / 16,1 px |
| Aceitas na bola / aceitas com a bola visível | 306/321 | 301/320 | 367/379 | 382/391 |
| Trilhas iniciadas | 59 | 60 | 51 | 48 |
| Episódios no escuro: recuperados / perdidos | 157/27 | 156/28 | 128/25 | 129/24 |

**Reprodução do 010** (`--coords cur`, lateral F1): 118 saltos, 12 de trilha nova, tremor 0,61 / 0,55 px, passo no escuro 3,0 / 16,2 px, 306 de 321 aceitas na bola. Batem com o 010. A divisão entre "no escuro" e "retomada" sai 66/36 contra 62/40 do script avulso: aqui um quadro de trilha **tentativa** sem medição conta como "sem medição", e a retomada exige medição no quadro.

## Leitura

- **F2 tem quase o dobro de saltos no escuro que o F1** (157 contra 65 no lateral) e um passo p90 no escuro de 27 px. No fundo do campo um erro pequeno de velocidade em m/s vira muitos px. É o alvo da Tarefa 2 (limites físicos no F2).
- **No F2 o ganho não é isotrópico:** 0,98 na profundidade (onde o Σm em metros é grande, o Σd em m/s² ainda domina) e 0,81 na lateral.
- **Os episódios no escuro quase sempre terminam com a bola recuperada** (157 de 184 no lateral F1). A predição não é inútil; o problema é como ela é desenhada (Tarefa 1) e a velocidade com que anda (Tarefas 2 e 3).
- O tremor na referência é um pouco maior que no quadro atual (0,70 contra 0,61): o erro da homografia de estabilização entra no ponto da bola.

## Consequências

- Toda tarefa da limpeza é comparada com esta tabela, nos quatro casos.
- `outputs/diag_baseline.csv` guarda a tabela de partida.
