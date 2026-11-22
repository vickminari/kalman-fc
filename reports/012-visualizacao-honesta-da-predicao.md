# 012 — Visualização honesta da predição no escuro

**Data:** 22/11/2026

**Código:** `src/viz.py` (`trail_segments`, `dashed_polyline`, `blend`, `Overlay`), `src/minimap.py` (`Minimap.draw`)

**Testes:** `tests/test_viz.py` (5 testes)

## Contexto

O [010](010-diagnostico-do-kalman-no-video.md) mostrou que boa parte das "trajetórias loucas" vistas no `tracked.mp4` é a **predição no escuro** desenhada igual à estimativa corrigida. O rastro laranja ligava todos os pontos da trilha, com ou sem medição, então:

- uma velocidade mal estimada virava uma linha reta longa saindo da bola (até 20 quadros × 16 px no p90);
- a retomada ligava a predição, já longe, à medição de volta, com outro risco reto;
- no minimapa (F2), o mesmo efeito, ampliado no fundo do campo.

O filtro não mudou nada aqui: é só **o que é desenhado**. A regra da aula continua: medição em azul, predição em vermelho, estimativa em laranja e indicação de "no escuro".

## Decisão

**No vídeo (`Overlay`):**

- **Rastro laranja sólido só entre quadros seguidos com medição aceita.** Um trecho sem medição corta o rastro, e a retomada começa um segmento novo (o salto predição → medição não vira linha).
- **Predição no escuro tracejada e esmaecida** (opacidade 0,45), a partir do último ponto medido. O ponto atual vira um círculo vazio, também esmaecido.
- **Deixa de desenhar a predição depois de `max_drawn_dark` = 10 quadros** no escuro. A trilha continua viva até `max_dark` (20), a **região de busca** (elipse vermelha) continua aparecendo, porque é onde o filtro procura, e a faixa "NO ESCURO" ganha "predicao oculta".
- **O ponto laranja cheio só aparece com correção** (estimativa de verdade). A cruz vermelha (centro previsto) some junto com a predição depois de 10 quadros.
- Trilha nova continua começando segmento novo.

**No minimapa (`Minimap`):** o mesmo, com a mesma função (`trail_segments`): rastro sólido só com medição; no escuro, pontos laranja esmaecidos e círculo vazio (vermelho no estado NO_ESCURO) por até 10 quadros.

**Por que 10 quadros:** no lateral F1, a predição anda 3 px/quadro na mediana, mas 16 px no p90 ([011](011-regua-de-diagnostico-da-trilha.md)). Em 10 quadros a predição "ruim" já saiu ~160 px da bola, mais que o suficiente para mostrar que a predição existe sem desenhar a fuga inteira. A maioria das retomadas acontece antes disso (ver Evidência).

## Alternativas

- **Não desenhar a predição no escuro:** descartado. O enunciado pede a indicação do "no escuro", e o E3 precisa mostrar a predição atravessando a oclusão.
- **Desenhar só a elipse:** a elipse sozinha não mostra para onde a predição está indo.
- **Mudar o filtro (amortecer a velocidade no escuro):** fora desta tarefa. Em px foi descartado no [005](005-gating-e-trilha.md) (quebrava a oclusão); os limites em metros são a Tarefa 2 ([013](013-limites-fisicos-no-f2.md)).

**Revisão no mesmo dia:** na primeira versão o tracejado de um trecho no escuro já encerrado continuava no rastro (45 quadros), ainda uma linha reta longa ao lado da bola recuperada (quadro 2790 do lateral). Agora **só o trecho no escuro em andamento** é desenhado: quando a bola volta ou a trilha morre, a predição daquele trecho some, e o histórico fica só com o que foi medido (`trail_segments(..., keep_past_dark=False)`, o padrão).

## Evidência

**Linhas > 20 px desenhadas no rastro** (passos entre quadros consecutivos, na referência), contadas simulando o rastro com a própria `trail_segments` sobre os logs de referência:

| | Antes: rastro laranja (tudo ligado) | Depois: sólido | Depois: tracejado esmaecido (só durante o escuro, ≤ 10 quadros) |
|---|---|---|---|
| Lateral F1 | 105 | **4** | 32 |
| Lateral F2 | 211 | **12** | 76 |
| Centro F1 | 68 | **4** | 20 |
| Centro F2 | 104 | **5** | 32 |

As linhas sólidas longas que sobraram são saltos entre duas medições seguidas (a classe "outro" do [011](011-regua-de-diagnostico-da-trilha.md): chute de verdade ou troca de alvo), e não a predição.

**Duração dos episódios no escuro que terminam recuperando a bola** (logs de referência): mediana de 1–2 quadros, p90 de 6–9 quadros; **93–94% duram até 10 quadros**. Com `max_drawn_dark` = 10, quase toda predição que "deu certo" aparece inteira; o que passa disso é sobretudo a trilha morrendo.

**Conferência visual** (`outputs/lateral_F1_t1/tracked.mp4`, `outputs/t1_clip/tracked.mp4`):

- **quadros 2772–2779:** a predição sai tracejada e esmaecida do último ponto medido, com a elipse e a cruz vermelhas; no 11º quadro a faixa passa a dizer "predicao oculta" e o tracejado para de crescer;
- **quadro 2791:** a trilha nova começa sem nenhum resto do trecho anterior;
- **quadros 2014–2036:** a predição sai pela esquerda do quadro (a bola entrou no gol); no vídeo não sobra linha, e no minimapa a predição aparece como pontos esmaecidos.

**Métricas de rastreamento:** iguais às de referência (lateral F1: 73,5% na bola no vídeo todo), como esperado, porque só o desenho mudou.

Comandos:

```bash
python -m src.main videos/campo_lateral.MOV --scorer data/models/ball_scorer_campo_lateral.json --save-video --tag lateral_F1_t1
python -m src.main videos/campo_lateral.MOV --scorer data/models/ball_scorer_campo_lateral.json --save-video --tag t1_clip --start 91.5 --end 93.5
```

## Consequências

- O vídeo mostra o que o filtro **sabe** (rastro sólido) separado do que ele **supõe** (tracejado). Isso também é material para a discussão do E3 no relatório.
- As métricas de rastreamento não mudam (a trilha é a mesma).
