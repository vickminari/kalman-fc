# 003 — Manter a câmera na mão e compensar com estabilização por homografia

**Data:** 21/11/2026 · **Código:** `src/stabilize.py`, `tools/stabilize_video.py` · **Testes:** `tests/test_stabilize.py`

## Contexto

O escopo previa a câmera fixa num suporte. Os vídeos de teste foram gravados com a **câmera na mão**, e a dupla decidiu manter assim **se der para fazer um bom trabalho com isso**. Isso conta como dificuldade extra na avaliação da proposta própria. Se virar impedimento (junto com o campo sujo), os vídeos serão regravados com a câmera fixa.

Câmera em movimento quebra três coisas:

1. **A subtração de fundo (MOG2):** o fundo "anda".
2. **O filtro de Kalman:** o movimento da câmera vira velocidade aparente da bola.
3. **A homografia imagem → campo:** ela deixaria de ser única para o vídeo.

**Diagnóstico** (correlação de fase entre quadros, `src/motion.py`), nos `_low`: a câmera se mexe em 54% (lateral) e 72% (centro) dos quadros, com deslocamento acumulado de centenas de pixels. Nos `.MOV` a câmera anda bem menos: ~50 px a 640 px de largura ao longo do vídeo. A maior parte do movimento visto nos `_low` vinha do recorte.

## Decisão

Estimar, para cada quadro t, a homografia **H_t: quadro t → quadro de referência** (o primeiro quadro), e:

- rodar o filtro de Kalman nas **coordenadas da referência**: as medições são levadas por H_t e a visualização volta por H_t⁻¹;
- **(futuro)** compor a homografia do campo: H_campo(t) = H_ref→campo · H_t. Assim só um quadro precisa de calibração.

**Por que uma homografia:** a câmera na mão quase só gira, e para uma rotação pura a relação entre dois quadros é exatamente uma homografia, para qualquer ponto da cena. Para pontos do chão (plano), ela vale até com translação.

**Algoritmo:**

1. **Quadro a quadro:** cantos de Shi-Tomasi (até 500), fluxo óptico de Lucas-Kanade piramidal (janela 21, 3 níveis) com **checagem ida e volta** (erro < 1 px) e homografia por **RANSAC** (2 px). A homografia é encadeada: H_t = H_{t−1} · H_{t−1→t}⁻¹.
2. **Correção da deriva:** a cada 10 quadros, ou quando há menos de 150 inliers, o quadro atual é casado por **ORB** (2000 pontos, teste da razão 0,8) com o quadro-chave de maior sobreposição. Os casamentos precisam concordar com a predição (≤ 25 px).
3. **Proteções contra correções erradas:**
   - os inliers ORB precisam cobrir ≥ 20% da área do quadro (fecho convexo);
   - com o rastreamento quadro a quadro saudável, uma correção > 10 px é **rejeitada**, a menos que **duas propostas seguidas concordem** (≤ 3 px). Isso separa deriva real de casamento errado;
   - os cantos rastreados são renovados a cada 10 quadros.
4. Um novo quadro-chave é criado quando a sobreposição com os existentes fica abaixo de 60%.

## Alternativas

- **Regravar com a câmera fixa:** fica como plano B.
- **Transformação afim ou de similaridade** em vez de homografia: não modela a perspectiva da rotação da câmera.
- **Só encadear quadro a quadro, sem correção:** a deriva acumula.
- **Correção sem proteções:** testada.
  - **Problema:** correções de até **38,9 px** (lateral, quadros 3330 e 3820) produziam homografias distorcidas (o quadro virava um trapézio), por casamentos ORB concentrados num jogador ou na grama repetitiva.
  - **Primeira tentativa:** só rejeitar correções > 10 px. Isso bloqueou a correção de uma **deriva real** (quadros 2880–2980: 11 rejeições seguidas de 10–17 px).
  - **Solução:** aceitar quando duas propostas seguidas concordam.
- **Correlação de fase** (só translação): serve como diagnóstico, não como compensação.

## Evidência

Comando: `python -m tools.stabilize_video videos/<v>.MOV --save-video`.

| | lateral | centro |
|---|---|---|
| Rastreamento quadro a quadro bem-sucedido | 100% | 100% |
| Inliers (mediana) | ~204 | ~259 |
| Correções aceitas / rejeitadas | 401 / 9 | 372 / 0 |
| Correção aceita: mediana / p95 / máx. | 2,18 / 6,33 / 12,5 px | 1,77 / 3,66 / 6,51 px |
| Velocidade | ~116 fps | ~113 fps |

- **Mosaicos** (`outputs/campo_*/mosaic.jpg`): gol, rede e parede ficam nítidos ao longo do vídeo inteiro, o que indica alinhamento consistente.
- **Teste sintético:** movimento de câmera conhecido (rotação de ±1,5° e translação), recuperado com erro < 2 px.

## Consequências e limitações

- **Erro residual:** entre correções a deriva típica é ~2 px. Isso é comparável ao ruído do detector e entra implicitamente no Σm.
- **Escala do problema:** a câmera dos `.MOV` anda pouco (~50 px). Movimentos maiores, ou um jogador ocupando metade do quadro por muito tempo, podem degradar a estimativa.
- **Abre caminho para:**
  - subtração de fundo / diferença de quadros sobre a imagem estabilizada (o próximo passo contra os candidatos falsos);
  - mapa de "desordem estática" na trilha (objetos parados ficam parados nas coordenadas da referência).
- **Relatório:** registrar como dificuldade adicional da proposta.
