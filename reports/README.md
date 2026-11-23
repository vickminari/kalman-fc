# Registro de decisões — Kalman FC

Decisões técnicas tomadas durante o desenvolvimento, com o contexto, as alternativas consideradas e a evidência (números medidos) que levou a cada escolha. Servem de base para o relatório final (fundamentação, metodologia e discussão) e para os dois integrantes explicarem qualquer parte do código.

> Esta pasta entra no repositório desde 22/11/2026, para os dois integrantes trabalharem com o mesmo registro. Alguns reports citam arquivos que **não** estão no repositório: os planos da dupla (`plano_de_trabalho.md`, `plano_limpeza_kalman.md`, na pasta de fora do repositório), as saídas em `outputs/` e os vídeos em `videos/` (ignorados pelo git; os vídeos estão no Drive, link no README). As saídas citadas se regeneram com os comandos indicados em cada report.

## Formato

Cada arquivo segue o mesmo esqueleto:

- **Contexto:** o problema e o que se sabia na hora.
- **Decisão:** o que foi feito.
- **Alternativas:** o que foi considerado ou testado e descartado, e por quê.
- **Evidência:** números, com o comando ou script que os gerou.
- **Consequências:** o que a decisão implica para o resto do trabalho, incluindo as limitações.

As medidas em pixels são na **resolução de processamento** (640 px de largura, ou seja, 640×360 para vídeo 16:9), salvo indicação contrária.

## Índice

| # | Decisão | Data |
|---|---|---|
| [001](001-videos-e-resolucao.md) | Vídeos originais (.MOV 1080p) e processamento em 640 px | 21/11/2026 |
| [002](002-detector.md) | Detector da bola: branco sobre a grama, sem subtração de fundo | 21/11/2026 |
| [003](003-camera-na-mao-estabilizacao.md) | Manter a câmera na mão e compensar com estabilização por homografia | 21/11/2026 |
| [004](004-filtro-de-kalman.md) | Filtro de Kalman: velocidade constante, Σd de aceleração aleatória, validação contra o OpenCV. Revisão de 22/11: σm medido (0,1–0,5 px); 1,5 px fica como ruído efetivo | 21/11/2026 |
| [005](005-gating-e-trilha.md) | Gating, associação e lógica de trilha (incluindo os experimentos de ajuste) | 21/11/2026 |
| [006](006-estado-atual-e-proximos-passos.md) | Estado atual, métricas de referência e próximos passos | 21/11/2026 |
| [007](007-candidatos-falsos-e-detectores.md) | Candidatos falsos: blob, LoG e LSS comparados; score combinado (movimento + LSS + tamanho) | 21/11/2026 |
| [008](008-videos-de-teste-como-base.md) | Os vídeos de teste viram a base; oclusão natural como a proposital; gravação definitiva adiada | 22/11/2026 |
| [009](009-geometria-do-campo-e-filtro-em-metros.md) | Geometria do campo (pose por marcos + K do iPhone 15), filtro em metros (F2) e minimapa | 22/11/2026 |
| [010](010-diagnostico-do-kalman-no-video.md) | O que o Kalman faz no vídeo (K = 0,9: associação e predição, pouca suavização) e de onde vêm as trajetórias estranhas | 22/11/2026 |
| [011](011-regua-de-diagnostico-da-trilha.md) | Régua de diagnóstico da trilha (`experiments/track_diagnostics.py`): K em regime, saltos classificados, associação, episódios no escuro | 22/11/2026 |
| [012](012-visualizacao-honesta-da-predicao.md) | Visualização honesta da predição no escuro: rastro sólido só com medição, predição tracejada e esmaecida por até 10 quadros | 22/11/2026 |
| [013](013-limites-fisicos-no-f2.md) | Limites físicos no F2: velocidade máxima de 20 m/s; `max_dark` fica 20 (E3); replay da trilha sobre candidatos em cache | 22/11/2026 |
| [014](014-e2-sigma-a-e-ruido-de-processo-adaptativo.md) | E2 (três σa) e ruído de processo adaptativo; padrões novos: F1 σa = 2000 e F2 σa = 40 m/s², com detecção de manobra | 22/11/2026 |
| [015](015-falsos-positivos-de-pele-e-cor-da-bola.md) | Falsos positivos de pele medidos; a cor da bola (`cool`, matiz fria) entra no score combinado; modelos retreinados | 22/11/2026 |
| [016](016-e1-trajetoria-crua-x-filtrada.md) | E1: detector bruto × medição aceita × estimativa; o filtro cobre as falhas (+7–12 pontos na bola) e tira ~25% do tremor | 23/11/2026 |
| [017](017-e3-oclusao.md) | E3: oclusão simulada de 10/20/30 quadros e oclusões naturais; região de busca pela covariância com teto de 60 px; `max_dark` 20 confirmado | 23/11/2026 |

## Decisões de planejamento (fora do código)

Registradas no plano de trabalho (`atividadeFinal/plano_de_trabalho.md`, fora do repositório):

- **Prazo:** entrega no SIGAA em 01/12/2026.
- **Base de dados (22/11):** os vídeos de teste viram a base do trabalho; a gravação definitiva fica adiada ([008](008-videos-de-teste-como-base.md)).
- **Enquadramento:** só metade do campo (meio-campo → gol), porque não foi possível enquadrar o campo inteiro. Posição preferencial na lateral, alternativa no meio-campo olhando o gol; os dois cenários foram gravados nos vídeos de teste.
- **Escopo:**
  - mínimo do enunciado até 27/11;
  - versão completa (F2 em metros e minimapa) se o MVP estiver pronto até 24/11;
  - extras só se sobrar tempo.
