# Kalman FC

Rastreamento da bola em um jogo de futebol com um filtro de Kalman implementado do zero e projeção da trajetória em um campo virtual.

Trabalho final de Visão Computacional (UFPI, 2026.2). Dupla: Gabriel Lopes Bastos e José Victor Vieira de Oliveira.

> Em construção. O pipeline já roda de ponta a ponta: estabilização, detector, score combinado dos candidatos, Kalman com gating e lógica de trilha (em pixels ou em metros no campo), vídeo com as sobreposições e minimapa do campo. Os experimentos obrigatórios E1–E3 têm scripts próprios (seção "Experimentos do enunciado"). Faltam o relatório e o vídeo de entrega.

## Instalação

Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Vídeos

Os vídeos ficam em `videos/` e não vão para o git (são grandes). Link para download: [Google Drive](https://drive.google.com/drive/folders/1HU5ME9qZoU7-Lw3eNfwIuzcSpIY61OWP?usp=share_link).

| Arquivo | Conteúdo |
|---|---|
| `videos/campo_lateral.MOV` | Teste: metade do campo, câmera na lateral (1920×1080, 30 fps, HEVC) |
| `videos/campo_centro.MOV` | Teste: metade do campo, câmera no meio-campo olhando o gol (1920×1080, 30 fps, HEVC) |
| `videos/*_low.mp4` | Versões reduzidas e recortadas dos mesmos vídeos (848×480), só para comparação |

O OpenCV lê os `.MOV` diretamente; não é preciso converter. Os vídeos de teste foram gravados com a **câmera na mão**, e o movimento dela é compensado por `src/stabilize.py`.

## Como rodar

Todos os comandos rodam a partir da raiz do repositório.

### Rastrear a bola (pipeline completo)

```bash
python -m src.main videos/campo_lateral.MOV --scorer data/models/ball_scorer_campo_lateral.json --save-video
python -m src.main videos/campo_lateral.MOV --scorer data/models/ball_scorer_campo_lateral.json --metric --save-video   # F2
python -m src.main videos/campo_lateral.MOV --start 30 --end 60 --show   # janela ao vivo
python -m src.main 0 --show                                              # câmera ao vivo
```

Por quadro, o pipeline faz:

1. estabilização (H quadro → referência);
2. detector;
3. score combinado (com `--scorer`): movimento compensado + auto-similaridade local + tamanho esperado; os candidatos abaixo do limiar são descartados;
4. candidatos levados para a referência;
5. predição de Kalman;
6. gating por Mahalanobis (χ²);
7. correção ou "no escuro";
8. sobreposições.

Parâmetros do filtro:

- `--sigma-a`: aceleração não modelada, em px/s² (o compromisso entre filtro teimoso e nervoso);
- `--sigma-m`: ruído da medição, em px;
- `--gate`: probabilidade do χ² usada no gating;
- `--max-dark`: quantos quadros o filtro segue só com a predição antes de perder a trilha;
- `--no-stab`: desliga a compensação da câmera.

Filtro e campo:

- `--metric`: filtro F2, com o Kalman em metros no campo. Σm de cada quadro vem da propagação do σm (px) pelo jacobiano da homografia. Sem a opção, o filtro é o F1, em px do quadro de referência;
- `--field` / `--no-field`: calibração do campo. O padrão é `data/calib/<vídeo>_field.json`. Quando ela existe, o vídeo de saída ganha o minimapa e a velocidade em km/h.

Opções do score combinado:

- `--scorer`: modelo JSON gerado por `experiments/candidate_features.py`. Sem ele, a trilha usa o score do detector;
- `--no-filter`: só reordena os candidatos, sem descartar nenhum.

Gera em `outputs/<vídeo>/`:

- `track.csv`: estado, medição aceita, predição, estimativa, S e H, quadro a quadro;
- `track.json`: parâmetros, FPS e a comparação com as marcações manuais (quando existirem);
- `tracked.mp4`: vídeo com medição (azul), predição e região de busca (vermelho), estimativa e rastro (laranja) e a faixa "NO ESCURO".

### Calibrar o campo (homografia do chão)

```bash
python -m tools.calibrate_field videos/campo_lateral.MOV
```

Lê os marcos do quadro 0 em `data/calib/<vídeo>_landmarks.json`: bases e topos das traves, retas das paredes, e as opções de calibração. Estima a pose da câmera por mínimos quadrados, com a K do iPhone 15 (calibração de Zhang das fotos; f do vídeo estimada na cena).

Gera:

- `data/calib/<vídeo>_field.json`, a calibração;
- `outputs/<vídeo>/field_overlay.jpg`, com o campo projetado sobre o vídeo, para conferência.

O mundo é definido assim: X ao longo da linha do gol (escanteios em X = 0 e X = 13 m, traves em 5 e 8 m), Y entrando no campo, Z para cima.

### Analisar um vídeo

```bash
python -m tools.analyze_video videos/campo_lateral.MOV --save-video
```

Gera em `outputs/<vídeo>/`:

- `summary.json` com as estatísticas gerais;
- `frames.csv` e `candidates.csv`, com os dados quadro a quadro;
- `plots.png`, com o número de candidatos, o movimento da câmera e o tamanho da bola por posição;
- `debug_grid.jpg`, com as máscaras intermediárias;
- `overlay.mp4`, com a borda do campo e os candidatos (melhor em azul).

Se existir um arquivo de marcações em `data/annotations/<vídeo>.csv`, o resumo inclui a comparação com ele: recall do detector, frequência com que o melhor candidato é a bola, falsos candidatos por quadro e erro do centroide.

### Compensar o movimento da câmera

```bash
python -m tools.stabilize_video videos/campo_lateral.MOV --save-video
```

Estima, para cada quadro, a homografia que o leva ao plano do quadro de referência (o primeiro quadro). O rastreamento quadro a quadro usa Lucas-Kanade e RANSAC, e a deriva é corrigida periodicamente casando o quadro atual com um quadro-chave via ORB.

Gera em `outputs/<vídeo>/`:

- `stabilization.csv` e `stabilization.json`, com as homografias e as estatísticas;
- `mosaic.jpg`, com todos os quadros projetados no plano da referência;
- `stabilized.mp4`, o vídeo projetado sobre o mosaico. Se a estabilização estiver boa, parede, gol e grama ficam parados e só os jogadores e a bola se movem.

### Marcar a bola à mão (verdade de campo)

```bash
python -m tools.annotate videos/campo_lateral.MOV --every 5
```

- **Clique esquerdo** marca a bola. **Clique direito** marca que a bola não está visível (oclusão ou fora do quadro).
- **Navegação:** `espaço` dá play/pause, `a`/`d` volta/avança um quadro e `n` vai para o próximo quadro ainda sem marcação.
- **Visualização:** `c` liga/desliga os candidatos do detector e `m` alterna entre as máscaras.
- **Sair:** `ESC`.
- A lista completa de teclas está no topo de [tools/annotate.py](tools/annotate.py).
- As marcações são salvas em `data/annotations/<vídeo>.csv`, em pixels do vídeo original.

### Varrer parâmetros da trilha

```bash
python -m experiments.sweep_tracker videos/campo_lateral.MOV --end 30
```

Roda a estabilização e o detector uma vez e depois o rastreador para cada combinação de `--sigma-a` e `--gate`. Mostra, contra as marcações manuais, a porcentagem de quadros com a estimativa na bola, a fração do tempo no escuro e as medições aceitas com a bola fora de vista. Com `--out tabela.csv`, salva a tabela.

### Comparar detectores e pistas dos candidatos

```bash
python -m experiments.candidate_features videos/campo_lateral.MOV
```

Sobre os quadros marcados, compara três geradores de candidatos: HSV, blob por LoG em escala e `SimpleBlobDetector`. Para cada candidato, calcula atributos (movimento com a câmera compensada, auto-similaridade local de Shechtman & Irani, tamanho esperado pela perspectiva) e mede quanto cada um separa a bola dos falsos (AUC).

O ajuste usa a primeira metade das marcações e a avaliação, a segunda. O script salva o modelo do score combinado em `data/models/ball_scorer_<vídeo>.json` e os resultados em `outputs/experiments/<vídeo>/`.

### Experimentos do enunciado (E1–E3)

Os três repetem só a trilha sobre os candidatos guardados em `outputs/experiments/<vídeo>/replay_cache.pkl` (a primeira execução processa o vídeo, ~7 min; depois, segundos). Tabelas e gráficos ficam em `outputs/experiments/<vídeo>/`; a análise está em `reports/`.

```bash
python -m experiments.e1_raw_vs_filtered videos/campo_lateral.MOV            # E1: detector bruto × filtrada (reports/016)
python -m experiments.e2_sigma_a videos/campo_lateral.MOV --sigma-a 300 1000 4000 --adaptive 2000:3:8   # E2: três σa (reports/014)
python -m experiments.e3_occlusion videos/campo_lateral.MOV --max-dark 20 30 # E3: oclusão de 10–30 quadros (reports/017)
```

Acrescente `--metric` ao E1 e ao E2 para o filtro em metros (F2; no E2, com `--sigma-a 10 40 150 --adaptive 40:3:16`). O E3 roda F1 e F2 juntos.

### Testes

```bash
python -m pytest -q
```

## Estrutura

```
src/
  video.py      leitura de vídeo/câmera e redimensionamento
  detector.py   detector da bola (HSV + borda do campo + morfologia + forma)
  motion.py     movimento global da câmera entre quadros (correlação de fase, diagnóstico)
  stabilize.py  homografia quadro -> referência (compensação da câmera na mão)
  motion_cue.py movimento de cada candidato com a câmera compensada
  lss.py        descritor de auto-similaridade local (Shechtman & Irani, 2007)
  blob_detector.py  detectores de blob (LoG em escala, SimpleBlobDetector), para comparação
  scoring.py    score combinado dos candidatos (regressão logística sobre as pistas)
  field.py      intrínsecos (iPhone 15), pose da câmera por marcos, homografia do chão, jacobiano
  minimap.py    campo virtual (minimapa) com a bola, o rastro e a velocidade
  kalman.py     filtro de Kalman (predição/correção em NumPy) e modelo de velocidade constante
  gating.py     região de busca: Mahalanobis, limiar χ², elipse
  tracker.py    lógica de trilha (tentativa, rastreando, no escuro, perda, reinício)
  viz.py        sobreposições nas cores da aula
  evaluation.py comparação com a verdade de campo
  synthetic.py  sequências sintéticas para testar o filtro sem vídeo
  main.py       pipeline completo (arquivo ou câmera)
tools/
  analyze_video.py    análise de um vídeo
  stabilize_video.py  estabilização + mosaico + vídeo de verificação
  calibrate_field.py  calibração do campo a partir dos marcos
  annotate.py         marcação manual / inspeção do detector
tests/          testes automatizados (filtro × OpenCV, gating, trilha, detector, estabilização, pistas, campo)
data/
  annotations/  verdade de campo (campo_lateral.csv completo; centro pendente)
  calib/        marcos (<vídeo>_landmarks.json) e calibração do campo (<vídeo>_field.json)
  models/       modelos do score combinado (ball_scorer_<vídeo>.json)
experiments/
  sweep_tracker.py        varredura de σa × gating contra a verdade de campo
  candidate_features.py   detectores e pistas dos candidatos; gera o modelo do score
  replay_tracker.py       trilha repetida sobre candidatos em cache (base dos experimentos)
  track_diagnostics.py    régua de diagnóstico de uma trilha (saltos, associação, escuro)
  e1_raw_vs_filtered.py   E1: trajetória bruta do detector × filtrada
  e2_sigma_a.py           E2: três σa (teimoso / intermediário / nervoso) e manobra
  e3_occlusion.py         E3: oclusão simulada de 10–30 quadros e oclusões naturais
```
