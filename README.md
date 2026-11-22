# Kalman FC

Rastreamento da bola em um jogo de futebol com um filtro de Kalman implementado do zero e projeção da trajetória em um campo virtual.

Trabalho final de Visão Computacional (UFPI, 2026.2). Dupla: Gabriel Lopes Bastos e José Victor Vieira de Oliveira.

> Em construção. O pipeline já roda de ponta a ponta (estabilização, detector, Kalman com gating e lógica de trilha, vídeo com as sobreposições). A associação na cena real ainda está sendo melhorada. Faltam também a homografia do campo, o filtro em metros e o minimapa.

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
python -m src.main videos/campo_lateral.MOV --save-video
python -m src.main videos/campo_lateral.MOV --start 30 --end 60 --show   # janela ao vivo
python -m src.main 0 --show                                              # câmera ao vivo
```

Por quadro, o pipeline faz: estabilização (H quadro → referência) → detector → candidatos levados para a referência → predição de Kalman → gating por Mahalanobis (χ²) → correção ou "no escuro" → sobreposições.

Parâmetros do filtro:

- `--sigma-a`: aceleração não modelada, em px/s² (o compromisso entre filtro teimoso e nervoso);
- `--sigma-m`: ruído da medição, em px;
- `--gate`: probabilidade do χ² usada no gating;
- `--max-dark`: quantos quadros o filtro segue só com a predição antes de perder a trilha;
- `--no-stab`: desliga a compensação da câmera.

Gera em `outputs/<vídeo>/`:

- `track.csv`: estado, medição aceita, predição, estimativa, S e H, quadro a quadro;
- `track.json`: parâmetros, FPS e a comparação com as marcações manuais (quando existirem);
- `tracked.mp4`: vídeo com medição (azul), predição e região de busca (vermelho), estimativa e rastro (laranja) e a faixa "NO ESCURO".

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
  annotate.py         marcação manual / inspeção do detector
tests/          testes automatizados (filtro × OpenCV, gating, trilha, detector, estabilização)
data/
  annotations/  verdade de campo (campo_lateral.csv em andamento)
  calib/        calibração e homografia
experiments/
  sweep_tracker.py   varredura de σa × gating contra a verdade de campo
```
