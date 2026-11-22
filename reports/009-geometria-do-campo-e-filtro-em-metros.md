# 009 — Geometria do campo, filtro em metros (F2) e minimapa

**Data:** 22/11/2026

**Código:** `src/field.py`, `src/minimap.py`, `tools/calibrate_field.py`, `src/main.py` (`--metric`, `--field`)

**Dados:**

- `data/calib/<vídeo>_landmarks.json` (marcos e opções de calibração);
- `data/calib/<vídeo>_field.json` (calibração resultante).

**Testes:** `tests/test_field.py`

## Contexto

O "campo virtual" é a proposta da dupla, não o mínimo do enunciado. Para levar a bola da imagem para metros é preciso a **homografia do chão**.

**O que tínhamos:**

- as medidas do Gabriel: escanteio → trave 5 m, trave → trave 3 m, escanteio a escanteio 13 m;
- a calibração de Zhang do iPhone 15 (fotos HEIC 4284×5712, retrato): fx = 4389,5, fy = 4393,5, cx = 2144,2, cy = 2846,2, k1 = 0,155, k2 = −0,540, RMSE de 1,6 px;
- a matriz do ARKit para quadros de 1920×1440, com f ≈ 1478 px.

**Problema:** todos os pontos medidos estão **na linha do gol** (colineares). Com pontos alinhados não se obtém a homografia do chão.

## Decisão

**1. Mundo:**

- X na linha do gol: escanteios em X = 0 e X = 13, traves em X = 5 e X = 8;
- Y entrando no campo;
- Z para cima.

O sistema é destro: olhando para o gol de dentro do campo, X cresce para a esquerda.

**2. Calibração por pose da câmera** (`src/field.calibrate`), e não por homografia de 4 pontos. Com K conhecida, estimamos R e t por mínimos quadrados no erro de reprojeção, usando marcos com geometria conhecida mesmo sem medida:

- **bases das traves:** pontos 3D conhecidos;
- **topos das traves:** altura h do travessão, uma **incógnita compartilhada**;
- **eixo da trave:** X e Y fixos, Z livre (é vertical);
- **base da parede lateral:** perpendicular à linha do gol (X fixo, Y livre);
- **base do muro do fundo:** paralela à linha do gol (Y = d, incógnita compartilhada).

O resultado é H_mundo→imagem = K [r1 r2 t].

**Como os marcos foram obtidos:** as traves foram medidas por inspeção ampliada do quadro 0 (recortes de 5–6× com régua de pixels); as paredes, por uma reta RANSAC na borda da grama. A ferramenta interativa para a dupla refinar os marcos fica como próximo passo.

**3. Distância focal do vídeo: f = 1743 px**, estimada na cena do lateral, e não os 1477 px derivados das fotos.

- **Derivação a partir das fotos:** paisagem, largura total do sensor 5712 → 1920, recorte 16:9 → f ≈ 1476 px, coerente com o ARKit (~1478 px).
- **Na cena** (lateral, bem condicionado), com f livre: f = **1743 px**, erro de reprojeção de 2,28 px. Com f = 1477 fixo, o erro é de 3,62 px.
- **Interpretação:** o vídeo usa um recorte ~18% maior que as fotos, compatível com a estabilização eletrônica do iPhone.
- **Checagem pelo tamanho da bola** (diâmetro detectado / esperado para 0,22 m na posição calibrada, bolas verdadeiras do lateral): 1,10 com f = 1477; 1,04 com f = 1743. O borrão aumenta o diâmetro detectado, então o teste é só indicativo, mas favorece o f maior.

**4. Distorção desprezada.** A base da parede lateral, de x = 800 a 1900 px, é reta na imagem: resíduo de 1,45 px, flecha de ~3 px. Os coeficientes das fotos (k1 = 0,155) curvariam bastante as bordas. O iPhone já corrige a distorção no vídeo.

**5. Bola no plano Z = 0,11 m** (o raio), não no gramado. O centro detectado da bola está a essa altura; projetá-lo em Z = 0 jogava bolas encostadas na parede para "atrás da parede" (no lateral, 128 de 353 bolas caíam em X < 0 antes da correção).

**6. Vídeo do centro: calibração com priors.**

- **Por que só as traves não bastam:** elas ficam num plano quase de frente para a câmera; a pose não fica determinada e a câmera sai **no chão** (Z ≈ 0,0–0,4 m) para qualquer f.
- **Por que o muro do fundo não resolve:** a base dele é quase paralela, na imagem, à linha das traves. Ele fixa a direção, mas não a distância d (o ajuste a leva ao infinito).
- **Priors usados:**
  - **altura da câmera ~ 1,2 ± 0,25 m:** mesmo cinegrafista, que no lateral filmou a 1,07–1,22 m;
  - **d ~ −1,6 ± 0,4 m:** a borda atrás do gol, vista na calibração do lateral, fica a 1,0–2,0 m da linha.
- **Com f do lateral e os priors:** erro de 1,98 px, câmera em (7,0; 11,9; 1,33) m, d = −2,0 m.

**7. Filtro F2 em metros** (`--metric`): candidatos levados ao campo pela homografia (plano da bola); Σm de cada quadro = J Σm_px Jᵀ, com J = ∂(X, Y)/∂(u, v) no ponto da trilha. Parâmetros em metros (`TrackerParams.metric`):

| Parâmetro | Valor |
|---|---|
| σa | 150 m/s² |
| σv0 | 12 m/s |
| teto da região de busca | 2,5 m |
| raio de "parada" | 0,15 m |
| célula da desordem | 0,25 m |

Valores convertidos dos em px por ~0,04 m/px.

**8. Minimapa** (`src/minimap.py`): metade do campo vista de cima, com o gol em cima. Mostra traves, laterais, a câmera, a região do chão visível, a bola (laranja: estimativa; azul: medição), o rastro e a velocidade em km/h. Ativado sempre que existe `data/calib/<vídeo>_field.json`.

## Evidência

**Calibrações** (`python -m tools.calibrate_field videos/<v>.MOV`, sobreposição em `outputs/<v>/field_overlay.jpg`):

| | Lateral | Centro |
|---|---|---|
| f (px) | 1743,2 (estimado) | 1743,2 (do lateral) |
| Erro de reprojeção | 2,28 px | 1,98 px |
| Altura do travessão h | **2,09 m** | 2,25 m |
| Câmera (X, Y, Z) | (12,8; 3,8; 1,22) m | (7,0; 11,9; 1,33) m |
| Outras incógnitas | — | d = −2,02 m |

**Checagens independentes:**

- **h ≈ 2,1 m:** um gol de futsal ou society tem 2 m, e h não foi informado ao ajuste.
- **Altura da câmera ≈ 1,2 m:** celular na mão.
- **Sobreposição visual:** a grade do campo, a linha do gol, as laterais e as traves projetadas caem sobre o vídeo.
- **Centro menos confiável:** h dá 2,25 m contra 2,09 m no lateral.

**Teste sintético:** câmera conhecida (f = 1700, h = 2 m), partindo de f = 1476. Recupera f com erro < 20 px, h com erro < 5 cm e a posição da câmera com erro < 15 cm. O jacobiano analítico confere com diferenças finitas.

**F1 × F2 (E6)**, com score combinado, vídeo inteiro, % na bola (raio de 10 px no quadro atual):

| Vídeo | Filtro | Na bola (teste) | Na bola (todo) | Aceitas com a bola invisível | Velocidade mediana / p95 |
|---|---|---|---|---|---|
| Lateral | F1 (px) | 63,7% | 73,5% | 11 | 5,2 / 48,2 km/h* |
| Lateral | **F2 (m)** | **64,6%** | 70,2% | 13 | 5,0 / 61,6 km/h |
| Centro | F1 (px) | 66,3% | 69,6% | 9 | 5,7 / 96,5 km/h* |
| Centro | **F2 (m)** | **68,0%** | 72,1% | 10 | 5,9 / 49,2 km/h |

\*No F1, a velocidade em km/h é convertida da velocidade em px pela homografia. No fundo do campo um px vale muitos metros, e o p95 de 96 km/h do centro não é plausível. No F2 a velocidade é estado do filtro, em m/s.

Comando: `python -m src.main videos/<v>.MOV --scorer data/models/ball_scorer_<v>.json [--metric] --save-video`.

## Consequências e limitações

- **F2 e F1 rastreiam igualmente bem;** a diferença está na interpretação física: velocidade em km/h plausível e região de busca coerente com a geometria (maior no fundo). Fica disponível o E6 do escopo.
- **Bola no ar:** a homografia vale para o plano Z = 0,11 m. Bola alta ou encostada na parede aparece deslocada no minimapa. É a limitação prevista na proposta (extra 7.5).
- **O centro depende de priors.** Uma medida a mais resolveria: a distância da linha do gol até o muro do fundo, ou a altura do travessão para conferir h.
- **Os marcos foram medidos por inspeção ampliada,** com erro estimado de ±2–4 px. Uma ferramenta interativa para a dupla refinar os marcos ainda não existe (próximo passo).
- **A calibração vale para o quadro 0.** O `main.py` estabiliza desde o quadro 0 mesmo com `--start > 0`.
- **Prior de tamanho do score:** o campo agora permite calcular o tamanho esperado da bola pela geometria (`FieldModel.expected_diameter_px`), o caminho para tornar o score combinado independente do vídeo. Ainda não usado.
