"""Marcação manual da bola (verdade de campo) e inspeção do detector.

Uso (a partir da raiz do repositório):

    python -m tools.annotate videos/campo_lateral.mp4            # marca 1 a cada 5 quadros
    python -m tools.annotate videos/campo_lateral.mp4 --every 10 --start 30

As marcações ficam em data/annotations/<nome do vídeo>.csv, com x e y em pixels do
vídeo ORIGINAL (não da janela), para não depender da resolução de processamento.
O arquivo é salvo a cada alteração.

Mouse:
    clique esquerdo   marca a bola neste quadro e avança --every quadros
    clique direito    marca "bola não visível" (oclusão / fora do quadro) e avança

Teclado:
    espaço            play / pause
    d  ou  →          próximo quadro          a  ou  ←    quadro anterior
    e                 avança --every quadros  q            volta --every quadros
    w / s             avança / volta 1 segundo
    n                 próximo quadro ainda sem marcação (no passo --every)
    u                 apaga a marcação deste quadro
    c                 liga/desliga os candidatos do detector
    m                 alterna máscara: nenhuma / grama / branco / candidatos
    z                 liga/desliga a lupa ao lado do cursor
    + / -             aumenta / diminui o zoom da janela
    ESC               salva e sai
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from src.detector import BallDetector, DetectorParams, draw_candidates
from src.video import open_video, processing_size, video_info

KEY_LEFT = {2424832, 65361, 81}
KEY_RIGHT = {2555904, 65363, 83}
MASKS = [None, "grass", "white", "candidates"]


class Annotations:
    """Marcações por quadro: frame -> (x, y, visível), em pixels do vídeo original."""

    FIELDS = ["frame", "t", "x", "y", "visible"]

    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = fps
        self.data: dict[int, tuple[float, float, int]] = {}
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    x = float(row["x"]) if row["x"] not in ("", "nan") else float("nan")
                    y = float(row["y"]) if row["y"] not in ("", "nan") else float("nan")
                    self.data[int(row["frame"])] = (x, y, int(row["visible"]))

    def set(self, frame: int, x: float, y: float) -> None:
        self.data[frame] = (x, y, 1)
        self.save()

    def set_invisible(self, frame: int) -> None:
        self.data[frame] = (float("nan"), float("nan"), 0)
        self.save()

    def remove(self, frame: int) -> None:
        if self.data.pop(frame, None) is not None:
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.FIELDS)
            for fr in sorted(self.data):
                x, y, vis = self.data[fr]
                w.writerow([fr, f"{fr / self.fps:.4f}",
                            "" if np.isnan(x) else f"{x:.1f}", "" if np.isnan(y) else f"{y:.1f}", vis])


class FrameCache:
    """Leitura com busca aleatória e um pequeno cache (voltar quadros é comum)."""

    def __init__(self, cap: cv2.VideoCapture, size: tuple[int, int], capacity: int = 120):
        self.cap = cap
        self.size = size
        self.capacity = capacity
        self.cache: dict[int, np.ndarray] = {}
        self.next_idx = 0

    def get(self, idx: int) -> np.ndarray | None:
        if idx in self.cache:
            return self.cache[idx]
        if idx != self.next_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.next_idx = idx + 1
        frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        if len(self.cache) >= self.capacity:
            self.cache.pop(next(iter(self.cache)))
        self.cache[idx] = frame
        return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--every", type=int, default=5, help="passo entre quadros marcados (padrão 5)")
    ap.add_argument("--start", type=float, default=0.0, help="início em segundos")
    ap.add_argument("--width", type=int, default=640, help="largura de processamento do detector")
    ap.add_argument("--zoom", type=float, default=1.5, help="zoom inicial da janela")
    ap.add_argument("--out", default=None, help="CSV de saída (padrão data/annotations/<vídeo>.csv)")
    args = ap.parse_args()

    cap = open_video(args.video)
    info = video_info(cap, args.video)
    pw, ph = processing_size(info.width, info.height, args.width)
    to_orig = info.width / pw  # janela (processamento) -> original
    out = Path(args.out) if args.out else Path("data/annotations") / f"{Path(args.video).stem}.csv"
    ann = Annotations(out, info.fps)
    frames = FrameCache(cap, (pw, ph))
    detector = BallDetector(DetectorParams())

    state = dict(idx=int(args.start * info.fps), playing=False, show_cand=True, mask=0,
                 loupe=True, zoom=args.zoom, mouse=(0, 0), dirty=True)
    last = info.n_frames - 1
    win = "kalman-fc | anotacao"

    def goto(i: int) -> None:
        state["idx"] = int(np.clip(i, 0, last))
        state["dirty"] = True

    def on_mouse(event, x, y, flags, _):
        px, py = x / state["zoom"], y / state["zoom"]
        state["mouse"] = (px, py)
        if event == cv2.EVENT_LBUTTONDOWN:
            ann.set(state["idx"], px * to_orig, py * to_orig)
            goto(state["idx"] + args.every)
        elif event == cv2.EVENT_RBUTTONDOWN:
            ann.set_invisible(state["idx"])
            goto(state["idx"] + args.every)
        elif event == cv2.EVENT_MOUSEMOVE and state["loupe"]:
            state["dirty"] = True

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    print(f"{info.path}: {info.n_frames} quadros, {info.fps:.2f} fps. Marcações em {out} "
          f"({len(ann.data)} já existentes).")

    while True:
        if state["dirty"] or state["playing"]:
            idx = state["idx"]
            frame = frames.get(idx)
            if frame is None:
                state["playing"] = False
                goto(last - 1)
                continue
            view = frame
            det = None
            if state["show_cand"] or state["mask"]:
                det = detector.detect(frame, keep_masks=bool(state["mask"]))
            if state["mask"]:
                view = cv2.cvtColor(det.masks[MASKS[state["mask"]]], cv2.COLOR_GRAY2BGR)
            if state["show_cand"] and det is not None:
                view = draw_candidates(view, det)
            view = view.copy()

            if idx in ann.data:
                x, y, vis = ann.data[idx]
                if vis:
                    c = (int(round(x / to_orig)), int(round(y / to_orig)))
                    cv2.drawMarker(view, c, (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
            n_done = len(ann.data)
            status = "MARCADO" if idx in ann.data else ("a marcar" if idx % args.every == 0 else "")
            if idx in ann.data and not ann.data[idx][2]:
                status = "NAO VISIVEL"
            txt = (f"{idx}/{last}  {idx / info.fps:7.2f}s  {status}  | marcados: {n_done}"
                   f"  | mascara: {MASKS[state['mask']] or '-'}")
            cv2.rectangle(view, (0, 0), (view.shape[1], 18), (0, 0, 0), -1)
            cv2.putText(view, txt, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            z = state["zoom"]
            disp = cv2.resize(view, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST if z >= 2 else cv2.INTER_LINEAR)
            if state["loupe"]:
                mx, my = state["mouse"]
                r = 16
                x0, y0 = int(np.clip(mx - r, 0, pw - 2 * r)), int(np.clip(my - r, 0, ph - 2 * r))
                patch = frame[y0:y0 + 2 * r, x0:x0 + 2 * r]
                lz = cv2.resize(patch, (160, 160), interpolation=cv2.INTER_NEAREST)
                cv2.drawMarker(lz, (int((mx - x0) * 5), int((my - y0) * 5)), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
                ox, oy = disp.shape[1] - 165, 25
                disp[oy:oy + 160, ox:ox + 160] = lz
                cv2.rectangle(disp, (ox, oy), (ox + 160, oy + 160), (255, 255, 255), 1)
            cv2.imshow(win, disp)
            state["dirty"] = False

        delay = max(1, int(1000 / info.fps)) if state["playing"] else 30
        key = cv2.waitKeyEx(delay)
        if state["playing"]:
            state["idx"] = min(state["idx"] + 1, last)
        if key == -1:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue
        ch = key & 0xFF if key < 256 else None
        if key == 27:
            break
        elif ch == ord(" "):
            state["playing"] = not state["playing"]
        elif ch == ord("d") or key in KEY_RIGHT:
            goto(state["idx"] + 1)
        elif ch == ord("a") or key in KEY_LEFT:
            goto(state["idx"] - 1)
        elif ch == ord("e"):
            goto(state["idx"] + args.every)
        elif ch == ord("q"):
            goto(state["idx"] - args.every)
        elif ch == ord("w"):
            goto(state["idx"] + int(round(info.fps)))
        elif ch == ord("s"):
            goto(state["idx"] - int(round(info.fps)))
        elif ch == ord("n"):
            i = (state["idx"] // args.every + 1) * args.every
            while i <= last and i in ann.data:
                i += args.every
            goto(i)
        elif ch == ord("u"):
            ann.remove(state["idx"])
            state["dirty"] = True
        elif ch == ord("c"):
            state["show_cand"] = not state["show_cand"]
            state["dirty"] = True
        elif ch == ord("m"):
            state["mask"] = (state["mask"] + 1) % len(MASKS)
            state["dirty"] = True
        elif ch == ord("z"):
            state["loupe"] = not state["loupe"]
            state["dirty"] = True
        elif ch in (ord("+"), ord("=")):
            state["zoom"] = min(4.0, state["zoom"] + 0.25)
            state["dirty"] = True
        elif ch == ord("-"):
            state["zoom"] = max(0.5, state["zoom"] - 0.25)
            state["dirty"] = True

    ann.save()
    cv2.destroyAllWindows()
    print(f"{len(ann.data)} marcações salvas em {out}")


if __name__ == "__main__":
    main()
