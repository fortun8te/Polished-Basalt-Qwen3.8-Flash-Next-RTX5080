"""Charts for Polished Basalt 0.5 vs Basalt v0.5.7 on one RTX 5080 (250 W, 256K, seed 5080)."""
import json, pathlib
here = pathlib.Path(__file__).parent
d = json.loads((here / "results.json").read_text())
BG, GRID, INK, MUTED = "#050505", "#1c1c1e", "#ffffff", "#8a8a8e"
COL = ["#48484a", "#ffffff"]
FONT = "font-family='SF Pro Display,-apple-system,Inter,Segoe UI,Helvetica,Arial,sans-serif'"

def chart(title, sub, key, fname, top):
    W, H, L, B, T = 900, 400, 70, 320, 110
    cats = list(d[key].keys())
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' {FONT}>",
           f"<rect width='{W}' height='{H}' rx='18' fill='{BG}'/>",
           f"<text x='28' y='40' fill='{INK}' font-size='20' font-weight='650'>{title}</text>",
           f"<text x='28' y='64' fill='{MUTED}' font-size='13'>{sub}</text>"]
    for i, name in enumerate(d["columns"]):
        lx = W - 300 + i * 150
        out.append(f"<rect x='{lx}' y='30' width='10' height='10' rx='2' fill='{COL[i]}'/><text x='{lx+16}' y='39' fill='{MUTED}' font-size='12'>{name}</text>")
    for k in range(5):
        v = top * k / 4; y = B - (B - T) * k / 4
        out.append(f"<line x1='{L}' x2='{W-28}' y1='{y:.1f}' y2='{y:.1f}' stroke='{GRID}'/><text x='{L-10}' y='{y+4:.1f}' fill='{MUTED}' font-size='11' text-anchor='end'>{v:,.0f}</text>")
    gw = (W - 28 - L) / len(cats); bw = gw * 0.26
    for gi, c in enumerate(cats):
        a, b = d[key][c]
        x0 = L + gi * gw + gw / 2 - bw
        for j, v in enumerate((a, b)):
            h = (B - T) * v / top; x = x0 + j * (bw + 6)
            out.append(f"<rect x='{x:.1f}' y='{B-h:.1f}' width='{bw:.1f}' height='{h:.1f}' rx='4' fill='{COL[j]}'/>")
            out.append(f"<text x='{x+bw/2:.1f}' y='{B-h-7:.1f}' fill='{INK if j else MUTED}' font-size='12' text-anchor='middle'>{v:,.1f}</text>")
        pct = (b / a - 1) * 100
        out.append(f"<text x='{x0+bw+3:.1f}' y='{B+24}' fill='{INK}' font-size='13' font-weight='600' text-anchor='middle'>{c}</text>")
        out.append(f"<text x='{x0+bw+3:.1f}' y='{B+44}' fill='{INK}' font-size='12.5' font-weight='700' text-anchor='middle'>{pct:+.1f}%</text>")
    out.append("</svg>")
    (here / fname).write_text("\n".join(out))

chart("Decode speed", "tok/s at 256K context, higher is better", "decode_tps", "decode.svg", 200)
chart("Prompt reading", "prefill tok/s, higher is better", "prefill_tps", "prefill.svg", 5000)
print("wrote decode.svg prefill.svg")
