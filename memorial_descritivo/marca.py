# -*- coding: utf-8 -*-
"""
============================================================================
 BRANDING  —  logo de exemplo + derivação da marca d'água
============================================================================
• Se `assets/logo.png` não existir, cria uma logo CORPORATIVA de exemplo
  (símbolo + texto), claramente substituível pela logo real do cliente.
• Deriva automaticamente a marca d'água de fundo: a mesma logo, clareada,
  para repousar atrás do texto sem prejudicar a leitura.
Troque o PNG em assets/logo.png e rode de novo: tudo se ajusta.
============================================================================
"""
import os
import math
from PIL import Image, ImageDraw, ImageFont

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def gerar_logo_exemplo(caminho, cor_hex="1F3864"):
    """Cria uma logo placeholder: hexágono + raio + wordmark."""
    cor = _hex2rgb(cor_hex)
    W, H = 1600, 460
    img = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    d = ImageDraw.Draw(img)

    # ── símbolo: hexágono com raio (bolt) ──
    cx, cy, r = 230, H // 2, 180
    hexagon = [(cx + r * math.cos(a), cy + r * math.sin(a))
               for a in [math.radians(60 * k - 90) for k in range(6)]]
    d.polygon(hexagon, fill=cor)
    # raio branco
    bolt = [(cx + 18, cy - 110), (cx - 58, cy + 18), (cx - 6, cy + 18),
            (cx - 28, cy + 120), (cx + 70, cy - 30), (cx + 12, cy - 30)]
    d.polygon(bolt, fill=(255, 255, 255, 255))

    # ── wordmark ──
    fb = _font(FONT_BOLD, 132)
    fr = _font(FONT_REG, 52)
    tx = 470
    d.text((tx, 120), "LOGOTIPO", font=fb, fill=cor)
    d.text((tx + 4, 270), "ENGENHARIA ELÉTRICA", font=fr, fill=(90, 90, 90, 255))
    # linha de acento
    d.rectangle([tx + 4, 250, tx + 760, 258], fill=cor)

    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    img.save(caminho)
    return caminho


def derivar_marca_dagua(logo_path, saida_path, intensidade=0.10):
    """Clareia a logo para uso como marca d'água (mistura com branco)."""
    logo = Image.open(logo_path).convert("RGBA")
    fundo = Image.new("RGBA", logo.size, (255, 255, 255, 255))
    composto = Image.alpha_composite(fundo, logo).convert("RGB")
    branco = Image.new("RGB", composto.size, (255, 255, 255))
    # intensidade baixa → quase branco (marca d'água suave)
    marca = Image.blend(branco, composto, intensidade)
    marca.save(saida_path)
    return saida_path
