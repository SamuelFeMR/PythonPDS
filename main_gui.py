"""
main_gui.py — interface do projeto.

    python main_gui.py                      # abre a janela; escolha a porta lá
    python main_gui.py --port COM7          # já conecta na porta
    python main_gui.py --port COM7 --baud 921600
    python main_gui.py --demo               # sensor simulado, sem hardware

Tela única: linha de dados brutos no topo, cena 3D (placa + esfera com H e α) e gráficos.
Parâmetros do filtro: filter_config.json.  Geometria e calibração do ToF: geometry_config.json.
"""

import argparse

from gui.app import App


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="porta serial (ex.: COM7); conecta ao abrir")
    ap.add_argument("--baud", type=int, help="baud rate (padrão da janela: 115200)")
    ap.add_argument("--demo", action="store_true", help="usa o sensor simulado")
    args = ap.parse_args()
    App(port=args.port, baud=args.baud, demo=args.demo).mainloop()


if __name__ == "__main__":
    main()
