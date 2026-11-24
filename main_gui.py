"""
main_gui.py — interface do projeto.

    python main_gui.py

Página 1 (F1): pose.      Página 2 (F2): dados brutos do ESP32.
Sem hardware: botão "Demo" (sensor simulado) ou "Arquivo…" (reproduz um CSV).
"""

from gui.app import App

if __name__ == "__main__":
    App().mainloop()
