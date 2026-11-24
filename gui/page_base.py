"""gui/page_base.py — contrato de uma página."""

from tkinter import ttk


class Page(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app

    def on_show(self):
        pass

    def on_hide(self):
        pass

    def refresh(self):
        """Chamado ~25x/s enquanto a página está visível."""
