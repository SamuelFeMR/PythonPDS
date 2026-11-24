"""gui/theme.py — cores e estilos (tema escuro)."""

from tkinter import ttk

BG = "#1b1d23"
PANEL = "#242730"
PANEL2 = "#2c303a"
FG = "#e7e9ee"
MUTED = "#9aa1b2"
ACCENT = "#4f8cff"
GOOD = "#3ecf8e"
BAD = "#ff5d5d"
WARN = "#ffc857"
CANVAS_BG = "#14161b"
GRID = "#2e323d"
SERIES = ["#ff6b6b", "#4ecb71", "#4f8cff", "#ffc857"]

FONT = ("Segoe UI", 10)
FONT_MONO = ("Consolas", 11)


def apply_style(root):
    root.configure(bg=BG)
    st = ttk.Style(root)
    st.theme_use("clam")

    st.configure(".", background=BG, foreground=FG, font=FONT)
    st.configure("TFrame", background=BG)
    st.configure("Panel.TFrame", background=PANEL)
    st.configure("Top.TFrame", background=PANEL)
    st.configure("TLabel", background=BG, foreground=FG)
    st.configure("Panel.TLabel", background=PANEL, foreground=FG)
    st.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
    st.configure("H.TLabel", background=PANEL, foreground=ACCENT, font=("Segoe UI", 10, "bold"))
    st.configure("Mono.TLabel", background=PANEL, foreground=FG, font=FONT_MONO)
    st.configure("Status.TLabel", background=PANEL, foreground=MUTED)

    st.configure("TButton", background=PANEL2, foreground=FG, padding=(10, 5),
                 borderwidth=0, focusthickness=0)
    st.map("TButton", background=[("active", "#3a3f4c"), ("pressed", "#3a3f4c")])
    st.configure("Nav.TButton", background=PANEL, foreground=MUTED,
                 padding=(18, 9), font=("Segoe UI", 11, "bold"))
    st.map("Nav.TButton", background=[("active", PANEL2)], foreground=[("active", FG)])
    st.configure("NavActive.TButton", background=ACCENT, foreground="white",
                 padding=(18, 9), font=("Segoe UI", 11, "bold"))
    st.map("NavActive.TButton", background=[("active", ACCENT)])
    st.configure("Rec.TButton", background=BAD, foreground="white")
    st.map("Rec.TButton", background=[("active", BAD)])

    st.configure("TEntry", fieldbackground=PANEL2, foreground=FG, insertcolor=FG, padding=3)
    st.configure("TCheckbutton", background=PANEL, foreground=FG)
    st.map("TCheckbutton", background=[("active", PANEL)])
    st.configure("TCombobox", fieldbackground=PANEL2, background=PANEL2,
                 foreground=FG, arrowcolor=FG)
    st.map("TCombobox", fieldbackground=[("readonly", PANEL2)],
           foreground=[("readonly", FG)])
    st.configure("Horizontal.TScale", background=PANEL, troughcolor=PANEL2)
    root.option_add("*TCombobox*Listbox.background", PANEL2)
    root.option_add("*TCombobox*Listbox.foreground", FG)
