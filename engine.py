"""
engine.py

Thread que lê a fonte (serial/demo/arquivo), roda o pipeline e grava no
DataStore. A interface só lê o DataStore.
"""

import threading
import traceback

import quantities

from datastore import DataStore
from pipeline import Pipeline
from sample import parse_sample


class Engine:
    def __init__(self, store: DataStore, pipeline: Pipeline):
        self.store = store
        self.pipeline = pipeline
        self._thread = None
        self._stop = threading.Event()
        self._rec_lock = threading.Lock()
        self._rec_file = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------ controle
    def start(self, source_factory) -> bool:
        self.stop()
        self.store.clear()
        self.pipeline.reset()
        try:
            source = source_factory()
        except Exception as e:  # porta inexistente, ocupada, pyserial ausente...
            self.store.set_status("Desconectado")
            self.store.set_error(f"Não foi possível abrir: {e}")
            return False

        self.pipeline.force_default_frames = not source.uses_alignment
        self.store.set_status(f"Conectado: {source.label}")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(source,), daemon=True)
        self._thread.start()
        return True

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
            self.store.set_status("Desconectado")
        self.stop_recording()

    # ------------------------------------------------------------ gravação
    def start_recording(self, path: str):
        with self._rec_lock:
            self._close_rec()
            self._rec_file = open(path, "w", encoding="utf-8")

    def stop_recording(self):
        with self._rec_lock:
            self._close_rec()

    def _close_rec(self):
        if self._rec_file:
            self._rec_file.close()
        self._rec_file = None

    @property
    def recording(self) -> bool:
        return self._rec_file is not None

    # --------------------------------------------------------------- loop
    def _run(self, source):
        prev_t_us = None
        try:
            while not self._stop.is_set():
                try:
                    line = source.readline()
                except EOFError:
                    self.store.set_status("Fim do arquivo")
                    break
                except Exception as e:
                    self.store.set_status("Desconectado")
                    self.store.set_error(f"Erro de leitura: {e}")
                    break
                if line is None:
                    continue

                with self._rec_lock:
                    if self._rec_file:
                        self._rec_file.write(line.rstrip("\r\n") + "\n")

                s = parse_sample(line)
                if s is None:
                    self.store.count_bad()
                    continue

                if prev_t_us is not None and s.t_us < prev_t_us:
                    # timestamp voltou: o ESP32 reiniciou
                    self.store.clear()
                    self.pipeline.reset()
                prev_t_us = s.t_us

                self.store.add_raw(s)
                geom = self.pipeline.process_geometry(s)
                self.store.add_geom(geom)
                pose = pos = None
                try:
                    pose = self.pipeline.process(s)
                except Exception as e:
                    # erro no filtro: mostra na interface e continua lendo
                    last = traceback.extract_tb(e.__traceback__)[-1]
                    self.store.set_error(
                        f"Filtro: {type(e).__name__}: {e} ({last.filename.split('/')[-1].split(chr(92))[-1]}:{last.lineno})")
                if pose is not None:
                    self.store.add_pose(pose)
                    pos = self.pipeline.process_position(pose, geom, s)
                    self.store.add_position(pos)
                self.store.add_history(quantities.row(s, pose, geom, pos))
        finally:
            source.close()
