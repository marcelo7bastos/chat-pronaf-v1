"""
Compatibilidade Chroma ↔ pysqlite3.
Importe *antes* de qualquer coisa que use sqlite3.
"""
import sys, importlib, platform

def patch_sqlite():
    if platform.system() != "Windows":  # Linux, macOS, Streamlit Cloud
        try:
            import pysqlite3                # wheel ≥ 0.5.3
            sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
            importlib.invalidate_caches()
        except ModuleNotFoundError:
            # roda com sqlite3 da stdlib mesmo
            pass
