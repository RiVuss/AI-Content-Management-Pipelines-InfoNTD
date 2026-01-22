# ntd_tool/gui/app2.py
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import datetime, time
from pathlib import Path
import pandas as pd
from functools import reduce
import threading, queue    
import torch

from ntd_tool.utils import settings
from ntd_tool.data_fetch.arxiv_fetcher import fetch_arxiv
from ntd_tool.data_fetch.crossref_fetcher import fetch_crossref
from ntd_tool.data_fetch.pubmed_fetcher import fetch_pubmed
from ntd_tool.utils.deduplication import merge_and_deduplicate
from ntd_tool.classify.inclusion_classifier import run_inclusion_model
from ntd_tool.classify.disease_classifier import run_disease_model
from ntd_tool.export.export_results import export_results


class NTDToolApp2:
    # --------------------------- GUI ------------------------------- #
    def __init__(self, root):
        self.root = root
        self.root.title("NTD Tool")

        self.output_dir = ""
        self._log_q = queue.Queue()   # ← NEW
        self._make_widgets()
        self._flush_log()             # ← start polling

    def _make_widgets(self):
        frm = ttk.Frame(self.root, padding=10); frm.grid(row=0, column=0, sticky="nsew")

        # dates
        ttk.Label(frm, text="Start (YYYY-MM-DD):").grid(column=0, row=0, sticky="w")
        self.start_date_var = tk.StringVar(value=str(datetime.date.today()))
        ttk.Entry(frm, textvariable=self.start_date_var, width=12).grid(column=1, row=0, sticky="w")

        ttk.Label(frm, text="End (YYYY-MM-DD):").grid(column=0, row=1, sticky="w")
        self.end_date_var = tk.StringVar(value=str(datetime.date.today()))
        ttk.Entry(frm, textvariable=self.end_date_var, width=12).grid(column=1, row=1, sticky="w")

        # output dir
        ttk.Label(frm, text="Export folder:").grid(column=0, row=2, sticky="w")
        ttk.Button(frm, text="Browse", command=self._choose_dir).grid(column=1, row=2, sticky="w")

        # sources
        src = ttk.LabelFrame(frm, text="Data Sources"); src.grid(column=0, row=3, columnspan=2, pady=5, sticky="ew")
        self.use_pubmed, self.use_crossref, self.use_arxiv = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        for i, (lab, var) in enumerate([("PubMed", self.use_pubmed), ("CrossRef", self.use_crossref), ("arXiv", self.use_arxiv)]):
            ttk.Checkbutton(src, text=lab, variable=var).grid(column=i, row=0, sticky="w", padx=5)

        # inclusion
        inc = ttk.LabelFrame(frm, text="Inclusion Model"); inc.grid(column=0, row=4, columnspan=2, pady=5, sticky="ew")
        self.incl_svm_var, self.incl_bert_var = tk.BooleanVar(), tk.BooleanVar()
        for i, (lab, var) in enumerate([("SVM", self.incl_svm_var), ("BERT", self.incl_bert_var)]):
            ttk.Checkbutton(inc, text=lab, variable=var).grid(column=i, row=0, sticky="w", padx=5)
        ttk.Label(inc, text="Threshold:").grid(column=2, row=0, sticky="e")
        self.th_var = tk.StringVar(value="0.5")
        ttk.Entry(inc, textvariable=self.th_var, width=5).grid(column=3, row=0, sticky="w")

        # disease
        dis = ttk.LabelFrame(frm, text="Disease Model"); dis.grid(column=0, row=5, columnspan=2, pady=5, sticky="ew")
        self.dis_svm_var, self.dis_bert_var, self.dis_gem_var = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        for i, (lab, var) in enumerate([("SVM", self.dis_svm_var), ("BERT", self.dis_bert_var), ("Gemini", self.dis_gem_var)]):
            ttk.Checkbutton(dis, text=lab, variable=var).grid(column=i, row=0, sticky="w", padx=5)
        gembar = ttk.Frame(dis); gembar.grid(column=0, row=1, columnspan=3, sticky="w")
        ttk.Button(gembar, text="Set Gemini API-key", command=self._set_gem_key).pack(side="left", padx=4)
        ttk.Button(gembar, text="Edit Gemini prompt", command=self._edit_gem_prompt).pack(side="left", padx=4)

        # run
        ttk.Button(frm, text="🚀 Run Pipeline", command=self._run).grid(column=0, row=6, pady=10, sticky="w")

        # log
        self.log = tk.Text(frm, width=40, height=16, wrap="word"); self.log.grid(column=0, row=7, columnspan=2, pady=4)

    # ---------------- settings dialogs ---------------- #
    def _set_gem_key(self):
        cur = settings.load_setting("gemini_api_key") or ""
        key = simpledialog.askstring("Gemini API-key", "Enter / update your Google AI key:", initialvalue=cur, show="*")
        if key is not None:
            settings.save_setting("gemini_api_key", key.strip())
            self._log("✅ Gemini API-key saved.")

    def _edit_gem_prompt(self):
        cur = settings.load_setting("gemini_prompt") or ""
        prompt = simpledialog.askstring("Gemini Prompt", "Edit prompt (use {title} {abstract}):", initialvalue=cur)
        if prompt is not None:
            settings.save_setting("gemini_prompt", prompt.strip())
            self._log("✅ Prompt saved.")

    # --------------- log helpers ---------------------- #
    def _log(self, txt):
        """Thread-safe: enqueue a message for the GUI thread."""
        self._log_q.put(txt)

    def _flush_log(self):
        """Poll queue and update the Text widget."""
        try:
            while True:
                msg = self._log_q.get_nowait()
                self.log.insert("end", msg + "\n")
                self.log.see("end")
        except queue.Empty:
            pass
        self.root.after(100, self._flush_log)   # poll every 100 ms

    # --------------- misc helpers --------------------- #
    def _choose_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir = path
            self._log(f"Export folder: {path}")

    def _fetch_source(self, label: str, fn, *args):
        """Download one data source, log rows when possible, and
        return exactly what the original fetch_* function returns."""
        self._log(f"→ {label}: downloading…")
        res = fn(*args)                       # whatever the fetcher returns

        df = None
        # If the fetcher gave back a DataFrame (or tuple with DF first),
        # we can count rows without breaking anything.
        if isinstance(res, pd.DataFrame):
            df = res
        elif isinstance(res, tuple) and isinstance(res[0], pd.DataFrame):
            df = res[0]

        if df is not None:
            self._log(f"   ✓ {label}: {len(df)} rows")
        else:
            self._log(f"   ✓ {label}: done")

        return res       


    # -------------------- PIPELINE -------------------- #
    def _run(self):
        """Kick off the pipeline in a background thread."""
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        if not self.output_dir:
            self._log("❌ Choose an export folder first.")
            messagebox.showerror("No folder", "Choose an export folder first.")
            return
        try:
            # FETCH
            s, e = self.start_date_var.get(), self.end_date_var.get()
            self._log(f"📥 Fetching data {s} → {e}")
            sources = []
            if self.use_arxiv.get():
                sources.append(self._fetch_source("arXiv",   fetch_arxiv,   s, e, "temp/arxiv.csv"))
            if self.use_crossref.get():
                sources.append(self._fetch_source("CrossRef", fetch_crossref, s, e, "temp/crossref.csv"))
            if self.use_pubmed.get():
                sources.append(self._fetch_source("PubMed",   fetch_pubmed,  s, e, "temp/pubmed.csv"))
            df_base = merge_and_deduplicate(sources)
            self._log(f"✅ Total unique papers found: {len(df_base)} rows")

            # INCLUSION
            self._log("🔎 Inclusion step…")
            th = float(self.th_var.get())
            incl = {}
            if self.incl_svm_var.get():
                self._log("   • SVM")
                incl["svm"] = run_inclusion_model(df_base.copy(), "svm") \
                              .rename(columns={"inclusion_prob": "inclusion_prob_svm"})
            if self.incl_bert_var.get():
                self._log("   • BERT")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._log(f"     🔧 BERT model running on: {device}")
                incl["bert"] = run_inclusion_model(df_base.copy(), "bert") \
                               .rename(columns={"inclusion_prob": "inclusion_prob_bert"})
            if not incl:
                raise RuntimeError("Select at least one inclusion model.")

            core = ["title", "abstract", "doi", "_source"]
            def mrg(a, b): return pd.merge(a, b, on=core, how="outer")
            merg = reduce(mrg, incl.values())
            keep = merg[[c for c in merg if c.startswith("inclusion_prob_")]].ge(th).any(axis=1)
            filt = merg[keep].copy()
            self._log(f"📊 Rows ≥{th}: {len(filt)}")

            # DISEASE
            self._log("🩺 Disease classification…")
            dis_res = dict.fromkeys(["svm", "bert", "gemini"])
            def _run_dis(name, **kw):
                self._log(f"   • {name.capitalize()}")
                df = run_disease_model(filt.copy(), model=name, **kw)
                pc = f"inclusion_prob_{name}"
                if pc in df.columns:
                    df.rename(columns={pc: "inclusion_prob"}, inplace=True)
                dis_res[name] = df
                self._log(f"     ✓ {name.capitalize()} done")

            if self.dis_svm_var.get():  _run_dis("svm")
            if self.dis_bert_var.get(): 
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._log(f"   • BERT\n     🔧 Disease model running on: {device}")
                _run_dis("bert")
            if self.dis_gem_var.get():
                key = settings.load_setting("gemini_api_key") or ""
                if not key:
                    raise RuntimeError("Gemini selected but API-key missing (use button).")
                _run_dis("gemini",
                         api_key=key,
                         prompt=settings.load_setting("gemini_prompt"),
                         gemini_model=settings.load_setting("gemini_model"))

            # ensure every model key exists
            for m, dfm in incl.items():
                if dis_res[m] is None:
                    base = filt[["title", "abstract", "doi", "_source", f"inclusion_prob_{m}"]].copy()
                    base.rename(columns={f"inclusion_prob_{m}": "inclusion_prob"}, inplace=True)
                    dis_res[m] = base

            # EXPORT
            self._log("💾 Exporting to Excel…")
            out = Path(self.output_dir) / "ntd_results.xlsx"
            export_results(dis_res["svm"], dis_res["bert"], dis_res["gemini"], str(out))
            self._log(f"✅ Excel saved to {out}")
            self._log("🎉 Pipeline finished")

        except Exception as exc:
            self._log(f"❌ {exc}")
            messagebox.showerror("Error", str(exc))


if __name__ == "__main__":
    root = tk.Tk()
    NTDToolApp2(root)
    root.mainloop()
