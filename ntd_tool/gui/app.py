# ntd_tool/gui/app.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from pathlib import Path
from ntd_tool.classify.inclusion_classifier import run_inclusion_model
from ntd_tool.classify.disease_classifier import run_disease_model
from ntd_tool.export.export_results import export_results
from ntd_tool.utils.deduplication import merge_and_deduplicate
from ntd_tool.data_fetch.arxiv_fetcher import fetch_arxiv
from ntd_tool.data_fetch.crossref_fetcher import fetch_crossref
from ntd_tool.data_fetch.pubmed_fetcher import fetch_pubmed
import datetime


class NTDToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NTD Tool - Proof of Concept")

        self.output_dir = ""
        self.models_used = []

        self.create_widgets()

    def create_widgets(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frm, text="📅 Start Date (YYYY-MM-DD):").grid(column=0, row=0, sticky="w")
        self.start_date_var = tk.StringVar(value=str(datetime.date.today()))
        ttk.Entry(frm, textvariable=self.start_date_var).grid(column=1, row=0, sticky="w")

        ttk.Label(frm, text="📅 End Date (YYYY-MM-DD):").grid(column=0, row=1, sticky="w")
        self.end_date_var = tk.StringVar(value=str(datetime.date.today()))
        ttk.Entry(frm, textvariable=self.end_date_var).grid(column=1, row=1, sticky="w")

        ttk.Label(frm, text="📁 Choose export folder:").grid(column=0, row=2, sticky="w")
        self.output_button = ttk.Button(frm, text="Browse", command=self.select_output_dir)
        self.output_button.grid(column=1, row=2, sticky="w")

        self.incl_svm_var = tk.BooleanVar()
        self.incl_bert_var = tk.BooleanVar()
        incl_frame = ttk.LabelFrame(frm, text="🧠 Inclusion Model")
        incl_frame.grid(column=0, row=3, columnspan=2, pady=5, sticky="ew")
        ttk.Checkbutton(incl_frame, text="Use SVM", variable=self.incl_svm_var).pack(anchor="w")
        ttk.Checkbutton(incl_frame, text="Use BERT", variable=self.incl_bert_var).pack(anchor="w")

        self.disease_svm_var = tk.BooleanVar()
        self.disease_bert_var = tk.BooleanVar()
        self.threshold_var = tk.StringVar(value="0.5")
        disease_frame = ttk.LabelFrame(frm, text="🦠 Disease Model")
        disease_frame.grid(column=0, row=4, columnspan=2, pady=5, sticky="ew")
        ttk.Checkbutton(disease_frame, text="Use SVM", variable=self.disease_svm_var).pack(anchor="w")
        ttk.Checkbutton(disease_frame, text="Use BERT", variable=self.disease_bert_var).pack(anchor="w")
        th_frame = ttk.Frame(disease_frame)
        th_frame.pack(anchor="w")
        ttk.Label(th_frame, text="Threshold:").pack(side="left")
        ttk.Entry(th_frame, textvariable=self.threshold_var, width=5).pack(side="left")

        self.run_button = ttk.Button(frm, text="🚀 Run Pipeline", command=self.run_pipeline)
        self.run_button.grid(column=0, row=5, pady=10, sticky="w")

        self.log_box = tk.Text(frm, height=15, width=80, wrap="word")
        self.log_box.grid(column=0, row=6, columnspan=2, pady=5)

    def log(self, msg):
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.root.update()

    def select_output_dir(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_dir = folder
            self.log(f"Selected output folder: {self.output_dir}")

    def run_pipeline(self):
        if not self.output_dir:
            messagebox.showerror("Error", "Please select an output folder.")
            return

        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid threshold value.")
            return

        self.models_used = []

        try:
            start = self.start_date_var.get()
            end = self.end_date_var.get()

            self.log("📥 Fetching from arXiv…")
            arxiv_path = fetch_arxiv(start, end)

            self.log("📥 Fetching from CrossRef…")
            crossref_path = fetch_crossref(start, end)

            self.log("📥 Fetching from PubMed…")
            pubmed_path = fetch_pubmed(start, end)

            self.log("🔄 Deduplicating…")
            df = merge_and_deduplicate([arxiv_path, crossref_path, pubmed_path])

            if self.incl_svm_var.get():
                df = run_inclusion_model(df, model="svm")
                self.models_used.append("svm")

            if self.incl_bert_var.get():
                df = run_inclusion_model(df, model="bert")
                self.models_used.append("bert")

            if self.disease_svm_var.get():
                df = run_disease_model(df, model="svm", threshold=threshold)

            if self.disease_bert_var.get():
                df = run_disease_model(df, model="bert", threshold=threshold)

            self.log("💾 Exporting results…")
            export_path = Path(self.output_dir) / "ntd_results.xlsx"
            export_results(df, used_models=self.models_used, output_path=str(export_path))

            self.log(f"✅ Done! Results saved to {export_path}")

        except Exception as e:
            self.log(f"❌ Error: {e}")
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = NTDToolApp(root)
    root.mainloop()
