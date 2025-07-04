# ntd_tool/gui/app2.py
import tkinter as tk
from tkinter import messagebox, ttk
import datetime

class NTDToolApp2:
    def __init__(self, root):
        self.root = root
        self.root.title("NTD Tool - GUI")
        self.create_widgets()

    def create_widgets(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")

        # Date Range Inputs
        ttk.Label(frm, text="📅 Start Date (YYYY-MM-DD):").grid(column=0, row=0, sticky="w")
        self.start_date_var = tk.StringVar(value=str(datetime.date.today()))
        ttk.Entry(frm, textvariable=self.start_date_var).grid(column=1, row=0, sticky="w")

        ttk.Label(frm, text="📅 End Date (YYYY-MM-DD):").grid(column=0, row=1, sticky="w")
        self.end_date_var = tk.StringVar(value=str(datetime.date.today()))
        ttk.Entry(frm, textvariable=self.end_date_var).grid(column=1, row=1, sticky="w")

        # Data Sources
        source_frame = ttk.LabelFrame(frm, text="📚 Data Sources")
        source_frame.grid(column=0, row=2, columnspan=2, pady=5, sticky="ew")
        self.use_pubmed = tk.BooleanVar()
        self.use_crossref = tk.BooleanVar()
        self.use_arxiv = tk.BooleanVar()
        for i, (label, var) in enumerate([("PubMed", self.use_pubmed), ("CrossRef", self.use_crossref), ("arXiv", self.use_arxiv)]):
            ttk.Checkbutton(source_frame, text=label, variable=var).grid(column=i, row=0, sticky="w", padx=5)

        # Inclusion Model Options
        incl_frame = ttk.LabelFrame(frm, text="🧠 Inclusion Model")
        incl_frame.grid(column=0, row=3, columnspan=2, pady=5, sticky="ew")
        self.incl_svm_var = tk.BooleanVar()
        self.incl_bert_var = tk.BooleanVar()
        for i, (label, var) in enumerate([("Use SVM", self.incl_svm_var), ("Use BERT", self.incl_bert_var)]):
            ttk.Checkbutton(incl_frame, text=label, variable=var).grid(column=i, row=0, sticky="w", padx=5)

        # Disease Model Options
        disease_frame = ttk.LabelFrame(frm, text="🦠 Disease Model")
        disease_frame.grid(column=0, row=4, columnspan=2, pady=5, sticky="ew")
        self.disease_svm_var = tk.BooleanVar()
        self.disease_bert_var = tk.BooleanVar()
        self.disease_gemini_var = tk.BooleanVar()
        for i, (label, var) in enumerate([("Use SVM", self.disease_svm_var), ("Use BERT", self.disease_bert_var), ("Use Gemini", self.disease_gemini_var)]):
            ttk.Checkbutton(disease_frame, text=label, variable=var).grid(column=i, row=0, sticky="w", padx=5)

        th_frame = ttk.Frame(disease_frame)
        th_frame.grid(column=0, row=1, columnspan=3, sticky="w", pady=5)
        self.threshold_var = tk.StringVar(value="0.5")
        ttk.Label(th_frame, text="Threshold:").pack(side="left")
        ttk.Entry(th_frame, textvariable=self.threshold_var, width=5).pack(side="left")

        # Gemini Controls
        gemini_frame = ttk.LabelFrame(frm, text="🔧 Gemini Settings")
        gemini_frame.grid(column=0, row=5, columnspan=2, pady=5, sticky="ew")
        ttk.Button(gemini_frame, text="Edit Prompt").grid(column=0, row=0, sticky="w", padx=5, pady=2)
        ttk.Button(gemini_frame, text="Set API Token").grid(column=1, row=0, sticky="w", padx=5, pady=2)

        # Run and Log
        self.run_button = ttk.Button(frm, text="🚀 Run Pipeline", command=self.run_mock_pipeline)
        self.run_button.grid(column=0, row=6, pady=10, sticky="w")

        self.log_box = tk.Text(frm, height=15, width=80, wrap="word")
        self.log_box.grid(column=0, row=7, columnspan=2, pady=5)
        self.insert_sample_logs()

    def run_mock_pipeline(self):
        messagebox.showinfo("Run Pipeline", "Pipeline execution placeholder. Replace with backend logic.")

    def insert_sample_logs(self):
        sample_logs = [
            "📅 Start date: 2025-06-01",
            "📅 End date: 2025-07-01",
            "📥 PubMed selected: Fetching records...",
            "📥 CrossRef selected: Fetching records...",
            "📥 arXiv selected: Fetching records...",
            "🔍 Running inclusion model: SVM",
            "🧪 Running disease model: Gemini",
            "💾 Exporting results to Excel...",
            "✅ Pipeline finished successfully. Results saved to C:/Users/user1/Downloads/test/ntd_results.xlsx"
        ]
        for line in sample_logs:
            self.log_box.insert("end", line + "\n")
        self.log_box.see("end")

if __name__ == "__main__":
    root = tk.Tk()
    app = NTDToolApp2(root)
    root.mainloop()
