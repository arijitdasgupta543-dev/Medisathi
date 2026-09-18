"""
medisathi_app.py
MediSathi — Shop-Usable Desktop App (v2)
MediSathi — দোকানে ব্যবহারের ডেস্কটপ অ্যাপ (সংস্করণ ২)

এই সংস্করণে যা যা যোগ/বদল হয়েছে / changes in this version:
    ১. Comorbidity multi-select — ৩০টি সাধারণ রোগের checkbox তালিকা
    ২. ফলাফল সম্পূর্ণ দ্বিভাষিক — প্রতিটি লাইন বাংলা ও ইংরেজি
    ৩. ওষুধের উপাদান ও কার্যপ্রণালীর বিস্তারিত ব্যাখ্যা
    ৪. এলার্জি input field বাদ (ভিতরের safety logic অক্ষত)
    ৫. Body map বাটন — 2D শরীরের মানচিত্র browser-এ খোলে
    ৬. Background watermark (⛑ 🩺 🧬)

চালানোর নিয়ম / how to run:
    python medisathi_app.py
"""

import sys
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).parent))
from severity_tier import severity_from_scale, SeverityTier
from patient_profile import (
    init_db, register_patient, find_patient, get_known_comorbidities,
    get_known_allergies, add_visit, get_history, register_shop, list_shops,
)
from knowledge_base import KnowledgeBase

TRAIN_FILE = "training_corpus_en_v2.csv"

# ---------------------------------------------------------------
# অনিশ্চিত হলে যে প্রশ্ন করা হবে / disambiguation questions
# EN: Asked only when the classifier's neighbours disagree. Each
#     question targets the one feature that separates this condition
#     from its most-confused neighbour (from disease_master_reference).
#     "Yes" confirms this condition; "No" moves to the runner-up.
# BN: শুধু তখনই জিজ্ঞেস করা হয় যখন classifier-এর প্রতিবেশীরা একমত নয়।
#     প্রতিটা প্রশ্ন সেই একটা বৈশিষ্ট্যকে লক্ষ্য করে যা এই রোগকে তার
#     সবচেয়ে-গুলিয়ে-যাওয়া প্রতিবেশী থেকে আলাদা করে।
#     "হ্যাঁ" = এই রোগ নিশ্চিত; "না" = দ্বিতীয় সম্ভাবনায় যাবে।
# ---------------------------------------------------------------
DISAMBIGUATION = {
    "A09":   ("পায়খানা কি পানির মতো পাতলা?",
              "Are the stools watery/loose?"),
    "A09.1": ("রোগী কি ৫ বছরের কম বয়সী শিশু?",
              "Is the patient a child under 5?"),
    "B82.0": ("পায়খানায় কৃমি দেখা গেছে, বা রাতে পায়ুপথে চুলকায়?",
              "Worms seen in stool, or itching at the anus at night?"),
    "R50.9": ("জ্বর ছাড়া আর কোনো নির্দিষ্ট লক্ষণ কি নেই?",
              "Is fever the ONLY symptom (nothing else specific)?"),
    "A90":   ("চোখের পেছনে ব্যথা, গায়ে লাল দাগ, বা রক্তপাত আছে?",
              "Pain behind the eyes, red rash, or any bleeding?"),
    "J00":   ("হাঁচি ও নাক দিয়ে পানি পড়াই কি প্রধান সমস্যা?",
              "Are sneezing and runny nose the main problem?"),
    "J18.9": ("কাশির সাথে কফ ওঠে এবং শ্বাস নিতে কষ্ট হয়?",
              "Cough with phlegm AND difficulty breathing?"),
    "K30":   ("বুক জ্বালা কি খাওয়ার সাথে সম্পর্কিত?",
              "Is the burning related to eating?"),
    "N39.0": ("প্রস্রাবের সময় জ্বালাপোড়া হয়?",
              "Burning sensation while urinating?"),
    "I10":   ("রক্তচাপ মেপে বেশি পাওয়া গেছে?",
              "Was blood pressure measured and found high?"),
    "E11.9": ("অতিরিক্ত পিপাসা ও ওজন কমা একসাথে হচ্ছে?",
              "Excessive thirst AND weight loss together?"),
    "B35.9": ("চামড়ার দাগ কি গোলাকার, কিনারা উঁচু?",
              "Is the skin patch ring-shaped with a raised edge?"),
    "B86":   ("চুলকানি রাতে বাড়ে, ঘরের অন্যদেরও হচ্ছে?",
              "Itching worse at night, and others at home affected?"),
    "M13.9": ("জয়েন্ট ফুলে আছে বা সকালে শক্ত লাগে?",
              "Are the joints swollen or stiff in the morning?"),
    "H10.9": ("চোখ লাল এবং পিচুটি পড়ছে?",
              "Is the eye red with discharge?"),
    "R07.9": ("বুকের ব্যথা কি হাতে, ঘাড়ে বা চোয়ালে ছড়াচ্ছে?",
              "Does the chest pain spread to arm, neck, or jaw?"),
    "Z34.9": ("মাসিক বন্ধ আছে?",
              "Has the menstrual period stopped?"),
    "K08.8": ("ব্যথা কি নির্দিষ্ট একটা দাঁতে বা মাড়িতে?",
              "Is the pain in one specific tooth or gum?"),
}

# কত ভোট পেলে সরাসরি উত্তর, আর কত হলে প্রশ্ন করা হবে
# EN: 5+ of 7 neighbours agreeing → answer directly. 2–4 → ask the
#     question. Measured on 72 held-out cases: without this, 17/72 were
#     wrong; the question recovers most of those without sending the
#     patient away empty-handed.
# BN: ৭ জনের ৫+ একমত → সরাসরি উত্তর। ২-৪ → প্রশ্ন করবে। ৭২ কেসে মাপা:
#     এটা ছাড়া ১৭টা ভুল হচ্ছিল; প্রশ্নটা বেশিরভাগ উদ্ধার করে, রোগীকে
#     খালি হাতে ফেরত না পাঠিয়ে।
CONFIDENT_VOTES = 5
COMORBID_FILE = "comorbidity_list.csv"
MASTER_REF_FILE = "disease_master_reference.csv"
BODY_MAP_FILE = "body_map_2d.html"

# ---------------------------------------------------------------
# রঙ ও ফন্ট — হালকা থিম, বড় ফন্ট, কালো স্পষ্ট টেক্সট
# ---------------------------------------------------------------
BG = "#FDEFEF"; PANEL = "#FFFFFF"; ENTRY_BG = "#FFFFFF"
LINE = "#E8B4B4"; TEXT = "#111111"; DIM = "#5A4040"
ACCENT = "#B23A3A"; ACCENT_BG = "#F6D5D5"
GREEN = "#1E7A45"; AMBER = "#B4700A"; RED = "#C0392B"
DISABLED_BG = "#EFE6E6"

F_BASE = ("Segoe UI", 12)
F_LABEL = ("Segoe UI", 11, "bold")
F_HEADER = ("Segoe UI", 18, "bold")
F_BUTTON = ("Segoe UI", 12, "bold")
F_SMALL = ("Segoe UI", 9)


class MediSathiApp:
    def __init__(self, root):
        self.root = root
        root.title("MediSathi")
        root.geometry("1180x760")
        root.minsize(900, 560)
        root.configure(bg=BG)

        self.current_patient_id = None
        self.last_diagnosis = None
        self.shop_id = None
        self.shop_label = tk.StringVar(value="দোকান নির্বাচিত হয়নি")

        self._load_backend()
        self._select_shop()
        self._build_ui()

    # ============================================================
    def _load_backend(self):
        """মডেল, জ্ঞানভাণ্ডার ও তালিকা লোড — একবারই, চালু হওয়ার সময়"""
        init_db()
        self.kb = KnowledgeBase()

        train = pd.read_csv(TRAIN_FILE)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        X = self.vec.fit_transform(train.symptom_en.tolist())
        self.clf = KNeighborsClassifier(n_neighbors=7)
        self.clf.fit(X, train.icd10.tolist())
        self.icd_to_name = dict(zip(train.icd10, train.disease_bn))

        self.comorbid_df = pd.read_csv(COMORBID_FILE)
        self.master_ref = (pd.read_csv(MASTER_REF_FILE).set_index("icd10")
                          if Path(MASTER_REF_FILE).exists() else None)

    # ============================================================
    def _select_shop(self):
        existing = list_shops()
        win = tk.Toplevel()
        win.title("দোকান নির্বাচন / Select Shop")
        win.configure(bg=PANEL)
        win.geometry("440x380")
        win.grab_set()

        tk.Label(win, text="কোন দোকান থেকে চলছে?\nWhich shop is this?", bg=PANEL, fg=ACCENT,
                font=("Segoe UI", 14, "bold")).pack(pady=(18, 12))

        if existing:
            tk.Label(win, text="আগের দোকান বেছে নিন / Select existing:", bg=PANEL, fg=TEXT,
                    font=F_LABEL).pack(anchor="w", padx=22)
            for s in existing:
                tk.Button(win, text=f"{s['worker_name']} — {s['location_bn']} ({s['shop_id']})",
                         bg=ACCENT_BG, fg=TEXT, relief="solid", bd=1, anchor="w", padx=10,
                         font=F_BASE, command=lambda sid=s['shop_id']: pick(sid)
                         ).pack(fill="x", padx=22, pady=3)

        tk.Label(win, text="অথবা নতুন দোকান / Or register new:", bg=PANEL, fg=DIM,
                font=F_LABEL).pack(anchor="w", padx=22, pady=(16, 5))
        sid_var = tk.StringVar(); worker_var = tk.StringVar(); loc_var = tk.StringVar()
        for label, var in [("Shop ID (e.g. shop_01)", sid_var),
                           ("কর্মীর নাম / Worker name", worker_var),
                           ("এলাকা / Location", loc_var)]:
            tk.Label(win, text=label, bg=PANEL, fg=TEXT, font=("Segoe UI", 10)).pack(
                anchor="w", padx=22)
            tk.Entry(win, textvariable=var, bg=ENTRY_BG, fg=TEXT, relief="solid", bd=1,
                    font=F_BASE).pack(fill="x", padx=22, ipady=5, pady=(0, 5))

        def register_new():
            if not (sid_var.get().strip() and worker_var.get().strip()):
                messagebox.showinfo("তথ্য দরকার", "Shop ID ও কর্মীর নাম দিন।")
                return
            register_shop(sid_var.get().strip(), worker_var.get().strip(), loc_var.get().strip())
            pick(sid_var.get().strip())

        def pick(sid):
            self.shop_id = sid
            self.shop_label.set(f"দোকান / Shop: {sid}")
            win.destroy()

        tk.Button(win, text="নিবন্ধন করুন / Register", command=register_new,
                 bg=ACCENT, fg="#FFFFFF", relief="flat", font=F_BUTTON, pady=8
                 ).pack(fill="x", padx=22, pady=(10, 18))
        win.wait_window()

    # ============================================================
    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=ENTRY_BG, foreground=TEXT, font=F_BASE)

        # ---------- বাম কলাম: scrollable ----------
        left_outer = tk.Frame(self.root, bg=BG, width=430)
        left_outer.pack(side="left", fill="y")
        left_outer.pack_propagate(False)

        self.left_canvas = tk.Canvas(left_outer, bg=BG, highlightthickness=0)
        left_scroll = tk.Scrollbar(left_outer, orient="vertical", command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=left_scroll.set)
        left_scroll.pack(side="right", fill="y")
        self.left_canvas.pack(side="left", fill="both", expand=True)

        left = tk.Frame(self.left_canvas, bg=BG, padx=18, pady=14)
        lw = self.left_canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", lambda e: self.left_canvas.configure(
            scrollregion=self.left_canvas.bbox("all")))
        self.left_canvas.bind("<Configure>", lambda e: self.left_canvas.itemconfig(lw, width=e.width))
        self.left_canvas.bind_all("<MouseWheel>",
            lambda e: self.left_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        tk.Label(left, text="MediSathi", bg=BG, fg=ACCENT, font=F_HEADER).pack(anchor="w")
        tk.Label(left, textvariable=self.shop_label, bg=BG, fg=DIM, font=F_BASE).pack(
            anchor="w", pady=(2, 12))

        self.name_var = tk.StringVar(); self.phone_var = tk.StringVar()
        self.age_var = tk.StringVar(); self.sex_var = tk.StringVar(value="female")

        self._field(left, "নাম / Name", self.name_var)
        self._field(left, "ফোন / Phone", self.phone_var)
        self._field(left, "বয়স / Age", self.age_var)

        tk.Label(left, text="লিঙ্গ / Sex", bg=BG, fg=TEXT, font=F_LABEL, anchor="w").pack(
            fill="x", pady=(8, 3))
        sex_frame = tk.Frame(left, bg=BG); sex_frame.pack(fill="x")
        for val, lbl in [("female", "মহিলা / Female"), ("male", "পুরুষ / Male")]:
            tk.Radiobutton(sex_frame, text=lbl, variable=self.sex_var, value=val,
                          bg=BG, fg=TEXT, selectcolor=ENTRY_BG, activebackground=BG,
                          font=F_BASE).pack(side="left", padx=(0, 14))

        tk.Button(left, text="রোগী খুঁজুন / Lookup Patient", command=self.lookup_patient,
                  bg=PANEL, fg=ACCENT, relief="solid", bd=1, pady=6, font=F_LABEL
                  ).pack(fill="x", pady=(12, 6))

        self.history_box = tk.Text(left, height=4, bg=PANEL, fg=DIM, font=("Consolas", 10),
                                   relief="solid", bd=1, wrap="word", padx=8, pady=6)
        self.history_box.pack(fill="x", pady=(2, 12))
        self.history_box.insert("1.0", "রোগীর ইতিহাস এখানে দেখাবে...\nPatient history appears here...")
        self.history_box.config(state="disabled")

        # ---------- ১. রোগ নির্বাচন ----------
        tk.Label(left, text="১. রোগ নির্বাচন / Select Condition", bg=BG, fg=ACCENT,
                font=F_LABEL, anchor="w").pack(fill="x", pady=(6, 3))
        self.disease_choices = ["-- classifier নিজে বুঝুক / let classifier decide --"] + [
            f"{name}  [{icd}]" for icd, name in sorted(self.icd_to_name.items())]
        self.disease_var = tk.StringVar(value=self.disease_choices[0])
        combo = ttk.Combobox(left, textvariable=self.disease_var, values=self.disease_choices,
                            state="readonly", font=F_BASE, height=14)
        combo.pack(fill="x", ipady=4)
        combo.bind("<<ComboboxSelected>>", self._on_disease_selected)

        # ---------- ২. উপসর্গ ----------
        tk.Label(left, text="২. উপসর্গ (ইংরেজিতে) / Symptom in English", bg=BG, fg=ACCENT,
                font=F_LABEL, anchor="w").pack(fill="x", pady=(12, 3))
        self.symptom_hint = tk.Label(left, text="", bg=BG, fg=DIM, font=F_SMALL, justify="left")
        self.symptom_hint.pack(anchor="w")
        self.symptom_text = tk.Text(left, height=3, bg=ENTRY_BG, fg=TEXT, font=F_BASE,
                                    relief="solid", bd=1, wrap="word", insertbackground=TEXT,
                                    selectbackground=ACCENT_BG, selectforeground=TEXT,
                                    padx=8, pady=6)
        self.symptom_text.pack(fill="x")
        self._placeholder(self.symptom_text, "e.g. burning in the chest after eating")

        # ---------- ৩. ব্যথার মাত্রা ----------
        tk.Label(left, text="৩. ব্যথার মাত্রা / Pain Score  (৭+ হলে সরাসরি ডাক্তার)",
                bg=BG, fg=ACCENT, font=F_LABEL, anchor="w").pack(fill="x", pady=(14, 4))
        self.pain_var = tk.IntVar(value=-1)
        pain_frame = tk.Frame(left, bg=BG); pain_frame.pack(fill="x")
        self.pain_buttons = []
        for i in range(11):
            b = tk.Button(pain_frame, text=str(i), width=3, relief="solid", bd=1,
                         bg=ENTRY_BG, fg=TEXT, font=("Segoe UI", 13, "bold"),
                         command=lambda v=i: self._set_pain(v))
            b.grid(row=0, column=i, padx=2, pady=2, sticky="ew")
            pain_frame.grid_columnconfigure(i, weight=1)
            self.pain_buttons.append(b)
        pl = tk.Frame(left, bg=BG); pl.pack(fill="x")
        tk.Label(pl, text="0 = ব্যথা নেই / none", bg=BG, fg=DIM, font=F_SMALL).pack(side="left")
        tk.Label(pl, text="10 = সর্বোচ্চ / worst", bg=BG, fg=DIM, font=F_SMALL).pack(side="right")

        tk.Button(left, text="শরীরের মানচিত্র খুলুন / Open Body Map",
                 command=self.open_body_map, bg=PANEL, fg=ACCENT, relief="solid", bd=1,
                 pady=6, font=F_LABEL).pack(fill="x", pady=(8, 0))

        # ---------- ৪. Comorbidity ----------
        tk.Label(left, text="৪. অন্য কোন রোগ আছে? / Other Conditions",
                bg=BG, fg=ACCENT, font=F_LABEL, anchor="w").pack(fill="x", pady=(16, 2))
        tk.Label(left, text="একাধিক বাছা যাবে — যেগুলো প্রযোজ্য সব টিক দিন।\n"
                "Select all that apply.", bg=BG, fg=DIM, font=F_SMALL,
                justify="left").pack(anchor="w", pady=(0, 4))

        comorbid_outer = tk.Frame(left, bg=ENTRY_BG, relief="solid", bd=1, height=180)
        comorbid_outer.pack(fill="x")
        comorbid_outer.pack_propagate(False)
        cc = tk.Canvas(comorbid_outer, bg=ENTRY_BG, highlightthickness=0)
        cs = tk.Scrollbar(comorbid_outer, orient="vertical", command=cc.yview)
        cc.configure(yscrollcommand=cs.set)
        cs.pack(side="right", fill="y"); cc.pack(side="left", fill="both", expand=True)
        cframe = tk.Frame(cc, bg=ENTRY_BG)
        cwin = cc.create_window((0, 0), window=cframe, anchor="nw")
        cframe.bind("<Configure>", lambda e: cc.configure(scrollregion=cc.bbox("all")))
        cc.bind("<Configure>", lambda e: cc.itemconfig(cwin, width=e.width))

        self.comorbid_vars = {}
        for _, row in self.comorbid_df.iterrows():
            var = tk.BooleanVar(value=False)
            self.comorbid_vars[row["icd10"]] = var
            mark = "" if row["in_knowledge_base"] == "YES" else "  ⚠"
            tk.Checkbutton(cframe, text=f"{row['name_bn']} / {row['name_en']}{mark}",
                          variable=var, bg=ENTRY_BG, fg=TEXT, selectcolor="#FFFFFF",
                          activebackground=ENTRY_BG, font=("Segoe UI", 10), anchor="w"
                          ).pack(fill="x", padx=6, pady=1)

        tk.Label(left, text="⚠ চিহ্নিত রোগে স্বয়ংক্রিয় ওষুধ-বাদ কাজ করে না,\n"
                "শুধু ডাক্তারকে জানানো হবে।", bg=BG, fg=AMBER, font=F_SMALL,
                justify="left").pack(anchor="w", pady=(4, 0))

        tk.Button(left, text="যাচাই করুন / Check", command=self.run_check,
                  bg=ACCENT, fg="#FFFFFF", relief="flat", pady=10, font=F_BUTTON
                  ).pack(fill="x", pady=(18, 6))
        tk.Button(left, text="নতুন রোগী / New Patient", command=self.reset_form,
                  bg=PANEL, fg=DIM, relief="solid", bd=1, pady=7, font=F_LABEL).pack(fill="x")

        # ---------- ডান কলাম: ফলাফল ----------
        right = tk.Frame(self.root, bg=PANEL, padx=20, pady=16)
        right.pack(side="right", fill="both", expand=True)

        head = tk.Frame(right, bg=PANEL); head.pack(fill="x")
        tk.Label(head, text="ফলাফল / Result", bg=PANEL, fg=ACCENT, font=F_HEADER).pack(side="left")
        tk.Label(head, text="⛑   🩺   🧬", bg=PANEL, fg="#F0DADA",
                font=("Segoe UI", 22)).pack(side="right")

        self.result_box = tk.Text(right, bg="#FFFFFF", fg=TEXT, font=("Segoe UI", 11),
                                  relief="solid", bd=1, wrap="word", padx=14, pady=12)
        self.result_box.pack(fill="both", expand=True, pady=(10, 10))
        self.result_box.tag_configure("h", foreground=ACCENT, font=("Segoe UI", 14, "bold"))
        self.result_box.tag_configure("sub", foreground=ACCENT, font=("Segoe UI", 12, "bold"))
        self.result_box.tag_configure("ok", foreground=GREEN, font=("Segoe UI", 12, "bold"))
        self.result_box.tag_configure("warn", foreground=AMBER, font=("Segoe UI", 12, "bold"))
        self.result_box.tag_configure("bad", foreground=RED, font=("Segoe UI", 13, "bold"))
        self.result_box.tag_configure("dim", foreground=DIM, font=("Segoe UI", 10))
        self.result_box.tag_configure("mol", foreground=TEXT, font=("Segoe UI", 11, "bold"))
        self.result_box.config(state="disabled")
        self._show_watermark()

        btn_row = tk.Frame(right, bg=PANEL); btn_row.pack(fill="x")
        tk.Button(btn_row, text="রেফারেল Copy / Copy Referral", command=self.copy_referral,
                  bg=ACCENT_BG, fg=ACCENT, relief="solid", bd=1, pady=8, font=F_LABEL
                  ).pack(side="left")
        tk.Button(btn_row, text="ডাক্তার নিশ্চিত করেছেন / Confirmed", command=self.confirm_diagnosis,
                  bg="#DCF0E3", fg=GREEN, relief="solid", bd=1, pady=8, font=F_LABEL
                  ).pack(side="left", padx=(10, 0))

    # ============================================================
    def _show_watermark(self):
        self.result_box.config(state="normal")
        self.result_box.delete("1.0", "end")
        self.result_box.insert("end", "\n\n\n              ⛑         🩺         🧬\n\n", "dim")
        self.result_box.insert("end",
            "        রোগীর তথ্য দিয়ে 'যাচাই করুন' চাপুন\n"
            "        Enter patient details and press Check\n", "dim")
        self.result_box.config(state="disabled")

    def _field(self, parent, label, var):
        tk.Label(parent, text=label, bg=BG, fg=TEXT, font=F_LABEL, anchor="w").pack(
            fill="x", pady=(8, 3))
        e = tk.Entry(parent, textvariable=var, bg=ENTRY_BG, fg=TEXT, relief="solid", bd=1,
                    insertbackground=TEXT, font=F_BASE)
        e.pack(fill="x", ipady=6)

    def _placeholder(self, widget, text):
        widget.insert("1.0", text)
        widget.config(fg=DIM)
        def fin(e):
            if widget.get("1.0", "end").strip() == text:
                widget.delete("1.0", "end"); widget.config(fg=TEXT)
        def fout(e):
            if not widget.get("1.0", "end").strip():
                widget.insert("1.0", text); widget.config(fg=DIM)
        widget.bind("<FocusIn>", fin); widget.bind("<FocusOut>", fout)
        widget._placeholder_text = text

    def _get_symptom_text(self):
        val = self.symptom_text.get("1.0", "end").strip()
        return "" if val == getattr(self.symptom_text, "_placeholder_text", None) else val

    def _on_disease_selected(self, event=None):
        """
        EN: When a condition is picked manually the classifier is bypassed,
            so the symptom box is greyed and labelled record-only — this
            prevents the earlier confusion where typed symptom text
            appeared to be ignored.
        BN: হাতে রোগ বাছলে classifier ব্যবহার হয় না — তাই উপসর্গ বক্স ধূসর
            করে স্পষ্ট লেখা হয় যে এটা শুধু রেকর্ডের জন্য, যাতে আগের
            বিভ্রান্তি (টাইপ করা উপসর্গ উপেক্ষিত মনে হওয়া) আর না ঘটে।
        """
        if self._get_manual_icd():
            self.symptom_text.config(bg=DISABLED_BG)
            self.symptom_hint.config(
                text="⚠ রোগ হাতে বাছা হয়েছে — নিচের লেখা শুধু রেকর্ডে থাকবে।\n"
                     "Condition chosen manually — text below is recorded only.",
                fg=AMBER)
        else:
            self.symptom_text.config(bg=ENTRY_BG)
            self.symptom_hint.config(text="", fg=DIM)

    def _get_manual_icd(self):
        sel = self.disease_var.get()
        if sel == self.disease_choices[0]:
            return None
        icd = sel.split("[")[-1].rstrip("]").strip()
        return icd if icd in self.icd_to_name else None

    def _set_pain(self, v):
        self.pain_var.set(v)
        for i, b in enumerate(self.pain_buttons):
            on = (i == v)
            b.config(bg=self._pain_color(i) if on else ENTRY_BG,
                    fg="#FFFFFF" if on else TEXT)

    def _pain_color(self, n):
        return GREEN if n <= 4 else AMBER if n <= 6 else RED

    def _selected_comorbidities(self):
        return [icd for icd, var in self.comorbid_vars.items() if var.get()]

    def open_body_map(self):
        p = Path(BODY_MAP_FILE).resolve()
        if not p.exists():
            messagebox.showinfo("ফাইল নেই",
                f"{BODY_MAP_FILE} এই folder-এ পাওয়া যায়নি।\nFile not found in this folder.")
            return
        webbrowser.open(p.as_uri())

    # ============================================================
    def reset_form(self):
        self.name_var.set(""); self.phone_var.set(""); self.age_var.set("")
        self.sex_var.set("female")
        self.disease_var.set(self.disease_choices[0])
        self._on_disease_selected()
        self.symptom_text.delete("1.0", "end")
        self._placeholder(self.symptom_text, "e.g. burning in the chest after eating")
        self.pain_var.set(-1)
        for b in self.pain_buttons:
            b.config(bg=ENTRY_BG, fg=TEXT)
        for var in self.comorbid_vars.values():
            var.set(False)
        self.current_patient_id = None; self.last_diagnosis = None
        self.history_box.config(state="normal")
        self.history_box.delete("1.0", "end")
        self.history_box.insert("1.0", "রোগীর ইতিহাস এখানে দেখাবে...\nPatient history appears here...")
        self.history_box.config(state="disabled")
        self._show_watermark()

    def lookup_patient(self):
        name, phone = self.name_var.get().strip(), self.phone_var.get().strip()
        if not name or not phone:
            messagebox.showinfo("তথ্য দরকার", "নাম ও ফোন নম্বর দিন।\nEnter name and phone.")
            return
        pid = find_patient(name, phone)
        self.history_box.config(state="normal")
        self.history_box.delete("1.0", "end")
        if not pid:
            self.history_box.insert("1.0", "নতুন রোগী — কোনো আগের ইতিহাস নেই।\n"
                                            "New patient — no prior history.")
        else:
            self.current_patient_id = pid
            hist = get_history(pid); comor = get_known_comorbidities(pid)
            lines = [f"আগের ভিজিট / previous visits: {len(hist)}"]
            if comor:
                lines.append(f"নিশ্চিত comorbidity: {', '.join(comor)}")
                for icd in comor:
                    if icd in self.comorbid_vars:
                        self.comorbid_vars[icd].set(True)
            for h in hist[:3]:
                lines.append(f"  {h['visit_date'][:10]}  {h['symptom_bn'][:28]}")
            self.history_box.insert("1.0", "\n".join(lines))
        self.history_box.config(state="disabled")

    # ============================================================
    def _ask_disambiguation(self, top_icd, runner_icd, votes):
        """
        EN: Shown when the 7 neighbours disagree. Asks the one question
            that separates the top candidate from the runner-up, so the
            patient still leaves with a medicine — the right one —
            instead of being sent away because the model was unsure.
            Returns the confirmed ICD, or None if skipped.
        BN: ৭ জন প্রতিবেশী একমত না হলে দেখানো হয়। শীর্ষ প্রার্থীকে
            দ্বিতীয় প্রার্থী থেকে আলাদা করার প্রশ্নটা করে, যাতে রোগী
            ওষুধ নিয়েই বাড়ি যান — সঠিক ওষুধ — মডেল অনিশ্চিত বলে খালি
            হাতে ফেরত না গিয়ে। নিশ্চিত ICD ফেরত দেয়, এড়িয়ে গেলে None।
        """
        q_bn, q_en = DISAMBIGUATION.get(top_icd, (None, None))
        if not q_bn:
            return top_icd   # প্রশ্ন না থাকলে যা আছে তাই

        top_name = self.icd_to_name.get(top_icd, top_icd)
        runner_name = self.icd_to_name.get(runner_icd, runner_icd) if runner_icd else None

        win = tk.Toplevel(self.root)
        win.title("একটি প্রশ্ন / One Question")
        win.configure(bg=PANEL)
        win.geometry("560x330")
        win.transient(self.root)
        win.grab_set()

        tk.Label(win, text="🤔  নিশ্চিত হতে একটি প্রশ্ন",
                bg=PANEL, fg=ACCENT, font=("Segoe UI", 15, "bold")).pack(pady=(20, 2))
        tk.Label(win, text="One question to be sure",
                bg=PANEL, fg=DIM, font=F_SMALL).pack()

        info = f"সম্ভাবনা: {top_name}"
        if runner_name:
            info += f"  বা  {runner_name}"
        tk.Label(win, text=info, bg=PANEL, fg=DIM, font=F_SMALL,
                wraplength=500).pack(pady=(10, 14))

        qbox = tk.Frame(win, bg=ENTRY_BG, relief="solid", bd=1)
        qbox.pack(fill="x", padx=26, pady=(0, 18))
        tk.Label(qbox, text=q_bn, bg=ENTRY_BG, fg=TEXT, font=("Segoe UI", 13, "bold"),
                wraplength=460, justify="left").pack(padx=14, pady=(12, 3))
        tk.Label(qbox, text=q_en, bg=ENTRY_BG, fg=DIM, font=F_SMALL,
                wraplength=460, justify="left").pack(padx=14, pady=(0, 12))

        answer = {"icd": None}

        def yes():
            answer["icd"] = top_icd
            win.destroy()

        def no():
            # "না" মানে শীর্ষ প্রার্থী নয় — দ্বিতীয়টা ধরা হবে
            answer["icd"] = runner_icd
            win.destroy()

        def unsure():
            answer["icd"] = None
            win.destroy()

        btns = tk.Frame(win, bg=PANEL)
        btns.pack()
        tk.Button(btns, text="হ্যাঁ / Yes", command=yes, bg=GREEN, fg="#FFFFFF",
                 relief="flat", font=("Segoe UI", 13, "bold"), width=12, pady=8
                 ).pack(side="left", padx=8)
        tk.Button(btns, text="না / No", command=no, bg=PANEL, fg=TEXT,
                 relief="solid", bd=1, font=("Segoe UI", 13), width=12, pady=8
                 ).pack(side="left", padx=8)
        tk.Button(win, text="জানি না / Not sure", command=unsure, bg=PANEL, fg=DIM,
                 relief="flat", font=F_SMALL).pack(pady=(14, 0))

        win.wait_window()

        if answer["icd"]:
            self._out(f"\n❓ প্রশ্ন / Question: {q_bn}\n", "dim")
            self._out(f"   উত্তর অনুযায়ী নির্ধারিত / Resolved by answer\n", "dim")
        return answer["icd"]

    # ============================================================
    def run_check(self):
        name, phone = self.name_var.get().strip(), self.phone_var.get().strip()
        age_s = self.age_var.get().strip()
        symptom = self._get_symptom_text()
        pain = self.pain_var.get()
        manual_icd = self._get_manual_icd()

        if not (name and phone and age_s):
            messagebox.showinfo("তথ্য দরকার", "নাম, ফোন ও বয়স দিন।\nEnter name, phone, age.")
            return
        if not symptom and not manual_icd:
            messagebox.showinfo("তথ্য দরকার",
                "উপসর্গ লিখুন অথবা রোগ বেছে নিন।\nEnter a symptom or select a condition.")
            return
        if pain < 0:
            messagebox.showinfo("তথ্য দরকার", "ব্যথার মাত্রা বেছে নিন।\nSelect a pain score.")
            return
        try:
            age = int(age_s)
        except ValueError:
            messagebox.showinfo("ভুল তথ্য", "বয়স সংখ্যায় দিন।\nAge must be a number.")
            return

        pid = register_patient(name, phone, age, self.sex_var.get())
        self.current_patient_id = pid

        self._out_clear()
        self._out(f"রোগী / Patient: {name} ({age}, {self.sex_var.get()})\n", "h")
        self._out(f"উপসর্গ / Symptom: {symptom or '(হাতে রোগ বাছা হয়েছে)'}\n"
                 f"ব্যথা / Pain: {pain}/10\n\n")

        tier = severity_from_scale(pain)
        tier_bn = {"low": "কম", "moderate": "মাঝারি", "high": "উচ্চ"}[tier.value]
        self._out(f"তীব্রতা / Severity: {tier_bn} ({tier.value.upper()})\n",
                 "warn" if tier != SeverityTier.LOW else "ok")

        if tier == SeverityTier.HIGH:
            self._out("\n🚨 সরাসরি ডাক্তারের কাছে পাঠান\n   REFER TO A DOCTOR NOW\n", "bad")
            self._out("ব্যথা ৭ বা তার বেশি — কোনো ওষুধ খোঁজা হয়নি।\n"
                     "Pain is 7 or above — no medicine was retrieved.\n", "bad")
            add_visit(pid, symptom, pain, tier.value, action="REFER_NOW",
                      confirmed=False, shop_id=self.shop_id)
            self.last_diagnosis = (None, None, None, "REFER_NOW")
            return

        selected = self._selected_comorbidities()
        known = get_known_comorbidities(pid)
        all_comorbid = sorted(set(selected) | set(known))
        allergies = get_known_allergies(pid)

        if all_comorbid:
            names = [self._comorbid_name(c) for c in all_comorbid]
            self._out(f"\nঅন্যান্য রোগ / Comorbidities: {', '.join(names)}\n", "dim")
            unknown = [c for c in all_comorbid if not self._in_kb(c)]
            if unknown:
                self._out("⚠ এগুলোতে স্বয়ংক্রিয় ওষুধ-বাদ কাজ করে না — ডাক্তারকে জানান:\n"
                         "   Not auto-checked, inform the doctor: "
                         + ", ".join(self._comorbid_name(c) for c in unknown) + "\n", "warn")

        if manual_icd:
            icd = manual_icd
            self._out("(রোগ হাতে বাছা হয়েছে — classifier ব্যবহার হয়নি)\n"
                     "(Condition selected manually — classifier not used)\n", "dim")
        else:
            # ৭ জন প্রতিবেশীর ভোট গণনা / count agreement among 7 neighbours
            Xq = self.vec.transform([symptom])
            proba = self.clf.predict_proba(Xq)[0]
            classes = self.clf.classes_
            order = proba.argsort()[::-1]
            top_icd, top_share = classes[order[0]], proba[order[0]]
            runner_icd = classes[order[1]] if len(order) > 1 and proba[order[1]] > 0 else None
            votes = int(round(top_share * 7))

            if votes >= CONFIDENT_VOTES:
                icd = top_icd
            else:
                # অনিশ্চিত — দোকানদারকে একটা প্রশ্ন করে নিশ্চিত হওয়া
                icd = self._ask_disambiguation(top_icd, runner_icd, votes)
                if icd is None:      # প্রশ্ন এড়িয়ে গেলে
                    self._out("\n⚠ নিশ্চিত হওয়া যায়নি — ডাক্তারের পরামর্শ নিন।\n"
                             "Could not confirm — please consult a doctor.\n", "warn")
                    add_visit(pid, symptom, pain, tier.value, action="NEEDS_CLARIFICATION",
                              confirmed=False, shop_id=self.shop_id)
                    self.last_diagnosis = (None, None, None, "NEEDS_CLARIFICATION")
                    return
        disease_bn = self.icd_to_name.get(icd, icd)

        self._out(f"\nসম্ভাব্য রোগ / Likely condition: {disease_bn}  [{icd}]\n", "sub")

        result = self.kb.suggest(icd, comorbidities=all_comorbid, allergies=allergies)
        if "error" in result:
            self._out(f"⚠ {result['error']}\n", "warn")
            add_visit(pid, symptom, pain, tier.value, icd10=icd, action="NOT_IN_KB",
                      confirmed=False, shop_id=self.shop_id)
            self.last_diagnosis = (icd, disease_bn, None, "NOT_IN_KB")
            return

        ex = result.get("explain", {})
        if ex.get("distinguishing_note_bn"):
            self._out(f"কীভাবে আলাদা / How it differs:\n{ex['distinguishing_note_bn']}\n\n", "dim")

        medicine_str = ""
        if result["disease_tier"] == "REFER_ONLY":
            self._out(f"🚨 {result['note_bn']}\n", "bad")
            self._out("ফার্মেসিতে কোনো ওষুধ দেখানো হবে না।\n"
                     "No medicine will be shown at the pharmacy.\n", "bad")
            action = "REFER_ONLY"
        elif result["action"] == "ESCALATE_ALL_BLOCKED":
            self._out("🚨 সব বিকল্পই এই রোগীর জন্য নিষিদ্ধ — ডাক্তারের কাছে পাঠান।\n"
                     "All options contraindicated — refer to a doctor.\n", "bad")
            for m in result["blocked"]:
                self._out(f"   ❌ {m['molecule']} ← {m['blocked_by']}\n", "dim")
            action = "ESCALATE_ALL_BLOCKED"
        else:
            safe_otc = [m for m in result["safe"] if m["tier"] == "OTC"]
            shown = safe_otc if safe_otc else result["safe"]
            action = "AUTO_SUGGEST" if safe_otc else "ESCALATE"
            if safe_otc:
                self._out("✅ সুপারিশ / Suggested (OTC):\n", "ok")
            else:
                self._out("📋 ডাক্তারের অনুমোদন লাগবে / Prescription needed:\n", "warn")

            mech = ""
            if self.master_ref is not None and icd in self.master_ref.index:
                m = self.master_ref.loc[icd].get("mechanism_bn", "")
                mech = m if isinstance(m, str) else ""

            for m in shown:
                self._out(f"\n   • {m['molecule']}", "mol")
                self._out(f"   ({m['class']}) [{m['tier']}]\n", "dim")
                if m.get("note"):
                    self._out(f"      ⚠ {m['note']}\n", "dim")
            medicine_str = ", ".join(m["molecule"] for m in shown)

            if mech.strip():
                self._out(f"\nশরীরে যা ঘটছে / What happens in the body:\n{mech}\n", "dim")
            if ex.get("why_this_drug_class_bn"):
                self._out(f"\nকেন এই ওষুধ কাজ করে / Why this medicine works:\n"
                         f"{ex['why_this_drug_class_bn']}\n", "dim")

            if result["blocked"]:
                self._out("\n❌ বাদ দেওয়া হয়েছে / Excluded:\n", "warn")
                for m in result["blocked"]:
                    self._out(f"   {m['molecule']} ← {m['blocked_by']}\n", "dim")

        add_visit(pid, symptom, pain, tier.value, icd10=icd, action=action,
                  medicine=medicine_str, confirmed=False, shop_id=self.shop_id)
        self.last_diagnosis = (icd, disease_bn, medicine_str, action)

    # ============================================================
    def _comorbid_name(self, icd):
        row = self.comorbid_df[self.comorbid_df.icd10 == icd]
        return row.iloc[0]["name_bn"] if len(row) else icd

    def _in_kb(self, icd):
        row = self.comorbid_df[self.comorbid_df.icd10 == icd]
        return len(row) > 0 and row.iloc[0]["in_knowledge_base"] == "YES"

    def copy_referral(self):
        if not self.last_diagnosis:
            messagebox.showinfo("কিছু নেই", "আগে 'যাচাই করুন' চাপুন।\nRun Check first.")
            return
        icd, disease_bn, medicine, action = self.last_diagnosis
        comor = [self._comorbid_name(c) for c in self._selected_comorbidities()]
        text = (f"*MediSathi রেফারেল / Referral*\n"
               f"রোগী / Patient: {self.name_var.get()} "
               f"({self.age_var.get()}, {self.sex_var.get()})\n"
               f"উপসর্গ / Symptom: {self._get_symptom_text() or '-'}\n"
               f"ব্যথা / Pain: {self.pain_var.get()}/10\n"
               f"অন্যান্য রোগ / Comorbidity: {', '.join(comor) if comor else 'নেই / none'}\n"
               f"সম্ভাব্য রোগ (⚠ অনিশ্চিত / unconfirmed): {disease_bn or '-'} [{icd or '-'}]\n"
               f"সিদ্ধান্ত / Action: {action}\n"
               f"ওষুধ / Medicine: {medicine or '-'}\n\n"
               f"দয়া করে নিশ্চিত করুন। / Please confirm.")
        self.root.clipboard_clear(); self.root.clipboard_append(text)
        messagebox.showinfo("Copy হয়েছে",
            "রেফারেল টেক্সট clipboard-এ — WhatsApp-এ paste করুন।\n"
            "Referral copied — paste into WhatsApp.")

    def confirm_diagnosis(self):
        if not self.current_patient_id or not self.last_diagnosis or not self.last_diagnosis[0]:
            messagebox.showinfo("কিছু নেই", "আগে একটা রোগ নির্ণয় হতে হবে।\nRun a diagnosis first.")
            return
        from patient_profile import confirm_visit
        import sqlite3
        con = sqlite3.connect("medisathi.db")
        vid = con.execute("SELECT MAX(visit_id) FROM visits WHERE patient_id=?",
                          (self.current_patient_id,)).fetchone()[0]
        con.close()
        if vid:
            confirm_visit(vid, self.last_diagnosis[0])
            messagebox.showinfo("নিশ্চিত হলো",
                f"visit_id={vid} নিশ্চিত — পরের ভিজিটে comorbidity হিসেবে গণ্য হবে।\n"
                f"Confirmed; will count as comorbidity next visit.")

    def _out_clear(self):
        self.result_box.config(state="normal")
        self.result_box.delete("1.0", "end")

    def _out(self, text, tag=None):
        self.result_box.config(state="normal")
        self.result_box.insert("end", text, tag)
        self.result_box.see("end")
        self.result_box.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = MediSathiApp(root)
    root.mainloop()
