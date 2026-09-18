"""
knowledge_base.py
MediSathi — Mechanism-Mediated Medicine Knowledge Base
MediSathi — কারণ-ভিত্তিক ওষুধের জ্ঞানভাণ্ডার

Why this is a table and not a model / কেন এটা টেবিল, মডেল নয়:
    EN: Which molecule treats which condition is settled pharmacology, not
        something to infer statistically. A curated table can be read,
        checked and corrected by a physician; a model's output cannot.
        Retrieval only has to answer "which condition" — everything after
        that comes from here, and is therefore auditable.
    BN: কোন molecule কোন রোগে কাজ করে, সেটা প্রতিষ্ঠিত ফার্মাকোলজি —
        পরিসংখ্যান দিয়ে অনুমান করার বিষয় নয়। একটা curated টেবিল একজন
        চিকিৎসক পড়ে, যাচাই করে, সংশোধন করতে পারেন; মডেলের আউটপুট পারেন
        না। Retrieval-কে শুধু "কোন রোগ" বলতে হয় — তারপরের সবটুকু এখান
        থেকে আসে, তাই যাচাইযোগ্য থাকে।

What it enables / এটা যা সম্ভব করে:
    EN: Because molecules are grouped by class, a stock-out stops being a
        dead end. If Cetirizine is unavailable, the table still holds
        Loratadine and Fexofenadine under the same H1-blocker class.
    BN: Molecule গুলো শ্রেণি অনুযায়ী সাজানো বলে, দোকানে ওষুধ না থাকা আর
        অচলাবস্থা নয়। Cetirizine না থাকলে একই H1-blocker শ্রেণিতে
        Loratadine আর Fexofenadine টেবিলেই আছে।

⚠️ প্রতিটা সারিতে review_status = draft. তিনটা স্তর আছে:
     draft              — লেখা হয়েছে, কেউ যাচাই করেনি
     self-checked       — WHO/BNF-এর মতো উৎসে মিলিয়ে দেখা হয়েছে
     clinician-approved — চিকিৎসক অনুমোদন দিয়েছেন
   রোগীর কাছে ব্যবহারের আগে clinician-approved হতে হবে।
   Rows start as draft. Clinical use requires clinician-approved.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

DISEASES = Path("disease_knowledge.csv")
MOLECULES = Path("molecules.csv")


def _split(cell) -> List[str]:
    """সেমিকোলন-আলাদা ঘর থেকে তালিকা / list from a semicolon-separated cell"""
    if pd.isna(cell) or not str(cell).strip():
        return []
    return [x.strip() for x in str(cell).split(";") if x.strip()]


class KnowledgeBase:
    def __init__(self):
        if not DISEASES.exists() or not MOLECULES.exists():
            raise FileNotFoundError("disease_knowledge.csv বা molecules.csv পাওয়া যায়নি")
        self.dis = pd.read_csv(DISEASES)
        self.mol = pd.read_csv(MOLECULES)
        # সমৃদ্ধ ব্যাখ্যা (primary symptoms, distinguishing note, কেন এই drug
        # class) থাকলে সেটাও যোগ করা হয় — না থাকলে চুপচাপ বাদ, কিছু ভাঙে না
        ref_path = Path("disease_master_reference.csv")
        self.ref = pd.read_csv(ref_path).set_index("icd10") if ref_path.exists() else None

    # ---------- lookup ----------

    def disease(self, icd10: str) -> Optional[dict]:
        row = self.dis[self.dis.icd10.astype(str).str.upper() == icd10.upper()]
        return row.iloc[0].to_dict() if len(row) else None

    def molecules_for(self, icd10: str) -> pd.DataFrame:
        mask = self.mol.for_icd10.apply(lambda c: icd10.upper() in
                                        [x.upper() for x in _split(c)])
        return self.mol[mask]

    def conflicts(self, molecule_row, comorbidities: List[str]) -> List[str]:
        """
        EN: Returns the patient's ICD codes that clash with this molecule.
        BN: এই molecule-এর সাথে সংঘাতে থাকা রোগীর ICD কোডগুলো ফেরত দেয়।
        """
        bad = [x.upper() for x in _split(molecule_row.contraindicated_icd10)]
        return [c for c in comorbidities if c.upper() in bad]

    def allergy_conflict(self, molecule_row, allergies: List[str]) -> Optional[str]:
        """
        EN: An allergy is a direct molecule/class match, not a disease
            mechanism — so it is checked separately from comorbidities.
            Matches on either the exact molecule name or its drug class,
            since a patient may know they react to "penicillin" (the
            class) without knowing the exact brand they were given.
        BN: এলার্জি সরাসরি molecule/class-এর মিল, রোগের কারণ নয় — তাই
            comorbidity থেকে আলাদাভাবে চেক করা হয়। molecule-এর নাম বা
            তার drug class — দুটোর যেকোনো একটায় মিললেই ধরা পড়ে, কারণ
            রোগী হয়তো জানেন "পেনিসিলিনে" (শ্রেণি) সমস্যা হয়, ঠিক কোন
            brand দেওয়া হয়েছিল সেটা না জেনেই।
        """
        low = [a.strip().lower() for a in allergies]
        if molecule_row.molecule.strip().lower() in low:
            return molecule_row.molecule
        for a in low:
            if a and a in molecule_row.drug_class.strip().lower():
                return molecule_row.drug_class
        return None

    # ---------- the main entry point ----------

    def suggest(self, icd10: str, comorbidities: Optional[List[str]] = None,
               allergies: Optional[List[str]] = None) -> dict:
        comorbidities = comorbidities or []
        allergies = allergies or []
        d = self.disease(icd10)
        if d is None:
            return {"error": f"ICD {icd10} টেবিলে নেই / not in the table"}

        safe, blocked = [], []
        for _, m in self.molecules_for(icd10).iterrows():
            entry = {"molecule": m.molecule, "class": m.drug_class,
                     "route": m.route, "tier": m.tier,
                     "note": "" if pd.isna(m.contraindication_note_bn) else m.contraindication_note_bn}

            allergy_hit = self.allergy_conflict(m, allergies)
            clash = self.conflicts(m, comorbidities)

            if allergy_hit:
                entry["blocked_by"] = [f"allergy:{allergy_hit}"]
                blocked.append(entry)
            elif clash:
                entry["blocked_by"] = clash
                blocked.append(entry)
            else:
                safe.append(entry)

        # EN: if every option is contraindicated, an empty list is not an
        #     answer — the case has to go to a clinician. Silence would
        #     leave the worker to improvise, which is the outcome this
        #     system exists to prevent.
        # BN: সব option যদি নিষিদ্ধ হয়, খালি তালিকা কোনো উত্তর নয় —
        #     কেসটা ক্লিনিশিয়ানের কাছে যেতে হবে। চুপ থাকলে কর্মী নিজে
        #     আন্দাজে কিছু করবেন, যেটা ঠেকানোই এই সিস্টেমের উদ্দেশ্য।
        action = "SUGGEST"
        if d["tier"] == "REFER_ONLY":
            action = "REFER_ONLY"
        elif not safe:
            action = "ESCALATE_ALL_BLOCKED"

        # explainability — দোকানদার ও ডাক্তারের জন্য "কেন" এই তথ্য
        # explainability — the "why" for both pharmacy worker and physician
        explain = {}
        if self.ref is not None and icd10 in self.ref.index:
            r = self.ref.loc[icd10]
            explain = {
                "primary_symptoms_bn": r.get("primary_symptoms_bn", ""),
                "distinguishing_note_bn": r.get("distinguishing_note_bn", ""),
                "why_this_drug_class_bn": r.get("why_this_drug_class_bn", ""),
            }

        return {"icd10": icd10, "disease_bn": d["disease_bn"], "disease_en": d["disease_en"],
                "why_bn": d["pathophysiology_bn"], "drug_class": d["drug_class"],
                "disease_tier": d["tier"], "note_bn": d["notes_bn"],
                "explain": explain,
                "action": action, "safe": safe, "blocked": blocked}

    def alternatives(self, molecule: str) -> List[str]:
        """
        EN: Same-class substitutes, for when the named molecule is out of stock.
        BN: একই শ্রেণির বিকল্প — নির্দিষ্ট ওষুধ দোকানে না থাকলে।
        """
        row = self.mol[self.mol.molecule.str.lower() == molecule.strip().lower()]
        if not len(row):
            return []
        cls = row.iloc[0].drug_class
        same = self.mol[(self.mol.drug_class == cls) &
                        (self.mol.molecule.str.lower() != molecule.strip().lower())]
        return same.molecule.tolist()


# ============================================================
def _show(res):
    if "error" in res:
        print("  " + res["error"]); return
    print(f"  রোগ / condition : {res['disease_bn']}  [{res['icd10']}]")
    if res.get("explain", {}).get("distinguishing_note_bn"):
        print(f"  আলাদা করার চিহ্ন : {res['explain']['distinguishing_note_bn']}")
    print(f"  কেন হয় / why    : {res['why_bn']}")
    print(f"  যে শ্রেণি লাগে   : {res['drug_class']}")
    if res["disease_tier"] == "REFER_ONLY":
        print(f"  🚨 {res['note_bn']}")
        print("  → ফার্মেসিতে কোনো ওষুধ দেখানো হবে না।")
        return
    if res["safe"]:
        print("  ✅ ব্যবহারযোগ্য / usable:")
        for m in res["safe"]:
            print(f"       {m['molecule']:<24} {m['class']:<28} [{m['tier']}]")
    if res["blocked"]:
        print("  ❌ বাদ / blocked:")
        for m in res["blocked"]:
            print(f"       {m['molecule']:<24} ← {', '.join(m['blocked_by'])}: {m['note']}")
    if res["action"] == "ESCALATE_ALL_BLOCKED":
        print("  🚨 সব বিকল্পই এই রোগীর জন্য নিষিদ্ধ →")
        print("     ডাক্তার/ফার্মাসিস্টের কাছে পাঠান, নিজে কিছু দেবেন না।")
        print("     Every option is contraindicated — refer, do not improvise.")


if __name__ == "__main__":
    kb = KnowledgeBase()
    print(f"### {len(kb.dis)}টি রোগ, {len(kb.mol)}টি molecule লোড হয়েছে ###\n")

    print("── কেস ১: সাধারণ সর্দি, কোনো comorbidity নেই ──")
    _show(kb.suggest("J00"))

    print("\n── কেস ২: গ্যাস্ট্রিক, কিন্তু রোগীর উচ্চ রক্তচাপ আছে ──")
    _show(kb.suggest("K30", comorbidities=["I10"]))

    print("\n── কেস ৩: বাতের ব্যথা, রোগী গর্ভবতী ──")
    _show(kb.suggest("M13.9", comorbidities=["Z34.9"]))

    print("\n── কেস ৫: নিউমোনিয়া, রোগীর Penicillin-এ এলার্জি ──")
    _show(kb.suggest("J18.9", allergies=["Penicillin"]))

    print("\n── কেস ৪: বুকে ব্যথা ──")
    _show(kb.suggest("R07.9"))

    print("\n── স্টক-আউট: Cetirizine নেই, বিকল্প কী? ──")
    print("  →", ", ".join(kb.alternatives("Cetirizine")) or "কোনো বিকল্প নেই")
