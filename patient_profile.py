"""
patient_profile.py
MediSathi — Patient & Shop Profile with Visit History
MediSathi — রোগী ও দোকান প্রোফাইল, ভিজিট-ইতিহাস সহ

Why phone number is not the ID / কেন ফোন নম্বর সরাসরি ID না:
    EN: A raw phone number as the primary key means any future export —
        a research dataset, a regional report — carries the patient's
        real identity along with it. Instead, patient_id is a hash of
        name+phone+salt: the same person always maps to the same ID
        (so history still links up), but the ID itself reveals nothing.
        Name and phone are kept in a separate local table for the shop
        worker's own use, since they already know the patient in person.
    BN: ফোন নম্বর সরাসরি primary key হলে ভবিষ্যতে কোনো export — গবেষণার
        ডেটাসেট, আঞ্চলিক রিপোর্ট — তার সাথে রোগীর আসল পরিচয়ও চলে যায়।
        এর বদলে patient_id হলো name+phone+salt-এর একটা hash: একই মানুষ
        সবসময় একই ID পাবেন (তাই ইতিহাস মিলবে), কিন্তু ID নিজে কিছু
        প্রকাশ করে না। নাম ও ফোন আলাদা local টেবিলে থাকে দোকান কর্মীর
        নিজের ব্যবহারের জন্য, কারণ তিনি তো রোগীকে সামনাসামনি চেনেনই।

Why this matters for decisions / সিদ্ধান্তে এটা কেন কাজে লাগে:
    EN: When a returning patient is looked up, their past confirmed
        conditions are pulled automatically and fed into
        knowledge_base.suggest() as comorbidities — the worker does not
        have to re-ask or re-type what was already recorded last visit.
    BN: পুরনো রোগী এলে তার আগের নিশ্চিত রোগগুলো স্বয়ংক্রিয়ভাবে বের হয়ে
        knowledge_base.suggest()-এ comorbidity হিসেবে যোগ হয় — কর্মীকে
        আগের ভিজিটে যা রেকর্ড হয়েছে সেটা আবার জিজ্ঞেস বা টাইপ করতে হয় না।
"""

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

DB = Path("medisathi.db")
SALT = "medisathi-v1"   # ⚠️ বাস্তব ব্যবহারে এটা .env-এ রাখুন, কোডে লেখা রাখবেন না


def _hash_id(name: str, phone: str) -> str:
    """নাম+ফোন+salt থেকে স্থায়ী hash — একই মানুষ সবসময় একই ID পাবেন"""
    raw = f"{name.strip().lower()}|{phone.strip()}|{SALT}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _connect():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db():
    con = _connect()
    con.executescript("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id   TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        phone        TEXT NOT NULL,
        age          INTEGER,
        sex          TEXT,
        consent      TEXT DEFAULT 'YES',
        created_at   TEXT
    );
    CREATE TABLE IF NOT EXISTS visits (
        visit_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id   TEXT NOT NULL REFERENCES patients(patient_id),
        shop_id      TEXT REFERENCES shops(shop_id),  -- কোন দোকানের ভিজিট, ৫-দোকানের পরীক্ষায় আলাদা করার জন্য
        visit_date   TEXT,
        symptom_bn   TEXT,
        pain_score   INTEGER,
        tier         TEXT,
        icd10        TEXT,
        confirmed    TEXT DEFAULT 'NO',
        active       TEXT DEFAULT 'YES',   -- গর্ভাবস্থার মতো সাময়িক অবস্থা বন্ধ করতে NO করা হয়
        action       TEXT,
        medicine     TEXT
    );
    CREATE TABLE IF NOT EXISTS allergies (
        allergy_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id   TEXT NOT NULL REFERENCES patients(patient_id),
        allergen     TEXT NOT NULL,        -- molecule নাম বা drug class
        noted_date   TEXT,
        confirmed    TEXT DEFAULT 'NO'
    );
    CREATE TABLE IF NOT EXISTS shops (
        shop_id      TEXT PRIMARY KEY,
        worker_name  TEXT,
        location_bn  TEXT,
        created_at   TEXT
    );
    """)
    con.commit()
    con.close()


# ============================================================
# রোগী / Patient
# ============================================================

def register_patient(name: str, phone: str, age: int, sex: str,
                     consent: bool = True) -> str:
    """
    EN: Creates the patient if new, or returns the existing ID if this
        name+phone combination has been seen before — so history links up
        automatically without the worker needing to search first.
    BN: নতুন হলে রোগী তৈরি করে, আগে থেকে থাকলে existing ID ফেরত দেয় —
        তাই কর্মীকে আগে খুঁজতে হয় না, ইতিহাস নিজে থেকেই মিলে যায়।
    """
    pid = _hash_id(name, phone)
    con = _connect()
    existing = con.execute("SELECT patient_id FROM patients WHERE patient_id=?",
                           (pid,)).fetchone()
    if not existing:
        con.execute(
            "INSERT INTO patients (patient_id,name,phone,age,sex,consent,created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (pid, name.strip(), phone.strip(), age, sex,
             "YES" if consent else "NO", datetime.now().isoformat(timespec="seconds")))
        con.commit()
    con.close()
    return pid


def find_patient(name: str, phone: str) -> Optional[str]:
    """একই hash আসবে কিনা যাচাই — লুকআপের জন্য, নতুন তৈরি করে না"""
    pid = _hash_id(name, phone)
    con = _connect()
    row = con.execute("SELECT patient_id FROM patients WHERE patient_id=?", (pid,)).fetchone()
    con.close()
    return pid if row else None


def add_visit(patient_id: str, symptom_bn: str, pain_score: int, tier: str,
             icd10: str = "", action: str = "", medicine: str = "",
             confirmed: bool = False, shop_id: str = "") -> int:
    con = _connect()
    cur = con.execute(
        "INSERT INTO visits (patient_id,shop_id,visit_date,symptom_bn,pain_score,tier,"
        "icd10,confirmed,action,medicine) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (patient_id, shop_id or None, datetime.now().isoformat(timespec="seconds"), symptom_bn,
         pain_score, tier, icd10, "YES" if confirmed else "NO", action, medicine))
    con.commit()
    vid = cur.lastrowid
    con.close()
    return vid


def get_history(patient_id: str) -> List[dict]:
    con = _connect()
    rows = con.execute(
        "SELECT visit_date,symptom_bn,pain_score,tier,icd10,confirmed,action,medicine "
        "FROM visits WHERE patient_id=? ORDER BY visit_date DESC", (patient_id,)
    ).fetchall()
    con.close()
    cols = ["visit_date", "symptom_bn", "pain_score", "tier", "icd10", "confirmed",
            "action", "medicine"]
    return [dict(zip(cols, r)) for r in rows]


def confirm_visit(visit_id: int, icd10: Optional[str] = None):
    """
    EN: A pharmacist or physician calls this after reviewing a system
        guess, to promote it from a retrieval suggestion to a confirmed
        diagnosis. Only after this does the condition start counting as
        a comorbidity on future visits.
    BN: সিস্টেমের অনুমান দেখার পর একজন ফার্মাসিস্ট বা ডাক্তার এটা কল
        করেন, যাতে সেটা retrieval-suggestion থেকে নিশ্চিত রোগে উন্নীত
        হয়। এটা করার পরই সেই রোগ ভবিষ্যৎ ভিজিটে comorbidity হিসেবে
        গোনা শুরু হবে।
    """
    con = _connect()
    if icd10:
        con.execute("UPDATE visits SET confirmed='YES', icd10=? WHERE visit_id=?",
                   (icd10, visit_id))
    else:
        con.execute("UPDATE visits SET confirmed='YES' WHERE visit_id=?", (visit_id,))
    con.commit()
    con.close()


def get_known_comorbidities(patient_id: str) -> List[str]:
    """
    EN: Distinct ICD-10 codes from past *confirmed and still-active*
        visits only. Pregnancy (Z34.9) is time-limited — once resolved
        via resolve_condition(), it stops being returned, so drugs are
        not blocked for a patient months after delivery. A chronic
        diagnosis like hypertension is never resolved, so it keeps
        showing up on every future visit.
    BN: শুধু আগের *নিশ্চিত এবং এখনো-সক্রিয়* ভিজিট থেকে ICD-10 কোড।
        গর্ভাবস্থা (Z34.9) সাময়িক — resolve_condition() দিয়ে বন্ধ করার
        পর সেটা আর ফেরত আসবে না, তাই প্রসবের কয়েক মাস পরও ওষুধ আটকে
        থাকবে না। উচ্চ রক্তচাপের মতো দীর্ঘস্থায়ী রোগ কখনো resolve হয়
        না, তাই ভবিষ্যতের প্রতিটা ভিজিটে দেখা যেতেই থাকবে।
    """
    con = _connect()
    rows = con.execute(
        "SELECT DISTINCT icd10 FROM visits WHERE patient_id=? AND confirmed='YES' "
        "AND active='YES' AND icd10 != ''", (patient_id,)
    ).fetchall()
    con.close()
    return [r[0] for r in rows]


def resolve_condition(patient_id: str, icd10: str):
    """
    EN: Marks a temporary condition as no longer active — call this after
        delivery, for example, so pregnancy stops being treated as a
        current comorbidity.
    BN: একটা সাময়িক অবস্থাকে আর-সক্রিয়-নয় বলে চিহ্নিত করে — যেমন প্রসবের
        পরে এটা কল করলে গর্ভাবস্থা আর বর্তমান comorbidity হিসেবে গণ্য
        হবে না।
    """
    con = _connect()
    con.execute("UPDATE visits SET active='NO' WHERE patient_id=? AND icd10=?",
               (patient_id, icd10))
    con.commit()
    con.close()


# ============================================================
# এলার্জি / Allergies
# ============================================================

def add_allergy(patient_id: str, allergen: str, confirmed: bool = True):
    """
    EN: Recorded separately from comorbidities because an allergy blocks
        a specific molecule or class directly — it is not routed through
        disease mechanism the way a comorbidity contraindication is.
    BN: comorbidity থেকে আলাদাভাবে রাখা হয়, কারণ এলার্জি সরাসরি একটা
        নির্দিষ্ট molecule বা শ্রেণি আটকায় — comorbidity-র মতো রোগের
        কারণ দিয়ে ঘুরে আসে না।
    """
    con = _connect()
    con.execute(
        "INSERT INTO allergies (patient_id,allergen,noted_date,confirmed) VALUES (?,?,?,?)",
        (patient_id, allergen.strip(), datetime.now().isoformat(timespec="seconds"),
         "YES" if confirmed else "NO"))
    con.commit()
    con.close()


def get_known_allergies(patient_id: str) -> List[str]:
    con = _connect()
    rows = con.execute(
        "SELECT DISTINCT allergen FROM allergies WHERE patient_id=? AND confirmed='YES'",
        (patient_id,)
    ).fetchall()
    con.close()
    return [r[0] for r in rows]


# ============================================================
# দোকান / Shop
# ============================================================

def register_shop(shop_id: str, worker_name: str, location_bn: str = "") -> str:
    con = _connect()
    con.execute(
        "INSERT OR IGNORE INTO shops (shop_id,worker_name,location_bn,created_at) "
        "VALUES (?,?,?,?)",
        (shop_id, worker_name, location_bn, datetime.now().isoformat(timespec="seconds")))
    con.commit()
    con.close()
    return shop_id


def list_shops() -> List[dict]:
    """সব নিবন্ধিত দোকান — app-এর dropdown-এ দেখানোর জন্য"""
    con = _connect()
    rows = con.execute("SELECT shop_id, worker_name, location_bn FROM shops "
                       "ORDER BY worker_name").fetchall()
    con.close()
    return [{"shop_id": r[0], "worker_name": r[1], "location_bn": r[2]} for r in rows]


# ============================================================
if __name__ == "__main__":
    init_db()
    print("### ডেমো / Demo ###\n")

    shop = register_shop("shop_001", "করিম মিয়া", "খাগড়াছড়ি সদর")
    print(f"দোকান নিবন্ধিত / shop registered: {shop}")

    pid = register_patient("রহিমা বেগম", "01712345678", age=52, sex="female")
    add_visit(pid, "মাথা ঘোরে প্রেসার বেশি", pain_score=6, tier="MODERATE",
             icd10="I10", action="ESCALATE", medicine="Amlodipine", confirmed=True)
    print(f"\nরোগী নিবন্ধিত / patient registered: {pid}")
    print("প্রথম ভিজিট যোগ হলো (উচ্চ রক্তচাপ, নিশ্চিত)")

    pid2 = register_patient("রহিমা বেগম", "01712345678", age=52, sex="female")
    print(f"\nদ্বিতীয়বার একই নাম+ফোন দিয়ে খোঁজা → {'✅ একই ID' if pid2==pid else '❌ ভিন্ন ID'}")

    known = get_known_comorbidities(pid2)
    print(f"এই রোগীর আগে-নিশ্চিত comorbidity: {known}")
    print("→ knowledge_base.suggest('K30', comorbidities=known) কল করলে")
    print("  Sodium bicarbonate নিজে থেকেই বাদ পড়বে, কারণ I10 (উচ্চ রক্তচাপ)")
    print("  আগের ভিজিট থেকেই জানা আছে — নতুন করে জিজ্ঞেস করা লাগছে না।")

    print(f"\n### রহিমা বেগম-এর সম্পূর্ণ ইতিহাস ###")
    for v in get_history(pid):
        print(f"  {v['visit_date']}  {v['symptom_bn']:<28} pain={v['pain_score']} "
              f"tier={v['tier']:<9} icd10={v['icd10']} confirmed={v['confirmed']}")

    # ---------- গর্ভাবস্থা — সাময়িক, resolve করা যায় ----------
    print(f"\n### গর্ভাবস্থা — সাময়িক অবস্থার ডেমো ###")
    pid3 = register_patient("সাথী আক্তার", "01898765432", 26, "female")
    add_visit(pid3, "মাসিক বন্ধ বমি ভাব", 3, "LOW", icd10="Z34.9",
             action="AUTO_SUGGEST", confirmed=True)
    print(f"গর্ভাবস্থা নিশ্চিত: {get_known_comorbidities(pid3)}")

    resolve_condition(pid3, "Z34.9")   # প্রসবের পর
    print(f"প্রসবের পর resolve করার পর: {get_known_comorbidities(pid3)}")
    print("→ এখন আর NSAID/অন্য গর্ভাবস্থা-নিষিদ্ধ ওষুধ ভুলভাবে আটকাবে না")

    # ---------- এলার্জি ----------
    print(f"\n### এলার্জি — স্থায়ী, molecule-স্তরে ডেমো ###")
    add_allergy(pid, "Penicillin")
    print(f"রহিমা বেগমের এলার্জি: {get_known_allergies(pid)}")
    print("→ knowledge_base.suggest(..., allergies=['Penicillin']) কল করলে")
    print("  Amoxicillin-এর মতো penicillin-শ্রেণির সব ওষুধ বাদ পড়বে")
