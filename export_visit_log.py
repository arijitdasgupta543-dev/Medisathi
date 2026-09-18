"""
export_visit_log.py
MediSathi — Export Real-World Test Sessions for the Paper
MediSathi — আসল-জীবনের test session paper-এর জন্য export

EN: Every visit run through medisathi_app.py is already logged in
    medisathi.db (SQLite). This script pulls it into a clean CSV —
    patient (hashed, not name/phone), symptom, pain, tier, diagnosis,
    action, medicine, confirmation status — ready to summarise in the
    paper's evaluation section without exposing patient identity.
BN: medisathi_app.py দিয়ে চালানো প্রতিটা ভিজিট এমনিতেই medisathi.db
    (SQLite)-তে লগ হয়ে যায়। এই script সেটাকে একটা পরিষ্কার CSV-তে
    বের করে — রোগী (hashed, নাম/ফোন না), উপসর্গ, ব্যথা, tier, রোগ
    নির্ণয়, সিদ্ধান্ত, ওষুধ, নিশ্চিতকরণ অবস্থা — paper-এর evaluation
    অংশে সংক্ষিপ্ত করার জন্য প্রস্তুত, রোগীর পরিচয় প্রকাশ না করে।
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB = Path("medisathi.db")
OUT = Path("real_world_test_log.csv")


def main():
    if not DB.exists():
        print(f"❌ {DB} পাওয়া যায়নি — এখনো কোনো রোগী দেখা হয়নি।")
        return

    con = sqlite3.connect(DB)
    visits = pd.read_sql_query(
        "SELECT v.visit_id, v.patient_id, v.shop_id, s.worker_name, s.location_bn, "
        "v.visit_date, v.symptom_bn AS symptom_text, "
        "v.pain_score, v.tier, v.icd10, v.confirmed, v.action, v.medicine, "
        "p.age, p.sex "
        "FROM visits v "
        "JOIN patients p ON v.patient_id = p.patient_id "
        "LEFT JOIN shops s ON v.shop_id = s.shop_id "
        "ORDER BY v.visit_date", con)
    con.close()

    if not len(visits):
        print("⚠️  কোনো visit রেকর্ড নেই।")
        return

    visits.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"### {len(visits)}টা real-world ভিজিট export হলো ###\n")
    print(f"স্বতন্ত্র রোগী / unique patients: {visits.patient_id.nunique()}")
    print(f"স্বতন্ত্র দোকান / unique shops: {visits.shop_id.nunique()}")
    print(f"\nপ্রতি দোকানে ভিজিট / visits per shop:")
    print(visits.groupby(["shop_id","worker_name"]).size().to_string())
    print(f"\nসিদ্ধান্তের বিন্যাস / action distribution:")
    print(visits.action.value_counts().to_string())
    print(f"\nSeverity tier বিন্যাস:")
    print(visits.tier.value_counts().to_string())
    print(f"\nডাক্তার-নিশ্চিত / clinician-confirmed: "
          f"{(visits.confirmed=='YES').sum()}/{len(visits)}")

    print(f"\n💾 {OUT}")
    print("   এই ফাইলটাই paper-এর Results/Evaluation section-এ সংক্ষিপ্ত করার")
    print("   জন্য ব্যবহার করবেন। রোগীর নাম/ফোন এতে নেই — শুধু hashed ID।")


if __name__ == "__main__":
    main()
