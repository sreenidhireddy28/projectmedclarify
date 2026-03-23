import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class MedicalAI:
    def __init__(self):
        # --- 1. Load AI Models ---
        print("Loading Bio_ClinicalBERT (The Librarian)...")
        self.bert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        print("Loading T5-small (The Author)...")
        self.t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

        # --- 2. Build Robust Knowledge Base (300 Common Medicines) ---
        print("Building Extensive Internal Database (300 Items)...")
        
        self.knowledge_base = []
        self.valid_drugs = []

        # Format: ("Medicine Name", "Description", "Side Effects")
        raw_data = [
            # --- 1. ANALGESICS (PAIN) & NSAIDS ---
            ("Paracetamol", "Common painkiller and fever reducer.", "Liver damage (high dose), nausea."),
            ("Ibuprofen", "NSAID for pain and inflammation.", "Stomach ulcers, kidney strain."),
            ("Aspirin", "Blood thinner and pain reliever.", "Bleeding risk, stomach irritation."),
            ("Diclofenac", "NSAID for arthritis/acute pain.", "Indigestion, heart risk."),
            ("Naproxen", "Long-acting NSAID.", "Stomach upset, drowsiness."),
            ("Tramadol", "Opioid pain medication.", "Dizziness, addiction risk."),
            ("Celecoxib", "COX-2 inhibitor for arthritis.", "High blood pressure, stomach pain."),
            ("Morphine", "Strong opioid for severe pain.", "Drowsiness, respiratory depression."),
            ("Codeine", "Opioid for pain/cough.", "Constipation, drowsiness."),
            ("Oxycodone", "Strong opioid painkiller.", "High addiction risk, constipation."),
            ("Hydrocodone", "Opioid for moderate to severe pain.", "Drowsiness, nausea."),
            ("Fentanyl", "Potent synthetic opioid.", "Severe respiratory distress, overdose risk."),
            ("Methadone", "Opioid for pain/addiction detox.", "Arrhythmia, sweating."),
            ("Meloxicam", "NSAID for joint pain.", "Indigestion, dizziness."),
            ("Indomethacin", "Strong NSAID for gout.", "Headache, stomach bleeding."),
            ("Ketorolac", "Strong short-term NSAID.", "Kidney failure risk, bleeding."),
            ("Mefenamic Acid", "NSAID for menstrual pain.", "Stomach pain, drowsiness."),
            ("Etoricoxib", "COX-2 inhibitor for pain.", "Palpitations, hypertension."),
            ("Piroxicam", "Anti-inflammatory for arthritis.", "Skin reactions, stomach upset."),
            ("Buprenorphine", "Opioid for pain/addiction.", "Headache, withdrawal symptoms."),

            # --- 2. ANTIBIOTICS & INFECTIOUS DISEASE ---
            ("Amoxicillin", "Penicillin antibiotic.", "Nausea, rash, yeast infections."),
            ("Azithromycin", "Macrolide antibiotic.", "Diarrhea, stomach pain."),
            ("Ciprofloxacin", "Fluoroquinolone antibiotic.", "Tendon rupture, nerve damage."),
            ("Doxycycline", "Tetracycline antibiotic.", "Sun sensitivity, nausea."),
            ("Cephalexin", "Cephalosporin antibiotic.", "Diarrhea, rash."),
            ("Clindamycin", "Antibiotic for serious infections.", "Severe diarrhea (C. diff risk)."),
            ("Metronidazole", "Antibiotic for anaerobes.", "Metallic taste, no alcohol allowed."),
            ("Levofloxacin", "Respiratory antibiotic.", "Tendonitis, insomnia."),
            ("Augmentin", "Amoxicillin + Clavulanic Acid.", "Diarrhea, nausea."),
            ("Sulfamethoxazole", "Sulfa antibiotic (Bactrim).", "Sun sensitivity, rash."),
            ("Trimethoprim", "Used for UTIs.", "Folic acid deficiency, rash."),
            ("Nitrofurantoin", "Urinary tract antiseptic.", "Brown urine, cough."),
            ("Vancomycin", "For MRSA/C. diff.", "Kidney toxicity, hearing loss."),
            ("Gentamicin", "Strong IV antibiotic.", "Kidney damage, hearing loss."),
            ("Erythromycin", "Macrolide antibiotic.", "Stomach cramps, nausea."),
            ("Clarithromycin", "For H. pylori/respiratory.", "Metallic taste, drug interactions."),
            ("Penicillin V", "Basic penicillin.", "Allergic reaction, nausea."),
            ("Minocycline", "For acne and infections.", "Dizziness, skin discoloration."),
            ("Moxifloxacin", "Fluoroquinolone antibiotic.", "Heart rhythm changes."),
            ("Rifampin", "For TB/carriers.", "Orange urine/tears, liver strain."),
            ("Fluconazole", "Antifungal (Diflucan).", "Nausea, headache."),
            ("Ketoconazole", "Antifungal.", "Liver toxicity, adrenal issues."),
            ("Terbinafine", "Antifungal for nails.", "Taste loss, liver strain."),
            ("Acyclovir", "Antiviral for herpes.", "Nausea, kidney crystals."),
            ("Valacyclovir", "Antiviral (Valtrex).", "Headache, nausea."),
            ("Oseltamivir", "Antiviral for flu (Tamiflu).", "Nausea, vomiting."),

            # --- 3. CARDIOVASCULAR (HEART & BP) ---
            ("Lisinopril", "ACE inhibitor for BP.", "Dry cough, high potassium."),
            ("Amlodipine", "Calcium channel blocker.", "Ankle swelling, flushing."),
            ("Metoprolol", "Beta-blocker.", "Fatigue, slow heart rate."),
            ("Atorvastatin", "Statin for cholesterol.", "Muscle pain, liver enzymes."),
            ("Losartan", "ARB for blood pressure.", "Dizziness, back pain."),
            ("Simvastatin", "Statin.", "Muscle weakness, interaction risk."),
            ("Carvedilol", "Beta-blocker for heart failure.", "Weight gain, dizziness."),
            ("Clopidogrel", "Antiplatelet (Plavix).", "Bleeding, bruising."),
            ("Warfarin", "Blood thinner.", "Severe bleeding risk."),
            ("Furosemide", "Diuretic (Lasix).", "Dehydration, low potassium."),
            ("Hydrochlorothiazide", "Diuretic.", "Frequent urination, sun sensitivity."),
            ("Spironolactone", "K-sparing diuretic.", "High potassium, gynecomastia."),
            ("Ramipril", "ACE inhibitor.", "Cough, hypotension."),
            ("Rosuvastatin", "Potent statin (Crestor).", "Headache, muscle pain."),
            ("Bisoprolol", "Beta-blocker.", "Fatigue, cold extremities."),
            ("Atenolol", "Beta-blocker.", "Depression, fatigue."),
            ("Propranolol", "Beta-blocker for anxiety/BP.", "Dreams, fatigue."),
            ("Diltiazem", "Calcium channel blocker.", "Edema, constipation."),
            ("Verapamil", "Calcium channel blocker.", "Constipation, dizziness."),
            ("Valsartan", "ARB for BP.", "Dizziness, kidney strain."),
            ("Telmisartan", "ARB for BP.", "Back pain, sinus pain."),
            ("Olmesartan", "ARB for BP.", "Dizziness, severe diarrhea."),
            ("Digoxin", "Heart failure med.", "Nausea, yellow vision."),
            ("Nitroglycerin", "For chest pain.", "Severe headache, hypotension."),
            ("Isosorbide Mononitrate", "Long-acting nitrate.", "Headache, dizziness."),
            ("Hydralazine", "Vasodilator.", "Headache, palpitations."),
            ("Clonidine", "Central BP med.", "Dry mouth, sedation."),
            ("Amiodarone", "Anti-arrhythmic.", "Thyroid/lung issues, blue skin."),
            ("Rivaroxaban", "Blood thinner (Xarelto).", "Bleeding risk."),
            ("Apixaban", "Blood thinner (Eliquis).", "Bleeding risk."),
            ("Dabigatran", "Blood thinner.", "Gastritis, bleeding."),
            ("Enoxaparin", "Injectable blood thinner.", "Bruising, bleeding."),
            ("Pravastatin", "Statin.", "Muscle pain (mild)."),
            ("Lovastatin", "Statin.", "Muscle pain."),
            ("Ezetimibe", "Cholesterol absorption inhibitor.", "Diarrhea, fatigue."),
            ("Fenofibrate", "Lowers triglycerides.", "Liver strain, muscle pain."),
            ("Gemfibrozil", "Lowers triglycerides.", "Indigestion, gallstones."),

            # --- 4. DIABETES & ENDOCRINE ---
            ("Metformin", "Type 2 diabetes first-line.", "Diarrhea, lactic acidosis risk."),
            ("Insulin Glargine", "Long-acting insulin.", "Hypoglycemia, weight gain."),
            ("Insulin Lispro", "Rapid-acting insulin.", "Hypoglycemia."),
            ("Glimepiride", "Stimulates insulin.", "Low blood sugar, weight gain."),
            ("Glipizide", "Sulfonylurea.", "Hypoglycemia, weight gain."),
            ("Glyburide", "Sulfonylurea.", "Hypoglycemia (long lasting)."),
            ("Sitagliptin", "DPP-4 inhibitor.", "Runny nose, joint pain."),
            ("Linagliptin", "DPP-4 inhibitor.", "Nasopharyngitis."),
            ("Empagliflozin", "SGLT2 inhibitor.", "UTIs, genital infections."),
            ("Dapagliflozin", "SGLT2 inhibitor.", "UTIs, dehydration."),
            ("Canagliflozin", "SGLT2 inhibitor.", "Amputation risk (rare), UTIs."),
            ("Pioglitazone", "TZD for diabetes.", "Fluid retention, heart failure risk."),
            ("Liraglutide", "GLP-1 agonist.", "Nausea, weight loss."),
            ("Semaglutide", "GLP-1 agonist (Ozempic).", "Nausea, vomiting."),
            ("Levothyroxine", "Thyroid hormone.", "Palpitations (if dose high)."),
            ("Methimazole", "For hyperthyroidism.", "Rash, liver issues."),
            ("Prednisone", "Corticosteroid.", "Weight gain, mood changes, high sugar."),
            ("Hydrocortisone", "Steroid.", "Fluid retention, hunger."),
            ("Dexamethasone", "Potent steroid.", "Insomnia, agitation."),
            ("Testosterone", "Hormone replacement.", "Acne, mood aggression."),
            ("Estradiol", "Estrogen replacement.", "Blood clot risk, nausea."),
            ("Progesterone", "Progestin hormone.", "Bloating, mood swings."),
            ("Alendronate", "For osteoporosis.", "Esophageal irritation, jaw necrosis."),

            # --- 5. GASTROINTESTINAL ---
            ("Omeprazole", "PPI for acid reflux.", "Headache, B12 deficiency."),
            ("Pantoprazole", "PPI.", "Diarrhea, bone fracture risk."),
            ("Esomeprazole", "Strong PPI.", "Headache, dry mouth."),
            ("Lansoprazole", "PPI.", "Stomach pain."),
            ("Rabeprazole", "PPI.", "Insomnia, gas."),
            ("Famotidine", "H2 blocker (Pepcid).", "Dizziness, constipation."),
            ("Ranitidine", "H2 blocker (Zantac).", "Headache (Recall issues)."),
            ("Cimetidine", "H2 blocker.", "Gynecomastia, drug interactions."),
            ("Ondansetron", "Anti-nausea (Zofran).", "Constipation, QT prolongation."),
            ("Metoclopramide", "Gut motility stimulator.", "Drowsiness, twitching."),
            ("Promethazine", "Anti-nausea.", "Severe sedation."),
            ("Bisacodyl", "Stimulant laxative.", "Cramps, electrolyte loss."),
            ("Senna", "Natural laxative.", "Cramps, brown urine."),
            ("Docusate", "Stool softener.", "Mild cramping."),
            ("Loperamide", "Anti-diarrheal (Imodium).", "Constipation, cramps."),
            ("Dicyclomine", "For IBS spasms.", "Dry mouth, blurred vision."),
            ("Hyoscyamine", "For gut spasms.", "Dry mouth, urinary retention."),
            ("Sucralfate", "Coats stomach ulcers.", "Constipation."),
            ("Misoprostol", "Prevents NSAID ulcers.", "Diarrhea, miscarriage risk."),
            ("Lactulose", "Laxative/Ammonia reducer.", "Gas, bloating."),

            # --- 6. MENTAL HEALTH & NEUROLOGY ---
            ("Sertraline", "SSRI (Zoloft).", "Nausea, insomnia, sexual dysfunction."),
            ("Fluoxetine", "SSRI (Prozac).", "Anxiety, weight loss."),
            ("Escitalopram", "SSRI (Lexapro).", "Fatigue, sweating."),
            ("Citalopram", "SSRI (Celexa).", "Drowsiness, heart rhythm changes."),
            ("Paroxetine", "SSRI (Paxil).", "Weight gain, withdrawal shocks."),
            ("Venlafaxine", "SNRI (Effexor).", "Nausea, high BP, sweating."),
            ("Duloxetine", "SNRI (Cymbalta).", "Nausea, dry mouth."),
            ("Bupropion", "Antidepressant (Wellbutrin).", "Anxiety, seizure risk."),
            ("Trazodone", "Sleep aid/antidepressant.", "Drowsiness, priapism."),
            ("Mirtazapine", "Antidepressant.", "Sedation, increased appetite."),
            ("Amitriptyline", "Tricyclic.", "Dry mouth, sedation, weight gain."),
            ("Nortriptyline", "Tricyclic.", "Constipation, dry mouth."),
            ("Alprazolam", "Benzo (Xanax).", "Sedation, high addiction risk."),
            ("Clonazepam", "Benzo (Klonopin).", "Drowsiness, coordination loss."),
            ("Lorazepam", "Benzo (Ativan).", "Sedation, memory loss."),
            ("Diazepam", "Benzo (Valium).", "Muscle weakness, drowsiness."),
            ("Buspirone", "Anti-anxiety.", "Dizziness, nausea."),
            ("Gabapentin", "Nerve pain.", "Dizziness, swelling."),
            ("Pregabalin", "Nerve pain (Lyrica).", "Weight gain, euphoria."),
            ("Topiramate", "Migraine/Seizures.", "Brain fog, weight loss."),
            ("Lamotrigine", "Mood stabilizer.", "Severe rash (Stevens-Johnson)."),
            ("Levetiracetam", "Anti-seizure (Keppra).", "Irritability, fatigue."),
            ("Valproic Acid", "Mood stabilizer.", "Liver toxicity, weight gain."),
            ("Quetiapine", "Antipsychotic (Seroquel).", "Sedation, weight gain."),
            ("Olanzapine", "Antipsychotic.", "Significant weight gain, diabetes."),
            ("Risperidone", "Antipsychotic.", "Movement disorders, hormonal changes."),
            ("Aripiprazole", "Antipsychotic (Abilify).", "Restlessness (akathisia)."),
            ("Lithium", "Bipolar treatment.", "Thirst, tremors, kidney issues."),
            ("Methylphenidate", "ADHD (Ritalin).", "Insomnia, appetite loss."),
            ("Amphetamine", "ADHD (Adderall).", "Insomnia, heart palpitations."),
            ("Donepezil", "For dementia.", "Nausea, diarrhea."),
            ("Memantine", "For Alzheimer's.", "Dizziness, confusion."),
            ("Sumatriptan", "Migraine rescue.", "Chest tightness, tingling."),

            # --- 7. RESPIRATORY & ALLERGY ---
            ("Albuterol", "Rescue inhaler.", "Tremors, racing heart."),
            ("Salmeterol", "Long-acting bronchodilator.", "Headache, throat irritation."),
            ("Fluticasone", "Steroid inhaler/spray.", "Thrush, nosebleeds."),
            ("Budesonide", "Steroid inhaler.", "Voice hoarseness."),
            ("Mometasone", "Steroid nasal spray.", "Headache, viral infection."),
            ("Montelukast", "Asthma/Allergies (Singulair).", "Mood changes, nightmares."),
            ("Ipratropium", "COPD inhaler.", "Dry mouth, bitter taste."),
            ("Tiotropium", "COPD inhaler (Spiriva).", "Dry mouth, constipation."),
            ("Theophylline", "Older asthma drug.", "Nausea, insomnia, toxicity."),
            ("Cetirizine", "Antihistamine (Zyrtec).", "Drowsiness (mild)."),
            ("Loratadine", "Antihistamine (Claritin).", "Headache."),
            ("Fexofenadine", "Antihistamine (Allegra).", "Headache, nausea."),
            ("Diphenhydramine", "Benadryl.", "Severe drowsiness, dry mouth."),
            ("Hydroxyzine", "Antihistamine/Anxiety.", "Sedation, dry mouth."),
            ("Guaifenesin", "Mucus thinner (Mucinex).", "Nausea."),
            ("Dextromethorphan", "Cough suppressant.", "Dizziness, drowsiness."),
            ("Pseudoephedrine", "Decongestant.", "Insomnia, high BP."),
            ("Benzonatate", "Cough pearl.", "Numbness, dizziness."),

            # --- 8. DERMATOLOGY & OTHERS ---
            ("Isotretinoin", "Severe acne (Accutane).", "Severe birth defects, dry skin."),
            ("Tretinoin", "Topical acne/wrinkles.", "Skin peeling, sun sensitivity."),
            ("Hydrocortisone Cream", "Itch relief.", "Skin thinning."),
            ("Mupirocin", "Topical antibiotic.", "Burning, stinging."),
            ("Ketoconazole Cream", "Fungal skin infection.", "Irritation."),
            ("Permethrin", "Scabies/Lice.", "Itching, redness."),
            ("Finasteride", "Hair loss/Prostate.", "Libido loss."),
            ("Dutasteride", "Enlarged prostate.", "Impotence."),
            ("Tamsulosin", "Flomax (Prostate).", "Dizziness, hypotension."),
            ("Sildenafil", "Viagra.", "Flushing, blue vision."),
            ("Tadalafil", "Cialis.", "Back pain, headache."),
            ("Allopurinol", "Gout prevention.", "Rash, kidney stones."),
            ("Colchicine", "Acute gout.", "Diarrhea, vomiting."),
            ("Methotrexate", "Rheumatoid Arthritis.", "Liver toxicity, nausea."),
            ("Hydroxychloroquine", "Lupus/RA.", "Retinal damage (eye exams needed)."),
            ("Folic Acid", "Supplement.", "Nausea."),
            ("Vitamin D", "Supplement.", "Fatigue (if overdose)."),
            ("Ferrous Sulfate", "Iron supplement.", "Constipation, black stools."),
            ("Cyanocobalamin", "Vitamin B12.", "Injection site pain."),
            ("Potassium Chloride", "Electrolyte.", "Stomach upset."),
            ("Magnesium Oxide", "Supplement/Laxative.", "Diarrhea.")
        ]

        # Populate the lists
        for name, desc, side_effect in raw_data:
            self.valid_drugs.append(name.lower())
            context = f"Drug: {name}. Description: {desc} Side Effects: {side_effect}"
            self.knowledge_base.append(context)

        # --- 3. Add Specific Interaction Rules (CRITICAL for T5) ---
        # T5 needs these explicitly to generate good interaction warnings
        self.knowledge_base.append("Interaction between Ibuprofen and Lisinopril: Ibuprofen reduces the efficacy of Lisinopril and strains kidneys.")
        self.knowledge_base.append("Interaction between Aspirin and Warfarin: Major bleeding risk.")
        self.knowledge_base.append("Interaction between Paracetamol and Warfarin: Prolonged use raises INR (bleeding risk).")
        self.knowledge_base.append("Interaction between Sildenafil and Nitrates (Nitroglycerin): Dangerous drop in blood pressure.")
        self.knowledge_base.append("Interaction between Antibiotics and Birth Control: Antibiotics may reduce contraceptive effectiveness.")
        self.knowledge_base.append("Interaction between Metformin and Prednisone: Steroids raise blood sugar, opposing Metformin.")
        self.knowledge_base.append("Interaction between Tramadol and SSRIs (like Sertraline): Risk of Serotonin Syndrome.")
        self.knowledge_base.append("Interaction between Simvastatin and Amlodipine: Increased risk of myopathy (muscle damage).")
        self.knowledge_base.append("Interaction between Levothyroxine and Calcium/Iron: Absorption of thyroid med is blocked.")
        self.knowledge_base.append("Interaction between Albuterol and Beta-Blockers: They oppose each other; risk of bronchospasm.")
        self.knowledge_base.append("Interaction between Clarithromycin and Statins: High risk of muscle breakdown.")
        self.knowledge_base.append("Interaction between Alcohol and Metronidazole: Severe vomiting and nausea.")
        
        print(f"Database Ready! I know about {len(self.valid_drugs)} medicines.")

        # --- 4. Calculate Embeddings ---
        self.kb_embeddings = self._get_bert_embeddings(self.knowledge_base)

    def _get_bert_embeddings(self, text_list):
        inputs = self.bert_tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def validate_drugs(self, drug_list):
        unknowns = []
        for drug in drug_list:
            if drug.lower() not in self.valid_drugs:
                unknowns.append(drug)
        return unknowns

    def get_response(self, user_query):
        # 1. Retrieval
        query_embedding = self._get_bert_embeddings([user_query])
        similarities = cosine_similarity(query_embedding, self.kb_embeddings)
        
        # Top 3 facts (Broad context)
        top_k = 3
        top_indices = np.argsort(similarities[0])[-top_k:]
        best_context = " ".join([self.knowledge_base[i] for i in top_indices])
        
        # 2. Generation
        input_text = f"question: {user_query}  context: {best_context}"
        input_ids = self.t5_tokenizer(input_text, return_tensors="pt").input_ids
        
        outputs = self.t5_model.generate(input_ids, max_length=150, length_penalty=2.0, num_beams=2)
        answer = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

print("System Starting...")
bot = MedicalAI()

@app.route('/ask', methods=['POST'])
def ask_bot():
    data = request.json
    user_text = data.get('message')
    drug_list = data.get('drugs')

    if not user_text or not drug_list:
        return jsonify({"reply": "Error: Missing data."})

    unknown_drugs = bot.validate_drugs(drug_list)
    if len(unknown_drugs) > 0:
        return jsonify({
            "reply": f"I do not recognize: {', '.join(unknown_drugs)}. Check spelling.",
            "is_error": True
        })

    ai_response = bot.get_response(user_text)
    return jsonify({"reply": ai_response, "is_error": False})

if __name__ == "__main__":
    print("Starting Medical AI Server on port 3000...")
    app.run(port=3000, debug=True)