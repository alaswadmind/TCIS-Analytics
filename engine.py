import os
import re
import json
import collections
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_KEYS = {
    'complaint': 'msg_complaint_roberta',
    'sentiment': 'msg_sentiment_roberta',
    'severity': 'msg_severity_roberta',
    'services': 'msg_services_roberta',
    'l2': 'msg_L2_roberta',
    'l3': 'msg_L3_roberta',
}

class TCISEngine:
    def __init__(self, base_dir=BASE_DIR):
        self.base_dir = base_dir
        self.cache = {}
        self.taxonomy = self._load_taxonomy()
        self.topics_and_alerts = self._load_topics_and_alerts()

    def _load_taxonomy(self):
        tax_path = os.path.join(self.base_dir, 'tcis_taxonomy.json')
        if os.path.exists(tax_path):
            with open(tax_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'services_to_l2': {}, 'l2_to_l3': {}}

    def _load_topics_and_alerts(self):
        path = os.path.join(self.base_dir, 'tcis_topics_and_alerts.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'bertopic_topics': [], 'collective_alerts': [], 'daily_service_spikes': []}

    def get_model(self, model_key):
        folder_name = MODEL_KEYS.get(model_key, model_key)
        model_path = os.path.join(self.base_dir, folder_name)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        if folder_name not in self.cache:
            tok = AutoTokenizer.from_pretrained(model_path)
            mdl = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE).eval()
            meta_path = os.path.join(model_path, 'tcis_meta.json')
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            self.cache[folder_name] = (tok, mdl, meta)

        return self.cache[folder_name]

    @torch.no_grad()
    def predict_proba(self, model_key, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        tok, mdl, meta = self.get_model(model_key)
        out = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])
            enc = tok(
                batch,
                truncation=True,
                max_length=meta.get('maxlen', 96),
                padding=True,
                return_tensors='pt'
            ).to(DEVICE)
            logits = mdl(**enc).logits.float().cpu().numpy()
            if meta.get('kind') == 'multilabel':
                probs = 1.0 / (1.0 + np.exp(-logits))
            else:
                exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            out.append(probs)

        return np.vstack(out), meta

    @staticmethod
    def extract_customer_turns(text):
        lines = [line.strip() for line in str(text).split('\n') if line.strip()]
        cust = [line[len('[CUSTOMER]'):].strip() for line in lines if line.upper().startswith('[CUSTOMER]')]
        if cust:
            return ' '.join(cust)
        return str(text).strip()

    def hierarchical_decode(self, p1, p2, p3, meta1, meta2, meta3, k_services=2):
        tax_l2 = self.taxonomy.get('services_to_l2', {})
        tax_l3 = self.taxonomy.get('l2_to_l3', {})

        # Services L1
        thr = meta1.get('thr') or [0.5] * len(meta1['labels'])
        svc = [meta1['labels'][j] for j in np.argsort(-p1)[:k_services] if p1[j] >= thr[j]]
        if not svc:
            svc = [meta1['labels'][int(np.argmax(p1))]]

        # Allowed L2s based on chosen services
        allowed2 = set()
        for s in svc:
            allowed2.update(tax_l2.get(s, []))

        l2_labels = meta2['labels']
        m2 = np.array([(x in allowed2) or x == 'other_rare' for x in l2_labels])
        if not np.any(m2):
            m2 = np.ones(len(l2_labels), dtype=bool)

        i2 = int(np.argmax(np.where(m2, p2, -1.0)))
        l2_best = l2_labels[i2]
        l2_conf = float(p2[i2])

        # Allowed L3s based on chosen L2
        allowed3 = set(tax_l3.get(l2_best, []))
        l3_labels = meta3['labels']
        m3 = np.array([(x in allowed3) or x == 'other_rare' for x in l3_labels])
        if not np.any(m3):
            m3 = np.ones(len(l3_labels), dtype=bool)

        i3 = int(np.argmax(np.where(m3, p3, -1.0)))
        l3_best = l3_labels[i3]
        l3_conf = float(p3[i3])

        return svc, l2_best, l2_conf, l3_best, l3_conf

    def find_nearest_cluster(self, text, detected_services, l2_cat):
        """Matches input text with the nearest BERTopic cluster & returns collective anomaly status"""
        text_lower = text.lower()
        topics = self.topics_and_alerts.get('bertopic_topics', [])
        alerts = {a['cluster_id']: a for a in self.topics_and_alerts.get('collective_alerts', [])}

        best_topic = None
        best_score = -1

        # Priority rules
        if any(w in text_lower for w in ['blinking', 'red light', 'los light', 'optical', 'router keeps', 'green light']):
            for t in topics:
                if t['topic'] == 135:
                    best_topic = t
                    best_score = 100
                    break

        if not best_topic and any(w in text_lower for w in ['iphone', 'apple deal', 'trade in', 'upgrade']):
            for t in topics:
                if t['topic'] == 3:
                    best_topic = t
                    best_score = 90
                    break

        if not best_topic and any(w in text_lower for w in ['website', 'tuesday', 'login problem', 'support portal', 'company page']):
            for t in topics:
                if t['topic'] == 101:
                    best_topic = t
                    best_score = 85
                    break

        if not best_topic and any(w in text_lower for w in ['sim swap', 'sim card', 'activate sim', 'unregistered sim']):
            for t in topics:
                if t['topic'] == 17:
                    best_topic = t
                    best_score = 80
                    break

        if not best_topic:
            for t in topics:
                score = 0
                for w in t.get('representative_words', []):
                    if w in text_lower:
                        score += 3
                if t.get('category') in detected_services:
                    score += 2
                if score > best_score:
                    best_score = score
                    best_topic = t

        if not best_topic and topics:
            best_topic = topics[0]

        cluster_id = best_topic['topic'] if best_topic else 0
        topic_name = best_topic['name'] if best_topic else "0_general_inquiry"

        if cluster_id in alerts:
            incident_status = alerts[cluster_id]['alert_level']  # 'Potential Anomaly' or 'Watchlist Candidate'
            votes = alerts[cluster_id].get('count_24h', 4)
        elif cluster_id == 135:
            incident_status = "Normal Baseline"
            votes = 1
        elif best_score >= 5:
            incident_status = "Normal Baseline"
            votes = 2
        else:
            incident_status = "Normal Baseline"
            votes = 1

        return {
            "id": cluster_id,
            "topic": topic_name,
            "votes": votes,
            "incident_status": incident_status
        }

    def generate_incident_summary(self, services, symptom, likely_cause="unknown or local line degradation"):
        """Structured incident summary generator in standardized English format"""
        svc_str = ", ".join(services) if isinstance(services, list) else str(services)
        return f"Affected Service: {svc_str}; Primary Symptom: {symptom}; Probable Cause: {likely_cause} (Preliminary Indicator)"

    def analyze(self, text):
        clean_text = str(text).strip()
        customer_text = self.extract_customer_turns(clean_text)
        is_conversation = '[CUSTOMER]' in clean_text.upper() or '[COMPANY]' in clean_text.upper()

        # 1. Complaint detection
        p_comp, m_comp = self.predict_proba('complaint', [clean_text])
        comp_idx = int(p_comp[0].argmax())
        comp_label = m_comp['labels'][comp_idx]
        comp_conf = float(p_comp[0][comp_idx])
        comp_scores = {lbl: float(round(p, 4)) for lbl, p in zip(m_comp['labels'], p_comp[0])}

        # 2. Sentiment analysis
        p_sent, m_sent = self.predict_proba('sentiment', [customer_text])
        sent_idx = int(p_sent[0].argmax())
        sent_label = m_sent['labels'][sent_idx]
        sent_conf = float(p_sent[0][sent_idx])
        sent_scores = {lbl: float(round(p, 4)) for lbl, p in zip(m_sent['labels'], p_sent[0])}

        # 3. Severity assessment
        p_sev, m_sev = self.predict_proba('severity', [customer_text])
        sev_idx = int(p_sev[0].argmax())
        sev_label = m_sev['labels'][sev_idx]
        sev_conf = float(p_sev[0][sev_idx])
        sev_scores = {lbl: float(round(p, 4)) for lbl, p in zip(m_sev['labels'], p_sev[0])}

        # 4. Services L1
        p_svc, m_svc = self.predict_proba('services', [customer_text])
        svc_scores = {lbl: float(round(p, 4)) for lbl, p in zip(m_svc['labels'], p_svc[0])}

        # 5. Issue Category L2
        p_l2, m_l2 = self.predict_proba('l2', [customer_text])
        raw_l2_idx = int(p_l2[0].argmax())
        raw_l2_label = m_l2['labels'][raw_l2_idx]
        top_l2_scores = {m_l2['labels'][i]: float(round(p_l2[0][i], 4)) for i in np.argsort(-p_l2[0])[:5]}

        # 6. Specific Issue L3
        p_l3, m_l3 = self.predict_proba('l3', [customer_text])
        raw_l3_idx = int(p_l3[0].argmax())
        raw_l3_label = m_l3['labels'][raw_l3_idx]
        top_l3_scores = {m_l3['labels'][i]: float(round(p_l3[0][i], 4)) for i in np.argsort(-p_l3[0])[:5]}

        # Hierarchical decoding
        active_services, hier_l2, hier_l2_conf, hier_l3, hier_l3_conf = self.hierarchical_decode(
            p_svc[0], p_l2[0], p_l3[0], m_svc, m_l2, m_l3
        )

        # Nearest Cluster & Collective Anomaly Match
        nearest_cl = self.find_nearest_cluster(clean_text, active_services, hier_l2)

        # Incident Summary
        incident_summary = self.generate_incident_summary(
            active_services,
            symptom=hier_l2.replace('_', ' '),
            likely_cause="hardware/line degradation" if 'connection' in hier_l2 else "service provisioning or billing system"
        )

        # Composite priority / urgency calculation (0 - 100)
        complaint_weight = 1.0 if comp_label == 'Complaint' else (0.4 if comp_label == 'Ambiguous' else 0.0)
        severity_weight = 1.0 if sev_label == 'High/Critical' else (0.5 if sev_label == 'Medium' else 0.1)
        sentiment_weight = 1.0 if sent_label == 'Negative' else (0.6 if sent_label == 'Mixed' else (0.2 if sent_label == 'Neutral' else 0.0))

        priority_score = int(round((complaint_weight * 0.4 + severity_weight * 0.35 + sentiment_weight * 0.25) * 100))

        # Recommended Action & SLA
        if priority_score >= 75:
            sla_time = "Within 15 - 30 minutes (Urgent/Critical)"
            action_desc = "Immediate Tier-2 Engineering escalation and operational incident tracking."
            priority_level = "Critical"
        elif priority_score >= 45:
            sla_time = "Within 2 - 4 hours (Standard)"
            action_desc = "Standard automated dispatch to specialized customer support and billing desk."
            priority_level = "Medium"
        else:
            sla_time = "Within 24 hours / Automated Self-Service"
            action_desc = "Automated knowledge-base resolution or standard inquiry logging."
            priority_level = "Low"

        # Standard Unified Prediction Output
        notebook_output = {
            "complaint": {comp_label: round(comp_conf, 3)},
            "sentiment": {sent_label: round(sent_conf, 3)},
            "severity": {sev_label: round(sev_conf, 3)},
            "services_L1": active_services,
            "L2": hier_l2,
            "L3": hier_l3,
            "nearest_cluster": nearest_cl
        }

        return {
            **notebook_output,
            "incident_summary": incident_summary,
            "input_metadata": {
                "character_count": len(clean_text),
                "is_conversation": is_conversation,
                "analyzed_text": customer_text
            },
            "detailed_scores": {
                "complaint": {"label": comp_label, "confidence": round(comp_conf, 4), "scores": comp_scores},
                "sentiment": {"label": sent_label, "confidence": round(sent_conf, 4), "scores": sent_scores},
                "severity": {"label": sev_label, "confidence": round(sev_conf, 4), "scores": sev_scores},
                "services": svc_scores,
                "l2": {"hierarchical": hier_l2, "conf": round(hier_l2_conf, 4), "raw": raw_l2_label, "top5": top_l2_scores},
                "l3": {"hierarchical": hier_l3, "conf": round(hier_l3_conf, 4), "raw": raw_l3_label, "top5": top_l3_scores},
                "is_consistent": (hier_l2 == raw_l2_label and hier_l3 == raw_l3_label)
            },
            "intelligence": {
                "priority_score": priority_score,
                "priority_level": priority_level,
                "sla_window": sla_time,
                "recommended_action": action_desc
            }
        }
