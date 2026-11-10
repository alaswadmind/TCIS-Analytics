import os
import io
import time
import json
import collections
from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine import TCISEngine, MODEL_KEYS

app = FastAPI(
    title="TCIS - Telecom Customer Intelligence System",
    description="Enterprise Artificial Intelligence Platform for Multi-Task Telecom Customer Analytics, Hierarchical Classification, and Collective Outage Surveillance.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = TCISEngine()

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

DEMO_SAMPLES = [
    {
        "id": "sample_fiber_outage",
        "title": "Home Fiber Outage & Router Optical Alarm",
        "category": "Internet & Hardware",
        "type": "conversation",
        "text": "[CUSTOMER] @telecom_care My optical fiber internet has been completely down since 8:00 AM! The LOS optical indicator on the router is flashing red continuously. I am unable to work remotely today and need this resolved immediately!\n[COMPANY] Hello! We apologize for the connection drop. Could you please share your service account number so we can test the local optical line?"
    },
    {
        "id": "sample_iphone_alert_cluster3",
        "title": "Collective Outage Anomaly: iPhone Trade-in & Plan Billing (Cluster 3)",
        "category": "Subscription & Billing",
        "type": "single",
        "text": "I attempted to upgrade my plan under the promotional iPhone trade-in campaign at your retail branch, but the automated billing system deducted the full retail price without crediting my balance."
    },
    {
        "id": "sample_portal_alert_cluster101",
        "title": "Watchlist Candidate: Support Portal 502 Downtime (Cluster 101)",
        "category": "Technical Support",
        "type": "conversation",
        "text": "[CUSTOMER] Both the customer self-service mobile app and the support account portal are failing to load since Tuesday morning with error 502 Bad Gateway.\n[COMPANY] We are currently investigating server gateway availability with our IT infrastructure team."
    },
    {
        "id": "sample_sim_swap_fraud_risk",
        "title": "SIM Swap & Network Deactivation (Cluster 17)",
        "category": "Number & SIM",
        "type": "single",
        "text": "I performed an authorized SIM swap at your central store yesterday, but now my handset shows 'No Service / Unregistered SIM'. I cannot place voice calls or receive essential bank SMS verification codes!"
    },
    {
        "id": "sample_speed_throttling",
        "title": "Severe Throughput Degradation & Streaming Buffer",
        "category": "Mobile Data",
        "type": "single",
        "text": "Subscribed to the uncapped 5G plan, but downstream throughput has plummeted to under 2 Mbps tonight. High-definition video streams buffer continuously."
    },
    {
        "id": "sample_polite_inquiry",
        "title": "Polite General Inquiry / Non-Complaint",
        "category": "General Information",
        "type": "single",
        "text": "Good day! Could you please provide information regarding the available weekly international roaming bundles for travelers visiting France next month? Thank you for your assistance!"
    }
]

class AnalyzeRequest(BaseModel):
    text: str

class SummarizeRequest(BaseModel):
    services: List[str]
    symptom: str
    likely_cause: Optional[str] = "unknown or line degradation"

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>TCIS Enterprise Analytics Platform is active.</h1>"

@app.get("/presentation", response_class=HTMLResponse)
async def serve_presentation():
    pres_file = os.path.join(engine.base_dir, "presentation.html")
    if os.path.exists(pres_file):
        with open(pres_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Presentation document not found.</h1>"

@app.get("/api/health")
async def health_check():
    return {
        "status": "operational",
        "engine": "TCIS Optimized Multi-Task Framework",
        "models_count": len(MODEL_KEYS),
        "execution_device": str(engine.get_model('complaint')[1].device if 'msg_complaint_roberta' in engine.cache else 'CPU/CUDA FP16'),
        "system_time": time.time()
    }

@app.get("/api/demo-samples")
async def get_demo_samples():
    return {"samples": DEMO_SAMPLES}

@app.get("/api/models-info")
async def get_models_info():
    info = {}
    for key, folder in MODEL_KEYS.items():
        meta_path = os.path.join(engine.base_dir, folder, "tcis_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            info[key] = {
                "folder": folder,
                "kind": meta.get("kind"),
                "maxlen": meta.get("maxlen"),
                "labels_count": len(meta.get("labels", [])),
                "labels": meta.get("labels", []),
                "thresholds": meta.get("thr")
            }
    return {
        "models": info,
        "taxonomy": engine.taxonomy
    }

@app.get("/api/topics")
async def get_topics():
    """Returns the BERTopic extracted topics, cluster statistics, and representative n-grams"""
    topics_list = engine.topics_and_alerts.get("bertopic_topics", [])
    
    # Calculate category aggregation for visual distribution
    cat_distribution = collections.defaultdict(int)
    for t in topics_list:
        cat_distribution[t.get('category', 'other')] += t.get('count', 0)

    return {
        "total_clusters": 228,
        "active_displayed_topics": len(topics_list),
        "noise_ratio": 0.348,
        "homogeneity_score_vs_l2": 0.597,
        "nmi_score": 0.473,
        "category_distribution": dict(cat_distribution),
        "topics": topics_list
    }

@app.get("/api/collective-alerts")
async def get_collective_alerts():
    """Returns Poisson 24h surveillance indicators, FDR q-values, and daily volume burst spikes"""
    return {
        "surveillance_engine": {
            "window_size_hours": 24,
            "statistical_model": "Poisson Survival Scan",
            "fdr_adjustment": "Benjamini-Hochberg (q-value)",
            "significance_threshold": 0.01,
            "notice": "Statistical surveillance indicators represent automated anomaly detection for operations and triage, not conclusive technical proof of physical carrier network failures without NOC validation."
        },
        "active_collective_alerts": engine.topics_and_alerts.get("collective_alerts", []),
        "daily_service_spikes": engine.topics_and_alerts.get("daily_service_spikes", [])
    }

@app.get("/api/model-comparison")
async def get_model_comparison():
    """Returns official test set Macro-F1 evaluation matrix comparing Deep Transformers against Baselines"""
    return {
        "metric": "Macro-F1 on Held-out Test Set",
        "matrix": engine.topics_and_alerts.get("model_comparison_matrix", [])
    }

@app.post("/api/summarize-incident")
async def summarize_incident(req: SummarizeRequest):
    """Generates structured incident summary formatted for technical triage"""
    summary = engine.generate_incident_summary(req.services, req.symptom, req.likely_cause or "unknown")
    return {"incident_summary": summary}

@app.post("/api/analyze")
async def analyze_text(request: AnalyzeRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="The input text is empty. Please provide valid customer feedback.")
    t0 = time.time()
    result = engine.analyze(request.text)
    latency_ms = round((time.time() - t0) * 1000, 2)
    result["latency_ms"] = latency_ms
    return result

@app.post("/api/batch-csv")
async def batch_analyze_csv(
    file: UploadFile = File(...),
    text_column: str = Form("text"),
    limit: Optional[int] = Form(50)
):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    if text_column not in df.columns:
        candidates = [c for c in df.columns if any(k in c.lower() for k in ['text', 'message', 'tweet', 'content', 'body', 'comment'])]
        text_column = candidates[0] if candidates else df.columns[0]

    limit = min(max(1, limit or 50), 200)
    sample_df = df.head(limit).copy()
    
    records = []
    t0 = time.time()
    for _, row in sample_df.iterrows():
        raw_val = str(row[text_column]) if pd.notna(row[text_column]) else ""
        if not raw_val.strip():
            continue
        res = engine.analyze(raw_val)
        records.append({
            "original_text": raw_val[:220],
            "complaint_label": list(res["complaint"].keys())[0],
            "complaint_confidence": list(res["complaint"].values())[0],
            "sentiment_label": list(res["sentiment"].keys())[0],
            "severity_label": list(res["severity"].keys())[0],
            "service_l1": ", ".join(res["services_L1"]),
            "category_l2": res["L2"],
            "issue_l3": res["L3"],
            "nearest_cluster_id": res["nearest_cluster"]["id"],
            "nearest_cluster_topic": res["nearest_cluster"]["topic"],
            "incident_status": res["nearest_cluster"]["incident_status"],
            "priority_score": res["intelligence"]["priority_score"],
            "priority_level": res["intelligence"]["priority_level"]
        })

    elapsed_time = round(time.time() - t0, 2)
    
    if records:
        res_df = pd.DataFrame(records)
        complaint_counts = res_df["complaint_label"].value_counts().to_dict()
        sentiment_counts = res_df["sentiment_label"].value_counts().to_dict()
        severity_counts = res_df["severity_label"].value_counts().to_dict()
        top_services = res_df["service_l1"].value_counts().head(5).to_dict()
    else:
        complaint_counts, sentiment_counts, severity_counts, top_services = {}, {}, {}, {}

    return {
        "total_rows_file": len(df),
        "analyzed_rows": len(records),
        "text_column_used": text_column,
        "elapsed_seconds": elapsed_time,
        "summary": {
            "complaints": complaint_counts,
            "sentiments": sentiment_counts,
            "severities": severity_counts,
            "top_services": top_services
        },
        "records": records
    }
