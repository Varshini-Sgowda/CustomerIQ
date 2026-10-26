"""
CustomerIQ Flask Web Application
Exposes RESTful APIs and serves the premium, multi-workspace responsive user interface.
"""

import os
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

from customeriq.data_loader import DataLoader
from customeriq.feature_pipeline import FeaturePipeline
from customeriq.ltv_model import LTVModelEngine
from customeriq.profiles import CustomerProfileEngine
from customeriq.copilot import CopilotEngine
from customeriq.anomaly_detector import AnomalyDetector
from customeriq.future_lab import CustomerFutureLab
from customeriq.action_planner import MarketingActionPlanner
from customeriq.insight_engine import InsightEngine
from customeriq.drift_monitor import ModelHealthMonitor
from customeriq.report_generator import ReportGenerator

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Global State Container
class AppContext:
    def __init__(self):
        self.data_loader = DataLoader()
        self.raw_df = self.data_loader.load_data()
        self.feature_pipeline = FeaturePipeline()
        self.features_df = self.feature_pipeline.transform(self.raw_df)

        self.model_engine = LTVModelEngine()
        if not self.model_engine.load_model():
            print("Training production LTV model...")
            self.model_engine.train(self.data_loader)

        self.features_df = self.model_engine.predict_features(self.features_df)

        self.profile_engine = CustomerProfileEngine(self.raw_df, self.features_df, self.model_engine)
        self.copilot_engine = CopilotEngine(self.raw_df, self.features_df, self.model_engine)
        self.anomaly_detector = AnomalyDetector(self.features_df)
        self.future_lab = CustomerFutureLab(self.features_df, self.model_engine)
        self.action_planner = MarketingActionPlanner(self.features_df)
        self.insight_engine = InsightEngine(self.raw_df, self.features_df, self.model_engine, self.data_loader)
        self.health_monitor = ModelHealthMonitor(self.features_df, self.features_df, self.model_engine)
        self.report_generator = ReportGenerator(self.raw_df, self.features_df, self.model_engine, self.data_loader)

CTX = AppContext()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/overview')
def api_overview():
    total_rev = float(round(CTX.features_df['total_monetary'].sum(), 2))
    total_pred = float(round(CTX.features_df['predicted_ltv_12m'].sum(), 2))
    tot_profit = float(round(CTX.features_df['total_profit'].sum(), 2))
    avg_margin = float(round((tot_profit / max(total_rev, 1.0)) * 100, 1))

    seg_counts = CTX.features_df['customer_segment'].value_counts().to_dict()
    recent_alerts = CTX.anomaly_detector.scan_all()

    return jsonify({
        'total_customers': int(len(CTX.features_df)),
        'total_revenue': total_rev,
        'predicted_12m_portfolio': total_pred,
        'total_profit': tot_profit,
        'portfolio_margin_pct': avg_margin,
        'segment_counts': seg_counts,
        'alert_summary': {
            'total': recent_alerts['total_alerts'],
            'critical': recent_alerts['critical_count'],
            'warning': recent_alerts['warning_count'],
            'opportunity': recent_alerts['opportunity_count']
        },
        'metadata': CTX.data_loader.metadata
    })


@app.route('/api/copilot/ask', methods=['POST'])
def api_copilot_ask():
    data = request.get_json() or {}
    query = data.get('query', '')
    if not query.strip():
        return jsonify({'status': 'ERROR', 'message': 'Query cannot be empty.'}), 400
    res = CTX.copilot_engine.ask(query)
    return jsonify(res)


@app.route('/api/customers')
def api_customers():
    search = request.args.get('search', '').strip().lower()
    segment = request.args.get('segment', '').strip()
    sort_by = request.args.get('sort_by', 'predicted_ltv_12m')
    order = request.args.get('order', 'desc')
    limit = min(int(request.args.get('limit', 100)), 793)

    df = CTX.features_df.copy()
    if search:
        df = df[
            df['customer_name'].str.lower().str.contains(search) |
            df['customer_id'].str.lower().str.contains(search)
        ]
    if segment:
        df = df[df['customer_segment'] == segment]

    ascending = (order == 'asc')
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)

    records = df.head(limit).to_dict(orient='records')
    # Filter columns for list view
    clean_records = []
    for r in records:
        clean_records.append({
            'customer_id': r['customer_id'],
            'customer_name': r['customer_name'],
            'segment': r['segment'],
            'customer_segment': r['customer_segment'],
            'total_monetary': r['total_monetary'],
            'predicted_ltv_12m': r['predicted_ltv_12m'],
            'ltv_lower_bound': r['ltv_lower_bound'],
            'ltv_upper_bound': r['ltv_upper_bound'],
            'recency_days': r['recency_days'],
            'frequency': r['frequency'],
            'avg_order_value': r['avg_order_value'],
            'profit_margin': r['profit_margin'],
            'region': r['region']
        })

    return jsonify({
        'total_matching': len(df),
        'returned_count': len(clean_records),
        'customers': clean_records
    })


@app.route('/api/customers/<customer_id>')
def api_customer_profile(customer_id):
    try:
        prof = CTX.profile_engine.get_customer_profile(customer_id)
        return jsonify(prof)
    except KeyError as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/customers/compare')
def api_customers_compare():
    id_a = request.args.get('id_a', '').strip()
    id_b = request.args.get('id_b', '').strip()
    if not id_a or not id_b:
        return jsonify({'error': 'Both id_a and id_b parameters are required.'}), 400
    try:
        comp = CTX.profile_engine.compare_customers(id_a, id_b)
        return jsonify(comp)
    except KeyError as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/alerts')
def api_alerts():
    contamination = float(request.args.get('contamination', 0.05))
    res = CTX.anomaly_detector.scan_all(contamination=contamination)
    return jsonify(res)


@app.route('/api/future-lab/simulate', methods=['POST'])
def api_future_lab_simulate():
    data = request.get_json() or {}
    cust_id = data.get('customer_id')
    changes = data.get('changes', {})
    if not cust_id:
        return jsonify({'error': 'customer_id is required'}), 400
    try:
        res = CTX.future_lab.simulate(cust_id, changes)
        return jsonify(res)
    except KeyError as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/opportunity-center/plan', methods=['POST'])
def api_opportunity_center_plan():
    data = request.get_json() or {}
    budget = float(data.get('budget', 5000.0))
    priority_goal = data.get('priority_goal', 'PREVENT_CHURN_HIGH_LTV')
    cost_overrides = data.get('custom_costs', None)
    res = CTX.action_planner.plan_campaign(
        budget=budget,
        priority_goal=priority_goal,
        custom_cost_overrides=cost_overrides
    )
    return jsonify(res)


@app.route('/api/insights')
def api_insights():
    cards = CTX.insight_engine.generate_all_insights()
    return jsonify({'insights': cards})


@app.route('/api/model-health')
def api_model_health():
    audit = CTX.health_monitor.compute_health_audit()
    return jsonify(audit)


@app.route('/api/reports/download/<format>')
def api_download_report(format):
    format = format.lower()
    if format == 'pdf':
        pdf_path = CTX.report_generator.generate_pdf()
        return send_file(pdf_path, as_attachment=True, download_name=os.path.basename(pdf_path), mimetype='application/pdf')
    elif format == 'excel':
        xlsx_path = CTX.report_generator.generate_excel()
        return send_file(xlsx_path, as_attachment=True, download_name=os.path.basename(xlsx_path), mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    elif format == 'csv':
        csv_path = CTX.report_generator.generate_csv()
        return send_file(csv_path, as_attachment=True, download_name=os.path.basename(csv_path), mimetype='text/csv')
    else:
        return jsonify({'error': f"Unsupported format '{format}'. Supported: pdf, excel, csv"}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port, debug=False)

