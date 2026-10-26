/**
 * CustomerIQ — Core Frontend Controller & Analytics Engine
 * Manages client routing, reactive UI state, Chart.js renderings, and API interactions.
 */

// Global App State
const state = {
    currentView: 'copilot',
    theme: localStorage.getItem('customeriq-theme') || 'dark',
    overview: null,
    customers: [],
    selectedCustomer: null,
    alerts: null,
    currentChartInstances: {},
    activeCompare: { a: null, b: null }
};

// Workspace Titles
const WORKSPACES = {
    'copilot': { title: 'Ask CustomerIQ', subtitle: 'Natural-Language Autonomous Analytics Copilot' },
    'profiles': { title: 'Customer 360 Dossier', subtitle: 'Explainable Profiles with Local SHAP Feature Drivers & Next Steps' },
    'alerts': { title: 'Early-Warning Radar', subtitle: 'Behavioral Drift, Velocity Drop & Isolation Forest Anomaly Detection' },
    'future-lab': { title: 'Customer Future Lab', subtitle: 'Interactive Scenario-Based Digital Twin with Defensible Prediction Intervals' },
    'opportunity-center': { title: 'Opportunity Center', subtitle: 'Intelligent Marketing Decision-Support & Budget Optimization' },
    'insights': { title: 'Automated Insight Feed', subtitle: 'Autonomous Revenue Concentration, Segment Shift & Unit Margin Audits' },
    'model-health': { title: 'Model Health & Drift Monitor', subtitle: 'Population Stability Index (PSI) & Segment Slice Diagnostics' },
    'reports': { title: 'Executive Business Reports', subtitle: 'Publication-Grade PDF Deliverables, Multi-Tab Excel Workbooks & CSVs' }
};

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initRouter();
    loadOverview();
    setupEventListeners();
});

// Theme Management
function initTheme() {
    document.documentElement.setAttribute('data-theme', state.theme);
    const icon = document.getElementById('theme-icon');
    if (icon) icon.textContent = state.theme === 'dark' ? '🌙' : '☀️';

    const toggleBtn = document.getElementById('theme-toggle');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            state.theme = state.theme === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', state.theme);
            localStorage.setItem('customeriq-theme', state.theme);
            if (icon) icon.textContent = state.theme === 'dark' ? '🌙' : '☀️';
            // Re-render any active charts for theme contrast
            Object.values(state.currentChartInstances).forEach(c => c && c.update && c.update());
        });
    }
}

// Router
function initRouter() {
    window.addEventListener('hashchange', handleRoute);
    const initialRoute = window.location.hash.replace('#/', '') || 'copilot';
    navigateTo(initialRoute);
}

function handleRoute() {
    const raw = window.location.hash.replace('#/', '');
    const parts = raw.split('?');
    const route = parts[0] || 'copilot';
    navigateTo(route, parts[1]);
}

function navigateTo(viewName, queryString = '') {
    if (!WORKSPACES[viewName]) viewName = 'copilot';
    state.currentView = viewName;

    // Update nav items
    document.querySelectorAll('.nav-item').forEach(el => {
        el.classList.toggle('active', el.getAttribute('data-view') === viewName);
    });

    // Update view panes
    document.querySelectorAll('.view-pane').forEach(el => {
        el.classList.remove('active');
    });
    const activePane = document.getElementById(`view-${viewName}`);
    if (activePane) activePane.classList.add('active');

    // Update topbar titles
    const meta = WORKSPACES[viewName];
    document.getElementById('current-workspace-title').textContent = meta.title;
    document.getElementById('current-workspace-subtitle').textContent = meta.subtitle;

    // Trigger view-specific loads
    if (viewName === 'profiles') loadProfilesView(queryString);
    else if (viewName === 'alerts') loadAlertsView();
    else if (viewName === 'future-lab') loadFutureLabView();
    else if (viewName === 'opportunity-center') loadOpportunityCenterView();
    else if (viewName === 'insights') loadInsightsView();
    else if (viewName === 'model-health') loadModelHealthView();
}

// Global Overview
async function loadOverview() {
    try {
        const res = await fetch('/api/overview');
        const data = await res.json();
        state.overview = data;

        document.getElementById('topbar-ltv').textContent = `$${Number(data.predicted_12m_portfolio).toLocaleString(undefined, {maximumFractionDigits: 0})}`;
        document.getElementById('topbar-accounts').textContent = data.total_customers.toLocaleString();
        document.getElementById('meta-customer-count').textContent = `${data.total_customers} Verified Accounts`;
        document.getElementById('meta-date-span').textContent = `${data.metadata.date_min} to ${data.metadata.date_max}`;
        document.getElementById('nav-alert-count').textContent = data.alert_summary.critical + data.alert_summary.warning;
    } catch (e) {
        console.error('Failed to load overview:', e);
    }
}

// Event Listeners
function setupEventListeners() {
    // Copilot query pills
    document.querySelectorAll('.query-pill').forEach(btn => {
        btn.addEventListener('click', () => {
            const q = btn.getAttribute('data-q');
            document.getElementById('copilot-input').value = q;
            submitCopilotQuery(q);
        });
    });

    // Copilot form submit
    const copilotForm = document.getElementById('copilot-form');
    if (copilotForm) {
        copilotForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const q = document.getElementById('copilot-input').value.trim();
            if (q) submitCopilotQuery(q);
        });
    }

    // Profiles search and filters
    const searchInput = document.getElementById('customer-search');
    const segSelect = document.getElementById('customer-segment-filter');
    const sortSelect = document.getElementById('customer-sort-filter');

    if (searchInput) searchInput.addEventListener('input', debounce(filterCustomerList, 300));
    if (segSelect) segSelect.addEventListener('change', filterCustomerList);
    if (sortSelect) sortSelect.addEventListener('change', filterCustomerList);

    // Compare modal
    const compareBtn = document.getElementById('btn-open-compare');
    const modal = document.getElementById('compare-modal');
    const modalClose = document.getElementById('modal-close');
    if (compareBtn) compareBtn.addEventListener('click', openCompareModal);
    if (modalClose) modalClose.addEventListener('click', () => { modal.style.display = 'none'; });
    const runCompareBtn = document.getElementById('btn-run-compare');
    if (runCompareBtn) runCompareBtn.addEventListener('click', executeCustomerComparison);

    // Alert filter tabs
    document.querySelectorAll('.alert-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.alert-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            renderAlertCards(tab.getAttribute('data-severity'));
        });
    });

    // Future lab sliders
    setupFutureLabSliders();

    // Opportunity Center
    const optBtn = document.getElementById('btn-optimize-campaign');
    if (optBtn) optBtn.addEventListener('click', runOpportunityOptimization);
}

function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

/* ==========================================================================
   1. Ask CustomerIQ (Copilot) Execution
   ========================================================================== */

async function submitCopilotQuery(query) {
    const area = document.getElementById('copilot-response-area');
    area.innerHTML = `
        <div class="copilot-loading" style="text-align: center; padding: 40px;">
            <div class="empty-icon">⏳</div>
            <p>Executing deterministic query plan against validated feature registry...</p>
        </div>
    `;

    try {
        const res = await fetch('/api/copilot/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        const data = await res.json();
        renderCopilotResponse(data);
    } catch (e) {
        area.innerHTML = `<div class="constraint-error-box">Error executing query: ${e.message}</div>`;
    }
}

function renderCopilotResponse(data) {
    const area = document.getElementById('copilot-response-area');
    const chartId = 'copilot-chart-' + Date.now();

    let tableHtml = '';
    if (data.data_table && data.data_table.length > 0) {
        const headers = Object.keys(data.data_table[0]);
        tableHtml = `
            <div class="table-responsive" style="margin-top: 16px;">
                <table class="data-table">
                    <thead>
                        <tr>${headers.map(h => `<th>${h.replace(/_/g, ' ')}</th>`).join('')}</tr>
                    </thead>
                    <tbody>
                        ${data.data_table.map(row => `
                            <tr>${headers.map(h => `<td>${row[h]}</td>`).join('')}</tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    let followUpsHtml = '';
    if (data.suggested_follow_ups && data.suggested_follow_ups.length > 0) {
        followUpsHtml = `
            <div style="margin-top: 20px;">
                <div style="font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px;">Suggested Follow-Up Analyses</div>
                <div class="suggested-pills">
                    ${data.suggested_follow_ups.map(q => `
                        <button class="query-pill" onclick="submitCopilotQuery('${q.replace(/'/g, "\\'")}')">${q}</button>
                    `).join('')}
                </div>
            </div>
        `;
    }

    area.innerHTML = `
        <div class="copilot-answer-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span class="answer-question-title">${data.question_answered}</span>
                <span class="epistemic-badge ${data.status === 'SUCCESS' ? 'badge-fact' : 'badge-crit'}">${data.status}</span>
            </div>
            
            <div class="answer-explanation">${data.explanation}</div>

            ${data.chart ? `<div class="answer-chart-wrap"><canvas id="${chartId}"></canvas></div>` : ''}

            ${tableHtml}

            <div class="provenance-box">
                <div class="provenance-title">Calculation Provenance & Execution Details</div>
                <div><b>Filters Applied:</b> ${data.calculation_details.filters_applied}</div>
                <div><b>Metric Formula:</b> <code>${data.calculation_details.formula}</code></div>
                <div><b>Source Engine:</b> ${data.calculation_details.provenance}</div>
            </div>

            ${followUpsHtml}
        </div>
    `;

    // Render Chart if provided
    if (data.chart) {
        const ctx = document.getElementById(chartId).getContext('2d');
        renderUniversalChart(ctx, data.chart, chartId);
    }
}

function renderUniversalChart(ctx, chartSpec, chartId) {
    if (state.currentChartInstances[chartId]) {
        state.currentChartInstances[chartId].destroy();
    }

    const isDark = state.theme === 'dark';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.06)';
    const textColor = isDark ? '#94a3b8' : '#475569';

    if (chartSpec.type === 'scatter') {
        state.currentChartInstances[chartId] = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: chartSpec.title,
                    data: chartSpec.data_points,
                    backgroundColor: '#ef4444',
                    borderColor: '#ef4444',
                    pointRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (c) => `${c.raw.name}: ${c.raw.x} days, $${c.raw.y.toLocaleString()}`
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: chartSpec.x_label, color: textColor },
                        grid: { color: gridColor },
                        ticks: { color: textColor }
                    },
                    y: {
                        title: { display: true, text: chartSpec.y_label, color: textColor },
                        grid: { color: gridColor },
                        ticks: { color: textColor }
                    }
                }
            }
        });
    } else {
        state.currentChartInstances[chartId] = new Chart(ctx, {
            type: chartSpec.type || 'bar',
            data: {
                labels: chartSpec.labels,
                datasets: chartSpec.datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: textColor }
                    }
                },
                scales: chartSpec.type === 'pie' ? {} : {
                    x: { grid: { color: gridColor }, ticks: { color: textColor } },
                    y: { grid: { color: gridColor }, ticks: { color: textColor } }
                }
            }
        });
    }
}

/* ==========================================================================
   2. Customer 360 Profiles
   ========================================================================== */

async function loadProfilesView(queryString = '') {
    const listCount = document.getElementById('customer-list-count');
    if (listCount) listCount.textContent = 'Fetching registry...';

    let targetCustId = null;
    if (queryString) {
        const params = new URLSearchParams(queryString);
        targetCustId = params.get('customer_id');
    }

    try {
        const res = await fetch('/api/customers?limit=793');
        const data = await res.json();
        state.customers = data.customers;
        renderCustomerList(data.customers);

        // Select initial customer (first or target from query param)
        if (targetCustId) {
            selectCustomer(targetCustId);
        } else if (data.customers.length > 0) {
            selectCustomer(data.customers[0].customer_id);
        }
    } catch (e) {
        console.error('Failed to load customers:', e);
    }
}

function renderCustomerList(customers) {
    const container = document.getElementById('customer-cards-container');
    const countEl = document.getElementById('customer-list-count');
    if (countEl) countEl.textContent = `${customers.length} accounts`;

    if (!container) return;
    container.innerHTML = customers.map(c => `
        <div class="cust-item-card ${state.selectedCustomer && state.selectedCustomer.customer_id === c.customer_id ? 'active' : ''}" 
             data-id="${c.customer_id}" onclick="selectCustomer('${c.customer_id}')">
            <div class="cust-card-top">
                <span class="cust-card-name">${c.customer_name}</span>
                <span class="cust-card-id">${c.customer_id}</span>
            </div>
            <div class="cust-card-mid">
                <span class="epistemic-badge badge-heuristic">${c.customer_segment}</span>
                <span class="epistemic-badge badge-fact">${c.region}</span>
            </div>
            <div class="cust-card-bottom">
                <span>Spend: $${Number(c.total_monetary).toLocaleString(undefined, {maximumFractionDigits: 0})}</span>
                <span class="cust-card-ltv">12M LTV: $${Number(c.predicted_ltv_12m).toLocaleString(undefined, {maximumFractionDigits: 0})}</span>
            </div>
        </div>
    `).join('');
}

function filterCustomerList() {
    const search = document.getElementById('customer-search').value.toLowerCase().trim();
    const seg = document.getElementById('customer-segment-filter').value;
    const sortVal = document.getElementById('customer-sort-filter').value;
    const [sortCol, sortDir] = sortVal.split(':');

    let filtered = state.customers.filter(c => {
        const matchSearch = !search || c.customer_name.toLowerCase().includes(search) || c.customer_id.toLowerCase().includes(search);
        const matchSeg = !seg || c.customer_segment === seg;
        return matchSearch && matchSeg;
    });

    filtered.sort((a, b) => {
        const valA = a[sortCol];
        const valB = b[sortCol];
        return sortDir === 'asc' ? (valA > valB ? 1 : -1) : (valA < valB ? 1 : -1);
    });

    renderCustomerList(filtered);
}

async function selectCustomer(customerId) {
    // Highlight list card
    document.querySelectorAll('.cust-item-card').forEach(el => {
        el.classList.toggle('active', el.getAttribute('data-id') === customerId);
    });

    const container = document.getElementById('customer-dossier-container');
    container.innerHTML = `<div style="text-align: center; padding: 40px;"><p>Loading explainable customer profile...</p></div>`;

    try {
        const res = await fetch(`/api/customers/${customerId}`);
        const profile = await res.json();
        state.selectedCustomer = profile;
        renderCustomerDossier(profile);
    } catch (e) {
        container.innerHTML = `<div class="constraint-error-box">Failed to load profile: ${e.message}</div>`;
    }
}

function renderCustomerDossier(p) {
    const container = document.getElementById('customer-dossier-container');
    const obs = p.observed_metrics;
    const pred = p.model_predictions;
    const rfm = p.rfm_classification;
    const meta = p.metadata;

    const trendChartId = 'dossier-trend-' + Date.now();
    const catChartId = 'dossier-cat-' + Date.now();

    // SHAP bars calculation
    const maxImpact = Math.max(...pred.all_shap_contributions.map(c => c.impact_magnitude), 1.0);
    const shapRowsHtml = pred.all_shap_contributions.slice(0, 6).map(c => {
        const pct = Math.min((c.impact_magnitude / maxImpact) * 100, 100);
        return `
            <div class="shap-row">
                <span class="shap-feat-name" title="${c.feature}">${c.feature.replace(/_/g, ' ')}</span>
                <div class="shap-bar-wrap">
                    <div class="shap-bar ${c.direction}" style="width: ${pct}%;"></div>
                </div>
                <span class="shap-val-text ${c.direction}">${c.shap_value > 0 ? '+' : ''}$${c.shap_value.toFixed(0)}</span>
            </div>
        `;
    }).join('');

    // Warnings HTML
    const warningsHtml = p.data_quality_warnings.length > 0 ? p.data_quality_warnings.map(w => `
        <div class="warning-callout">
            <b>${w.type}:</b> ${w.message}
        </div>
    `).join('') : `<p style="font-size: 12px; color: var(--accent-emerald);">✓ Zero data quality warnings. Account history is sound.</p>`;

    // Next steps HTML
    const nextStepsHtml = p.recommended_next_steps.map(s => `
        <div class="action-callout">
            <div style="font-weight: 700; margin-bottom: 2px;">${s.action} <span class="epistemic-badge badge-heuristic">${s.priority}</span></div>
            <div style="color: var(--text-secondary);">${s.rational}</div>
        </div>
    `).join('');

    container.innerHTML = `
        <div class="dossier-header-bar">
            <div>
                <h2 class="dossier-name">${meta.customer_name}</h2>
                <div class="dossier-id-sub">Customer ID: ${meta.customer_id} &bull; Region: ${meta.region} &bull; Type: ${meta.segment}</div>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
                <span class="epistemic-badge badge-fact">OBSERVED RECORD</span>
                <span class="epistemic-badge badge-heuristic">RFM: ${rfm.rfm_code} (${rfm.customer_segment})</span>
            </div>
        </div>

        <div style="background: var(--bg-surface); padding: 10px 14px; border-radius: 8px; border: 1px solid var(--border-color); font-size: 12px; margin-bottom: 18px;">
            <b>Segment Classification Rationale:</b> ${rfm.rule_explanation}
        </div>

        <!-- Metric Grid -->
        <div class="dossier-metrics-grid">
            <div class="dossier-metric-box">
                <span class="epistemic-badge badge-estimate">PREDICTED LTV</span>
                <div class="dossier-metric-val" style="color: var(--accent-blue);">$${pred.predicted_ltv_12m.toLocaleString()}</div>
                <div class="dossier-metric-label">80% CI: [$${pred.lower_bound_80} - $${pred.upper_bound_80}]</div>
            </div>
            <div class="dossier-metric-box">
                <span class="epistemic-badge badge-fact">LIFETIME SPEND</span>
                <div class="dossier-metric-val">$${obs.total_monetary.toLocaleString()}</div>
                <div class="dossier-metric-label">${obs.frequency} orders &bull; AOV $${obs.avg_order_value}</div>
            </div>
            <div class="dossier-metric-box">
                <span class="epistemic-badge badge-fact">RECENCY</span>
                <div class="dossier-metric-val" style="color: ${obs.recency_days > 180 ? 'var(--accent-rose)' : 'var(--text-primary)'};">${obs.recency_days} Days</div>
                <div class="dossier-metric-label">Tenure: ${obs.tenure_days} days</div>
            </div>
            <div class="dossier-metric-box">
                <span class="epistemic-badge badge-fact">PROFIT MARGIN</span>
                <div class="dossier-metric-val" style="color: ${obs.profit_margin < 0 ? 'var(--accent-rose)' : 'var(--accent-emerald)'};">${obs.profit_margin.toFixed(1)}%</div>
                <div class="dossier-metric-label">Avg Discount: ${(obs.avg_discount * 100).toFixed(1)}%</div>
            </div>
        </div>

        <!-- Charts Grid: SHAP Drivers + Spending Trend -->
        <div class="dossier-charts-grid">
            <div class="dossier-chart-card">
                <h4>
                    <span>SHAP Model Feature Contributions</span>
                    <span class="epistemic-badge badge-estimate">TreeExplainer</span>
                </h4>
                <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 10px;">
                    Baseline expectation: $${pred.shap_base_value.toFixed(0)}. Green pushes prediction higher, red pulls it lower.
                </div>
                <div class="shap-container">
                    ${shapRowsHtml}
                </div>
            </div>

            <div class="dossier-chart-card">
                <h4>
                    <span>Historical Quarterly Spend</span>
                    <span class="epistemic-badge badge-fact">Observed Trend</span>
                </h4>
                <div style="height: 180px; position: relative;">
                    <canvas id="${trendChartId}"></canvas>
                </div>
            </div>
        </div>

        <!-- Behavioral Momentum & Notes -->
        <div class="dossier-notes-grid">
            <div class="dossier-chart-card">
                <h4>Behavioral Change Audit</h4>
                <div style="font-size: 12px; margin-bottom: 8px;">
                    <b>Order Velocity:</b> ${p.behavioral_momentum.order_shift_description} (${p.behavioral_momentum.order_momentum_pct > 0 ? '+' : ''}${p.behavioral_momentum.order_momentum_pct}%)
                </div>
                <div style="font-size: 12px; margin-bottom: 12px;">
                    <b>Spending Shift:</b> ${p.behavioral_momentum.spend_shift_description}
                </div>
                <h4>Data Quality & Warnings</h4>
                ${warningsHtml}
            </div>

            <div class="dossier-chart-card">
                <h4>Recommended Next Steps</h4>
                ${nextStepsHtml}
            </div>
        </div>
    `;

    // Render Quarterly Trend Chart
    const trendCtx = document.getElementById(trendChartId).getContext('2d');
    const isDark = state.theme === 'dark';
    new Chart(trendCtx, {
        type: 'line',
        data: {
            labels: p.quarterly_trend.map(q => q.quarter),
            datasets: [{
                label: 'Quarterly Sales ($)',
                data: p.quarterly_trend.map(q => q.sales),
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.15)',
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.05)' } },
                y: { grid: { color: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.05)' } }
            }
        }
    });
}

/* ==========================================================================
   Compare 2 Customers Modal
   ========================================================================== */

function openCompareModal() {
    const modal = document.getElementById('compare-modal');
    modal.style.display = 'flex';

    const selA = document.getElementById('compare-cust-a');
    const selB = document.getElementById('compare-cust-b');

    const optionsHtml = state.customers.map(c => `
        <option value="${c.customer_id}">${c.customer_name} (${c.customer_id} - ${c.customer_segment})</option>
    `).join('');

    selA.innerHTML = optionsHtml;
    selB.innerHTML = optionsHtml;

    if (state.customers.length >= 2) {
        selA.selectedIndex = 0;
        selB.selectedIndex = 1;
        executeCustomerComparison();
    }
}

async function executeCustomerComparison() {
    const idA = document.getElementById('compare-cust-a').value;
    const idB = document.getElementById('compare-cust-b').value;
    const container = document.getElementById('compare-result-container');

    container.innerHTML = `<p style="padding: 20px; text-align: center;">Calculating comparative variance...</p>`;

    try {
        const res = await fetch(`/api/customers/compare?id_a=${idA}&id_b=${idB}`);
        const comp = await res.json();

        const a = comp.customer_a;
        const b = comp.customer_b;
        const diff = comp.variance_summary;

        container.innerHTML = `
            <div class="table-responsive" style="margin-top: 14px;">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Analytical Attribute</th>
                            <th>${a.metadata.customer_name} (${a.metadata.customer_id})</th>
                            <th>${b.metadata.customer_name} (${b.metadata.customer_id})</th>
                            <th>Comparative Variance (A - B)</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><b>RFM Segment</b></td>
                            <td><span class="epistemic-badge badge-heuristic">${a.rfm_classification.customer_segment}</span></td>
                            <td><span class="epistemic-badge badge-heuristic">${b.rfm_classification.customer_segment}</span></td>
                            <td>Segment Divergence</td>
                        </tr>
                        <tr>
                            <td><b>Predicted 12M LTV</b></td>
                            <td><b>$${a.model_predictions.predicted_ltv_12m.toLocaleString()}</b></td>
                            <td><b>$${b.model_predictions.predicted_ltv_12m.toLocaleString()}</b></td>
                            <td style="color: ${diff.predicted_ltv_diff >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)'}; font-weight: 700;">
                                ${diff.predicted_ltv_diff >= 0 ? '+' : ''}$${diff.predicted_ltv_diff.toLocaleString()}
                            </td>
                        </tr>
                        <tr>
                            <td><b>Lifetime Spend</b></td>
                            <td>$${a.observed_metrics.total_monetary.toLocaleString()}</td>
                            <td>$${b.observed_metrics.total_monetary.toLocaleString()}</td>
                            <td>${diff.total_spend_diff >= 0 ? '+' : ''}$${diff.total_spend_diff.toLocaleString()}</td>
                        </tr>
                        <tr>
                            <td><b>Order Frequency</b></td>
                            <td>${a.observed_metrics.frequency} orders</td>
                            <td>${b.observed_metrics.frequency} orders</td>
                            <td>${diff.frequency_diff >= 0 ? '+' : ''}${diff.frequency_diff} orders</td>
                        </tr>
                        <tr>
                            <td><b>Recency (Days Inactive)</b></td>
                            <td>${a.observed_metrics.recency_days} days</td>
                            <td>${b.observed_metrics.recency_days} days</td>
                            <td>${diff.recency_diff_days} days</td>
                        </tr>
                        <tr>
                            <td><b>Average Order Value</b></td>
                            <td>$${a.observed_metrics.avg_order_value}</td>
                            <td>$${b.observed_metrics.avg_order_value}</td>
                            <td>${diff.aov_diff >= 0 ? '+' : ''}$${diff.aov_diff}</td>
                        </tr>
                        <tr>
                            <td><b>Profit Margin %</b></td>
                            <td>${a.observed_metrics.profit_margin.toFixed(1)}%</td>
                            <td>${b.observed_metrics.profit_margin.toFixed(1)}%</td>
                            <td>${diff.margin_diff_pct >= 0 ? '+' : ''}${diff.margin_diff_pct}%</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        `;
    } catch (e) {
        container.innerHTML = `<div class="constraint-error-box">Comparison error: ${e.message}</div>`;
    }
}

/* ==========================================================================
   3. Early-Warning Radar
   ========================================================================== */

async function loadAlertsView() {
    const grid = document.getElementById('alerts-grid-container');
    grid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; padding: 40px;"><p>Scanning feature tables and executing Isolation Forest...</p></div>`;

    try {
        const res = await fetch('/api/alerts');
        const data = await res.json();
        state.alerts = data;

        document.getElementById('alert-crit-count').textContent = data.critical_count;
        document.getElementById('alert-warn-count').textContent = data.warning_count;
        document.getElementById('alert-opp-count').textContent = data.opportunity_count;
        document.getElementById('alert-iso-count').textContent = data.info_count;

        renderAlertCards('ALL');
    } catch (e) {
        grid.innerHTML = `<div class="constraint-error-box">Error loading alerts: ${e.message}</div>`;
    }
}

function renderAlertCards(severityFilter = 'ALL') {
    const grid = document.getElementById('alerts-grid-container');
    if (!state.alerts) return;

    let list = state.alerts.alerts;
    if (severityFilter !== 'ALL') {
        list = list.filter(a => a.severity === severityFilter);
    }

    grid.innerHTML = list.map(a => {
        const badgeClass = a.severity === 'CRITICAL' ? 'badge-crit' : (a.severity === 'WARNING' ? 'badge-warn' : (a.severity === 'OPPORTUNITY' ? 'badge-opp' : 'badge-info'));
        return `
            <div class="alert-item-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="alert-badge ${badgeClass}">${a.severity}</span>
                    <span style="font-size: 11px; color: var(--text-muted); font-family: var(--font-mono);">${a.customer_id}</span>
                </div>
                <h4 class="alert-card-title">${a.title}</h4>
                <div class="alert-evidence">${a.evidence}</div>
                <div style="font-size: 11.5px; margin-bottom: 6px;">
                    <b>Recommended Investigation:</b> ${a.recommended_investigation}
                </div>
                <div class="alert-meta-row">
                    <div><b>Detection Method:</b> ${a.detection_method}</div>
                    <div style="margin-top: 6px;">
                        <a href="${a.action_url}" style="color: var(--accent-blue); text-decoration: none; font-weight: 600;">Inspect Customer Profile →</a>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

/* ==========================================================================
   4. Customer Future Lab (Digital Twin)
   ========================================================================== */

function loadFutureLabView() {
    const custSelect = document.getElementById('lab-customer-select');
    if (custSelect && state.customers.length > 0) {
        custSelect.innerHTML = state.customers.map(c => `
            <option value="${c.customer_id}">${c.customer_name} (${c.customer_id} - ${c.customer_segment})</option>
        `).join('');
        syncSlidersToCustomer(state.customers[0].customer_id);
    }
}

function setupFutureLabSliders() {
    const sliders = [
        { id: 'lab-recency', valId: 'lab-recency-val', unit: ' d' },
        { id: 'lab-frequency', valId: 'lab-frequency-val', unit: ' orders' },
        { id: 'lab-aov', valId: 'lab-aov-val', unit: '', prefix: '$' },
        { id: 'lab-discount', valId: 'lab-discount-val', unit: '%' },
        { id: 'lab-recent-orders', valId: 'lab-recent-orders-val', unit: ' orders' }
    ];

    sliders.forEach(s => {
        const input = document.getElementById(s.id);
        const valEl = document.getElementById(s.valId);
        if (input && valEl) {
            input.addEventListener('input', () => {
                const prefix = s.prefix || '';
                valEl.textContent = `${prefix}${input.value}${s.unit}`;
            });
        }
    });

    const custSelect = document.getElementById('lab-customer-select');
    if (custSelect) {
        custSelect.addEventListener('change', () => {
            syncSlidersToCustomer(custSelect.value);
        });
    }

    const runBtn = document.getElementById('btn-run-simulation');
    if (runBtn) runBtn.addEventListener('click', executeFutureLabSimulation);

    const resetBtn = document.getElementById('btn-reset-simulation');
    if (resetBtn) resetBtn.addEventListener('click', () => {
        if (custSelect) syncSlidersToCustomer(custSelect.value);
    });
}

function syncSlidersToCustomer(customerId) {
    const c = state.customers.find(item => item.customer_id === customerId);
    if (!c) return;

    document.getElementById('lab-recency').value = c.recency_days;
    document.getElementById('lab-recency-val').textContent = `${c.recency_days} d`;

    document.getElementById('lab-frequency').value = c.frequency;
    document.getElementById('lab-frequency-val').textContent = `${c.frequency} orders`;

    document.getElementById('lab-aov').value = Math.round(c.avg_order_value);
    document.getElementById('lab-aov-val').textContent = `$${Math.round(c.avg_order_value)}`;

    const discPct = Math.round((c.avg_discount || 0.15) * 100);
    document.getElementById('lab-discount').value = discPct;
    document.getElementById('lab-discount-val').textContent = `${discPct}%`;

    executeFutureLabSimulation();
}

async function executeFutureLabSimulation() {
    const custId = document.getElementById('lab-customer-select').value;
    const errBox = document.getElementById('lab-error-box');
    const resultsContent = document.getElementById('lab-results-content');
    errBox.style.display = 'none';

    const changes = {
        recency_days: Number(document.getElementById('lab-recency').value),
        frequency: Number(document.getElementById('lab-frequency').value),
        avg_order_value: Number(document.getElementById('lab-aov').value),
        avg_discount: Number(document.getElementById('lab-discount').value) / 100.0,
        recent_orders_6m: Number(document.getElementById('lab-recent-orders').value)
    };

    try {
        const res = await fetch('/api/future-lab/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ customer_id: custId, changes })
        });
        const sim = await res.json();

        if (sim.status === 'INVALID_INPUT') {
            errBox.innerHTML = `<b>Constraint Validation Errors:</b><br/>${sim.errors.join('<br/>')}`;
            errBox.style.display = 'block';
            return;
        }

        renderSimulationResults(sim);
    } catch (e) {
        errBox.textContent = `Simulation error: ${e.message}`;
        errBox.style.display = 'block';
    }
}

function renderSimulationResults(sim) {
    const resultsContent = document.getElementById('lab-results-content');
    const delta = sim.variance.delta_ltv;
    const deltaPct = sim.variance.delta_ltv_pct;
    const color = delta > 0 ? 'var(--accent-emerald)' : (delta < 0 ? 'var(--accent-rose)' : 'var(--text-secondary)');

    resultsContent.innerHTML = `
        <div class="comparison-kpi-grid">
            <div class="comp-kpi-card">
                <span class="comp-kpi-title">Baseline Expected LTV</span>
                <div class="comp-kpi-val">$${sim.baseline.predicted_ltv.toLocaleString()}</div>
                <div class="comp-kpi-bound">80% Bound: [$${sim.baseline.prediction_interval_80[0]} - $${sim.baseline.prediction_interval_80[1]}]</div>
            </div>
            <div class="comp-kpi-card">
                <span class="comp-kpi-title">Simulated Scenario LTV</span>
                <div class="comp-kpi-val" style="color: ${color};">$${sim.simulation.predicted_ltv.toLocaleString()}</div>
                <div class="comp-kpi-bound">80% Bound: [$${sim.simulation.prediction_interval_80[0]} - $${sim.simulation.prediction_interval_80[1]}]</div>
            </div>
        </div>

        <div style="background: var(--bg-surface); padding: 12px 16px; border-radius: 8px; border: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <span style="font-size: 13px; font-weight: 700;">Simulated Variance:</span>
            <span style="font-size: 14px; font-weight: 800; color: ${color};">
                ${delta > 0 ? '+' : ''}$${delta.toLocaleString()} (${deltaPct > 0 ? '+' : ''}${deltaPct}%)
            </span>
        </div>

        <div class="table-responsive">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Feature Parameter</th>
                        <th>Baseline</th>
                        <th>Simulated State</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    ${sim.feature_comparison.map(f => `
                        <tr>
                            <td><b>${f.label}</b></td>
                            <td>${f.baseline}</td>
                            <td><b>${f.simulated}</b></td>
                            <td><span class="epistemic-badge ${f.status === 'MODIFIED' ? 'badge-sim' : 'badge-fact'}">${f.status}</span></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>

        <div class="disclaimer-banner">
            ⚠️ <b>Causal Inference Note:</b> ${sim.causal_disclaimer}
        </div>
    `;
}

/* ==========================================================================
   5. Opportunity Center
   ========================================================================== */

function loadOpportunityCenterView() {
    runOpportunityOptimization();
}

async function runOpportunityOptimization() {
    const budget = Number(document.getElementById('opp-budget').value) || 5000;
    const goal = document.getElementById('opp-goal').value;

    try {
        const res = await fetch('/api/opportunity-center/plan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ budget, priority_goal: goal })
        });
        const plan = await res.json();
        renderOpportunityPlan(plan);
    } catch (e) {
        console.error('Failed to optimize campaign:', e);
    }
}

function renderOpportunityPlan(plan) {
    const s = plan.summary;
    const b = plan.comparison_with_baseline;

    document.getElementById('opp-targeted-count').textContent = `${s.targeted_customers_count} accounts`;
    document.getElementById('opp-budget-used').textContent = `$${s.budget_utilized.toLocaleString()}`;
    document.getElementById('opp-incr-rev').textContent = `$${s.estimated_incremental_revenue.toLocaleString()}`;
    document.getElementById('opp-roi').textContent = `${s.expected_portfolio_roi}x ROI`;
    document.getElementById('opp-active-strategy').textContent = plan.parameters.priority_goal;

    // Benchmark card
    const benchEl = document.getElementById('opp-benchmark-content');
    benchEl.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 13px; font-weight: 700;">Strategy Benchmark vs Baseline</span>
            <span class="epistemic-badge badge-heuristic">Uplift Elasticity Model</span>
        </div>
        <p style="font-size: 12.5px; line-height: 1.5; color: var(--text-secondary);">${b.narrative}</p>
        <div style="font-size: 11px; color: var(--text-muted); margin-top: 8px;">${plan.uplift_disclaimer}</div>
    `;

    // Action Table
    const tbody = document.getElementById('opp-action-table-body');
    tbody.innerHTML = plan.targeted_customers.map((c, i) => `
        <tr>
            <td>#${i + 1}</td>
            <td><b>${c.customer_name}</b><br/><span style="font-size: 10px; color: var(--text-muted);">${c.customer_id}</span></td>
            <td><span class="epistemic-badge badge-heuristic">${c.segment}</span></td>
            <td>$${c.predicted_ltv_12m.toLocaleString()}</td>
            <td><b>${c.action_name}</b><br/><span style="font-size: 10px; color: var(--text-muted);">${c.action_reason}</span></td>
            <td>$${c.cost.toFixed(0)}</td>
            <td style="color: var(--accent-emerald); font-weight: 700;">+$${c.estimated_incremental_value.toFixed(0)}</td>
            <td style="color: var(--accent-blue); font-weight: 700;">${c.expected_net_roi.toFixed(1)}x</td>
        </tr>
    `).join('');
}

/* ==========================================================================
   6. Automated Insight Feed
   ========================================================================== */

async function loadInsightsView() {
    const grid = document.getElementById('insights-grid-container');
    grid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; padding: 40px;"><p>Computing autonomous analytical insights across portfolio...</p></div>`;

    try {
        const res = await fetch('/api/insights');
        const data = await res.json();
        renderInsightCards(data.insights);
    } catch (e) {
        grid.innerHTML = `<div class="constraint-error-box">Error loading insights: ${e.message}</div>`;
    }
}

function renderInsightCards(insights) {
    const grid = document.getElementById('insights-grid-container');
    grid.innerHTML = insights.map(c => {
        const chipsHtml = Object.entries(c.supporting_metric).map(([k, v]) => `
            <div class="metric-chip">
                <span class="chip-label">${k.replace(/_/g, ' ')}</span>
                <span class="chip-val">${v}</span>
            </div>
        `).join('');

        return `
            <div class="insight-card">
                <div class="insight-cat-tag">${c.category.replace(/_/g, ' ')}</div>
                <h3 class="insight-title">${c.title}</h3>
                
                <div class="insight-metrics-row">
                    ${chipsHtml}
                </div>

                <div class="insight-hypothesis-box">
                    ${c.possible_interpretation}
                </div>

                <div class="insight-action-box">
                    ${c.recommended_next_step}
                </div>

                <div class="insight-footer-meta">
                    <div><b>Population:</b> ${c.relevant_population}</div>
                    <div><b>Baseline:</b> ${c.comparison_baseline}</div>
                    <div><b>Method:</b> ${c.detection_method}</div>
                    <div style="margin-top: 4px; color: var(--accent-blue);"><b>Evidence:</b> ${c.evidence_qualification}</div>
                </div>
            </div>
        `;
    }).join('');
}

/* ==========================================================================
   7. Model Health & Drift Monitor
   ========================================================================== */

async function loadModelHealthView() {
    try {
        const res = await fetch('/api/model-health');
        const audit = await res.json();
        renderModelHealthAudit(audit);
    } catch (e) {
        console.error('Failed to load model health:', e);
    }
}

function renderModelHealthAudit(audit) {
    const statusEl = document.getElementById('health-main-status');
    statusEl.textContent = audit.overall_status;
    statusEl.className = `status-indicator-badge ${audit.overall_status === 'HEALTHY' ? 'status-healthy' : 'badge-warn'}`;

    document.getElementById('health-status-desc').textContent = audit.status_summary;
    document.getElementById('h-model-algo').textContent = audit.model_metadata.algorithm || 'Gradient Boosting';
    document.getElementById('h-model-mae').textContent = `$${audit.model_metadata.cv_mean_mae || 965.44}`;

    // PSI Table
    const psiTbody = document.getElementById('psi-table-body');
    psiTbody.innerHTML = audit.feature_drift_psi.features.map(f => {
        const dotClass = f.status === 'STABLE' ? 'dot-green' : (f.status === 'MODERATE_DRIFT' ? 'dot-amber' : 'dot-red');
        return `
            <tr>
                <td><b>${f.feature}</b></td>
                <td style="font-family: var(--font-mono); font-weight: 700;">${f.psi.toFixed(4)}</td>
                <td><span class="dot ${dotClass}"></span> ${f.status}</td>
                <td>${f.baseline_mean}</td>
                <td>${f.current_mean}</td>
                <td>${f.relative_mean_shift_pct > 0 ? '+' : ''}${f.relative_mean_shift_pct}%</td>
            </tr>
        `;
    }).join('');

    // Missingness
    const missingEl = document.getElementById('missingness-content');
    missingEl.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span>Overall Missingness:</span>
            <b>${audit.missingness_audit.overall_missing_percentage}% (Status: ${audit.missingness_audit.status})</b>
        </div>
        <p style="font-size: 11.5px; color: var(--text-muted);">${audit.missingness_audit.total_rows_inspected} rows verified against required schema.</p>
    `;

    // Slices
    const slicesEl = document.getElementById('slices-content');
    slicesEl.innerHTML = `
        <div class="table-responsive">
            <table class="data-table">
                <thead>
                    <tr><th>Segment Slice</th><th>Accounts</th><th>Mean LTV</th><th>Historical Margin</th></tr>
                </thead>
                <tbody>
                    ${audit.segment_slices.map(s => `
                        <tr>
                            <td><b>${s.segment}</b></td>
                            <td>${s.customer_count}</td>
                            <td>$${s.mean_predicted_ltv.toLocaleString()}</td>
                            <td>${s.mean_historical_margin}%</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
        <div style="font-size: 11px; color: var(--text-muted); margin-top: 10px;">${audit.drift_explanation}</div>
    `;
}

