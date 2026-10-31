"""Programmatically construct and validate `Bingham_Surrogate_Workflow.ows` using Orange 3."""
from __future__ import annotations

import base64
import os
import pickle
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from AnyQt.QtCore import QItemSelectionModel
from AnyQt.QtTest import QTest
from AnyQt.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from orangecanvas.registry import WidgetRegistry  # noqa: E402
from orangecanvas.scheme import (  # noqa: E402
    SchemeLink,
    SchemeNode,
    SchemeTextAnnotation,
)
from orangewidget.utils.filedialogs import RecentPath  # noqa: E402
from orangewidget.workflow import widgetsscheme  # noqa: E402
from Orange.canvas.config import Config  # noqa: E402
from Orange.widgets.data.owdatasampler import OWDataSampler  # noqa: E402
from Orange.widgets.data.owfeatureconstructor import ContinuousDescriptor  # noqa: E402
from Orange.widgets.evaluate.owtestandscore import OWTestAndScore  # noqa: E402


WORKFLOW_DIR = Path(__file__).resolve().parent
MAIN_CSV = WORKFLOW_DIR / "bingham_orange_ready_40k.csv"
QUERY_CSV = WORKFLOW_DIR / "unseen_rheology_mesh_t05.csv"
OWS_PATH = WORKFLOW_DIR / "Bingham_Surrogate_Workflow.ows"


def wait_idle(scheme: widgetsscheme.WidgetsScheme, timeout_s: float = 180.0) -> None:
    deadline = time.time() + timeout_s
    QTest.qWait(100)
    while time.time() < deadline:
        app.processEvents()
        sm = scheme.signal_manager
        pending = sm.has_pending()
        blocking = [
            node.title
            for node in scheme.nodes
            if scheme.widget_manager.widget_for_node(node).isBlocking()
        ]
        if not pending and not blocking:
            QTest.qWait(150)
            app.processEvents()
            if not sm.has_pending() and not any(
                scheme.widget_manager.widget_for_node(n).isBlocking()
                for n in scheme.nodes
            ):
                return
        QTest.qWait(100)
    raise TimeoutError("Timed out waiting for Orange workflow to finish computing.")


def main() -> None:
    t0 = time.time()
    discovery = Config.widget_discovery(WidgetRegistry())
    discovery.run(Config.widgets_entry_points())
    reg = discovery.registry

    scheme = widgetsscheme.WidgetsScheme()
    scheme.title = "Bingham Fluid CFD — ML Surrogate Workflow"
    scheme.description = (
        "Complete visual ML surrogate workflow for Bingham dam-break CFD simulations "
        "(COMSOL -> Random Forest & Gradient Boosting -> 3 Validation Protocols -> "
        "Error Diagnostics & 2D Contour Prediction)."
    )
    scheme.set_runtime_env("basedir", str(WORKFLOW_DIR))
    scheme.widget_manager.set_creation_policy(scheme.widget_manager.Immediate)

    def add_node(qn: str, title: str, pos: tuple[float, float]) -> tuple[SchemeNode, object]:
        desc = reg.widget(qn)
        node = SchemeNode(desc, title=title, position=pos)
        scheme.add_node(node)
        w = scheme.widget_manager.widget_for_node(node)
        return node, w

    def link(
        src: SchemeNode,
        src_ch: str,
        dst: SchemeNode,
        dst_ch: str,
    ) -> SchemeLink:
        lnk = SchemeLink(src, src_ch, dst, dst_ch)
        scheme.add_link(lnk)
        return lnk

    # --- Column 1: Data Sources ---
    n_file, w_file = add_node(
        "Orange.widgets.data.owfile.OWFile",
        "1. Bingham CFD Dataset (8 Cases)",
        (80.0, 250.0),
    )
    n_query_file, w_query_file = add_node(
        "Orange.widgets.data.owfile.OWFile",
        "5. Unseen Fluid Mesh @ t=0.5s",
        (600.0, 700.0),
    )

    # --- Column 2: Data Inspection & Sampling ---
    n_info, w_info = add_node(
        "Orange.widgets.data.owdatainfo.OWDataInfo",
        "Data Info",
        (250.0, 120.0),
    )
    n_table, w_table = add_node(
        "Orange.widgets.data.owtable.OWTable",
        "CFD Data Table",
        (250.0, 230.0),
    )
    n_cols, w_cols = add_node(
        "Orange.widgets.data.owselectcolumns.OWSelectAttributes",
        "Select Columns (5 Inputs -> Vf)",
        (250.0, 355.0),
    )
    n_sampler, w_sampler = add_node(
        "Orange.widgets.data.owdatasampler.OWDataSampler",
        "Data Sampler (50% Fast Mode)",
        (420.0, 355.0),
    )
    n_rank, w_rank = add_node(
        "Orange.widgets.data.owrank.OWRank",
        "Feature Importance (Rank)",
        (420.0, 150.0),
    )

    # --- Column 3: Learners & Splitters ---
    n_rf, w_rf = add_node(
        "Orange.widgets.model.owrandomforest.OWRandomForest",
        "Random Forest",
        (420.0, 495.0),
    )
    n_gb, w_gb = add_node(
        "Orange.widgets.model.owgradientboosting.OWGradientBoosting",
        "Gradient Boosting",
        (420.0, 610.0),
    )
    n_time_split, w_time_split = add_node(
        "Orange.widgets.data.owselectrows.OWSelectRows",
        "Time Split (time <= 1.05 s)",
        (600.0, 375.0),
    )
    n_case_split, w_case_split = add_node(
        "Orange.widgets.data.owselectrows.OWSelectRows",
        "Holdout Case 4 (tau=20, mu=0.3)",
        (600.0, 545.0),
    )

    # --- Column 4: 3 Evaluation Protocols + Surrogate Predictor ---
    n_eval_rand, w_eval_rand = add_node(
        "Orange.widgets.evaluate.owtestandscore.OWTestAndScore",
        "Eval 1: Random 80/20 (Interpolation)",
        (790.0, 130.0),
    )
    n_eval_loco, w_eval_loco = add_node(
        "Orange.widgets.evaluate.owtestandscore.OWTestAndScore",
        "Eval 2: Leave-One-Case-Out (8 Cases)",
        (790.0, 250.0),
    )
    n_eval_time, w_eval_time = add_node(
        "Orange.widgets.evaluate.owtestandscore.OWTestAndScore",
        "Eval 3: Future-Time (t > 1.05 s)",
        (790.0, 375.0),
    )
    n_rf_surr, w_rf_surr = add_node(
        "Orange.widgets.model.owrandomforest.OWRandomForest",
        "Trained Surrogate (7 Cases)",
        (790.0, 545.0),
    )
    n_pred_query, w_pred_query = add_node(
        "Orange.widgets.evaluate.owpredictions.OWPredictions",
        "Predict Unseen Fluid @ t=0.5s",
        (790.0, 700.0),
    )

    # --- Column 5: Validation Predictions, Model Export & 2D Surrogate Contour ---
    n_save_model, w_save_model = add_node(
        "Orange.widgets.model.owsavemodel.OWSaveModel",
        "Save Surrogate Model",
        (980.0, 465.0),
    )
    n_pred_case4, w_pred_case4 = add_node(
        "Orange.widgets.evaluate.owpredictions.OWPredictions",
        "Case 4 Validation Predictions",
        (980.0, 575.0),
    )
    n_contour_query, w_contour_query = add_node(
        "Orange.widgets.visualize.owscatterplot.OWScatterPlot",
        "2D Contour: Unseen Fluid (t=0.5s)",
        (980.0, 700.0),
    )

    # --- Column 6: Error Formula & Diagnostic Plots ---
    n_actual_pred, w_actual_pred = add_node(
        "Orange.widgets.visualize.owscatterplot.OWScatterPlot",
        "Actual vs Predicted Vf",
        (1165.0, 455.0),
    )
    n_formula, w_formula = add_node(
        "Orange.widgets.data.owfeatureconstructor.OWFeatureConstructor",
        "Formula (abs_error)",
        (1165.0, 575.0),
    )
    n_dist, w_dist = add_node(
        "Orange.widgets.visualize.owdistributions.OWDistributions",
        "Error Distribution",
        (1345.0, 455.0),
    )
    n_err_map, w_err_map = add_node(
        "Orange.widgets.visualize.owscatterplot.OWScatterPlot",
        "2D Interface Error Map (Case 4)",
        (1345.0, 575.0),
    )
    n_groupby, w_groupby = add_node(
        "Orange.widgets.data.owgroupby.OWGroupBy",
        "Mean Error by Time",
        (1345.0, 700.0),
    )
    n_err_time, w_err_time = add_node(
        "Orange.widgets.visualize.owscatterplot.OWScatterPlot",
        "Error vs Time Curve",
        (1510.0, 700.0),
    )

    # --- Configure Fast, Responsive Defaults ---
    w_sampler.sampling_type = OWDataSampler.FixedProportion
    w_sampler.sampleSizePercentage = 50
    w_sampler.use_seed = True
    w_sampler.commit()

    # Use fast O(N log N) Univariate Regression + connected Random Forest scorer (avoid O(N^2) RReliefF)
    w_rank.selected_methods = {"Univariate Regression"}

    w_rf.learner_name = "Random Forest"
    w_rf.n_estimators = 60
    w_rf.use_min_samples_split = True
    w_rf.min_samples_split = 20
    w_rf.use_random_state = True
    w_rf.apply()

    w_gb.learner_name = "Gradient Boosting"
    w_gb.n_estimators = 80
    w_gb.apply()

    w_rf_surr.learner_name = "RF_Surrogate"
    w_rf_surr.n_estimators = 80
    w_rf_surr.use_min_samples_split = True
    w_rf_surr.min_samples_split = 20
    w_rf_surr.use_random_state = True
    w_rf_surr.apply()

    # Configure Evaluation Widget modes BEFORE wiring so each runs only once
    w_eval_rand.resampling = OWTestAndScore.ShuffleSplit
    w_eval_rand.n_repeats = 0  # 2 repeats
    w_eval_rand.sample_size = 12  # 80 %

    w_eval_loco.resampling = OWTestAndScore.FeatureFold
    w_eval_time.resampling = OWTestAndScore.TestOnTest

    # Configure Formula descriptor before wiring
    w_formula.setDescriptors(
        [
            ContinuousDescriptor(
                name="abs_error",
                expression="abs(Vf - RF_Surrogate)",
                number_of_decimals=5,
                meta=False,
            )
        ]
    )

    # Load both CSV files into File widgets
    w_file.add_path(str(MAIN_CSV))
    w_file.load_data()

    w_query_file.add_path(str(QUERY_CSV))
    w_query_file.load_data()

    wait_idle(scheme)

    # --- Wire Stage 1: Data -> Inspection, Sampler, Rank, Splitters ---
    link(n_file, "Data", n_info, "Data")
    link(n_file, "Data", n_table, "Data")
    link(n_file, "Data", n_cols, "Data")
    link(n_cols, "Data", n_sampler, "Data")
    link(n_sampler, "Data Sample", n_rank, "Data")
    link(n_rf, "Learner", n_rank, "Scorer")

    link(n_sampler, "Data Sample", n_time_split, "Data")
    link(n_sampler, "Data Sample", n_case_split, "Data")

    wait_idle(scheme)

    # Configure Select Rows #1: time <= 1.05 s
    w_time_split.remove_all_rows()
    w_time_split.add_row(w_time_split.data.domain["time"], 3, ["1.05"])
    w_time_split.conditions_changed()
    w_time_split.commit.now()

    # Configure Select Rows #2: case_no is not Case_4 (1-based index 4 -> 'Case_4')
    w_case_split.remove_all_rows()
    w_case_split.add_row(w_case_split.data.domain["case_no"], 1, [4])
    w_case_split.conditions_changed()
    w_case_split.commit.now()

    wait_idle(scheme)

    # --- Wire Stage 2A: Send Data to Evaluation Widgets First (0s, no learners yet) ---
    link(n_sampler, "Data Sample", n_eval_rand, "Data")
    link(n_sampler, "Data Sample", n_eval_loco, "Data")
    link(n_time_split, "Matching Data", n_eval_time, "Data")
    link(n_time_split, "Unmatched Data", n_eval_time, "Test Data")

    wait_idle(scheme)

    # Bind exact resampling modes into open domain contexts BEFORE connecting learners
    w_eval_rand.n_repeats = 0  # 2 repeats
    w_eval_rand.sample_size = 12  # 80 %
    w_eval_rand.shuffle_split_changed()

    w_eval_loco.fold_feature = w_eval_loco.data.domain["case_no"]
    w_eval_loco.fold_feature_changed()

    w_eval_time.resampling = OWTestAndScore.TestOnTest
    w_eval_time._param_changed()

    # --- Wire Stage 2B: Connect Learners & Train Surrogate Once ---
    scheme.signal_manager.pause()
    link(n_rf, "Learner", n_eval_rand, "Learner")
    link(n_gb, "Learner", n_eval_rand, "Learner")
    link(n_rf, "Learner", n_eval_loco, "Learner")
    link(n_rf, "Learner", n_eval_time, "Learner")

    # Train surrogate on 7 cases (Case_1..Case_3, Case_5..Case_8) & predict on Case_4 + Query Mesh
    link(n_case_split, "Matching Data", n_rf_surr, "Data")
    link(n_rf_surr, "Model", n_save_model, "Model")

    link(n_rf_surr, "Model", n_pred_case4, "Predictors")
    link(n_case_split, "Unmatched Data", n_pred_case4, "Data")

    link(n_rf_surr, "Model", n_pred_query, "Predictors")
    link(n_query_file, "Data", n_pred_query, "Data")
    scheme.signal_manager.resume()

    wait_idle(scheme)

    # --- Wire Stage 3: Visualizations, Formula & Group By ---
    link(n_pred_query, "Predictions", n_contour_query, "Data")
    link(n_pred_case4, "Predictions", n_actual_pred, "Data")
    link(n_pred_case4, "Predictions", n_formula, "Data")

    wait_idle(scheme)
    w_formula.apply()
    wait_idle(scheme)

    link(n_formula, "Data", n_dist, "Data")
    link(n_formula, "Data", n_err_map, "Data")
    link(n_formula, "Data", n_groupby, "Data")

    wait_idle(scheme)

    # Configure Scatter Plot: 2D Contour on Unseen Fluid (X=x, Y=y, Color=RF_Surrogate)
    d_q = w_contour_query.data.domain
    w_contour_query.attr_x = d_q["x"]
    w_contour_query.attr_y = d_q["y"]
    w_contour_query.attr_color = d_q["RF_Surrogate"]
    w_contour_query.graph.point_width = 6
    w_contour_query.graph.alpha_value = 220

    # Configure Scatter Plot: Actual vs Predicted Vf
    d_ap = w_actual_pred.data.domain
    w_actual_pred.attr_x = d_ap["Vf"]
    w_actual_pred.attr_y = d_ap["RF_Surrogate"]
    w_actual_pred.attr_color = d_ap["time"]
    w_actual_pred.graph.show_reg_line = True
    w_actual_pred.graph.point_width = 5
    w_actual_pred.graph.alpha_value = 140

    # Configure Distributions: abs_error
    d_f = w_dist.data.domain
    w_dist.var = d_f["abs_error"]

    # Configure Scatter Plot: 2D Interface Error Map (X=x, Y=y, Color=abs_error)
    w_err_map.attr_x = d_f["x"]
    w_err_map.attr_y = d_f["y"]
    w_err_map.attr_color = d_f["abs_error"]
    w_err_map.graph.point_width = 6
    w_err_map.graph.alpha_value = 200

    # Configure Group by: group by `time`, aggregate `abs_error` Mean
    gb_view = w_groupby.controls.gb_attrs
    gb_view.clearSelection()
    gb_model = gb_view.model()
    sm = gb_view.selectionModel()
    for row_idx in range(gb_model.rowCount()):
        var = gb_model.data(gb_model.index(row_idx, 0), role=0)
        if str(var) == "time":
            sm.select(gb_model.index(row_idx, 0), QItemSelectionModel.Select)
            break

    wait_idle(scheme)

    link(n_groupby, "Data", n_err_time, "Data")
    wait_idle(scheme)

    if w_err_time.data is not None:
        d_et = w_err_time.data.domain
        if "time" in d_et:
            w_err_time.attr_x = d_et["time"]
        err_mean_candidates = [v for v in d_et.variables if "abs_error" in v.name]
        if err_mean_candidates:
            w_err_time.attr_y = err_mean_candidates[0]
        w_err_time.graph.show_reg_line = False
        w_err_time.graph.point_width = 7

    wait_idle(scheme)

    # --- Add Helpful Canvas Text Annotations ---
    annotations = [
        (
            (30.0, 20.0, 330.0, 68.0),
            "1) DATA & FEATURES\n"
            "Loads 8 Bingham CFD cases (x, y, time, tau_y, mu_p -> Vf).\n"
            "Roles & types are pre-configured automatically.",
        ),
        (
            (370.0, 20.0, 270.0, 68.0),
            "2) FEATURE RANKING & SAMPLING\n"
            "Data Sampler is set to 50% (20k rows) for fast interactive runs.\n"
            "Increase to 100% anytime for full resolution.",
        ),
        (
            (690.0, 20.0, 310.0, 68.0),
            "3) ALL 3 SCIENTIFIC VALIDATION PROTOCOLS\n"
            "- Eval 1: Random 80/20 (interpolation)\n"
            "- Eval 2: 8-Fold Leave-One-Case-Out (by case_no)\n"
            "- Eval 3: Future-Time Extrapolation (t > 1.05 s)",
        ),
        (
            (1060.0, 355.0, 460.0, 68.0),
            "4) SURROGATE DIAGNOSTICS & UNSEEN FLUID CONTOUR\n"
            "Top/Middle: Holdout Case 4 (tau_y=20, mu_p=0.3) predictions & interface error.\n"
            "Bottom: Instant 6,000-node 2D contour @ t=0.5s for unseen fluid (tau_y=12, mu_p=0.15).",
        ),
    ]
    for rect, text in annotations:
        scheme.add_annotation(
            SchemeTextAnnotation(
                rect,
                text,
                content_type="text/plain",
                font={"family": "Segoe UI", "size": 11},
            )
        )

    # --- Verify zero widget errors & print live scores ---
    print(f"=== LIVE WORKFLOW DIAGNOSTICS (elapsed {time.time() - t0:.1f}s) ===")
    for node in scheme.nodes:
        w = scheme.widget_manager.widget_for_node(node)
        errs = [str(m) for m in w.Error.active]
        warns = [str(m) for m in w.Warning.active]
        if errs or warns:
            print(f"  [!] Node [{node.title}]: errors={errs}, warnings={warns}")
        else:
            print(f"  [OK] Node [{node.title}]")

    for label, ev_w in [
        ("Eval 1 (Random 80/20)", w_eval_rand),
        ("Eval 2 (Leave-One-Case-Out)", w_eval_loco),
        ("Eval 3 (Future-Time)", w_eval_time),
    ]:
        model = ev_w.score_table.model
        headers = [model.headerData(c, 1) for c in range(model.columnCount())]
        print(f"--- {label} ---")
        for r in range(model.rowCount()):
            row_vals = [model.data(model.index(r, c)) for c in range(model.columnCount())]
            print(" ", dict(zip(headers, row_vals)))

    print("Case 4 predictions rows:", len(w_pred_case4.data) if w_pred_case4.data else None)
    print("Distributions domain:", w_dist.data.domain if w_dist.data else None)
    print("Groupby output rows:", len(w_err_time.data) if w_err_time.data else None)
    print("Groupby Y attr:", w_err_time.attr_y)
    print("Unseen mesh contour rows:", len(w_contour_query.data) if w_contour_query.data else None)

    # Sync widget settings into node properties & save .ows
    scheme.sync_node_properties()
    with open(OWS_PATH, "wb") as f:
        scheme.save_to(f, pretty=True, pickle_fallback=True)

    # Sanitize pickled RecentPath inside .ows so no Linux `/home/user/...` path is stored
    tree = ET.parse(OWS_PATH)
    root = tree.getroot()
    for prop in root.findall("node_properties/properties"):
        if prop.get("format") == "pickle" and prop.text:
            raw = base64.b64decode(prop.text.encode("ascii"))
            obj = pickle.loads(raw)
            changed = False
            if isinstance(obj, dict):
                ap = obj.get("active_path")
                if isinstance(ap, RecentPath):
                    rel = ap.relpath or os.path.basename(ap.abspath)
                    obj["active_path"] = RecentPath(rel, "basedir", rel, ap.title, ap.sheet, ap.file_format)
                    changed = True
                rps = obj.get("recent_paths")
                if isinstance(rps, list):
                    new_rps = []
                    for rp in rps:
                        if isinstance(rp, RecentPath):
                            rel = rp.relpath or os.path.basename(rp.abspath)
                            new_rps.append(RecentPath(rel, "basedir", rel, rp.title, rp.sheet, rp.file_format))
                            changed = True
                        else:
                            new_rps.append(rp)
                    obj["recent_paths"] = new_rps
            if changed:
                prop.text = base64.encodebytes(pickle.dumps(obj, protocol=4)).decode("ascii")

    tree.write(OWS_PATH, encoding="utf-8", xml_declaration=True)
    print(f"\nSaved portable workflow to {OWS_PATH} ({OWS_PATH.stat().st_size / 1e3:.1f} KB)")

    scheme.clear()
    scheme.deleteLater()
    QTest.qWait(50)


if __name__ == "__main__":
    main()
