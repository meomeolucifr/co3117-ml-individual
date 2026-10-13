"""Generates report/part1_summary.pdf from the real Part I results already committed
in this repository (MODEL_LOG.md, results/*.csv, results/figures/*.png).

This is a DRAFT generator: the prose is grounded in real numbers but should be
reviewed/personalized before being treated as final submission evidence, consistent
with every other weekly post in this repo (Section 12.3).

Run: python report/build_part1_summary.py
"""
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
OUT = ROOT / "report" / "part1_summary.pdf"


class Report(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 8, f"CO3117 Part I summary - Nguyen Thien Nhan - 2452879 - page {self.page_no()}", align="C")

    def h1(self, text):
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(20, 40, 90)
        self.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def h2(self, text):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 30, 30)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

    def body(self, text):
        self.set_font("Helvetica", "", 10.5)
        self.multi_cell(0, 5.2, text)
        self.ln(1)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10.5)
        self.multi_cell(0, 5.2, f"- {text}")

    def table(self, headers, rows, widths):
        self.set_font("Helvetica", "B", 9.5)
        for h, w in zip(headers, widths):
            self.cell(w, 6.5, h, border=1)
        self.ln()
        self.set_font("Helvetica", "", 9.5)
        for row in rows:
            for cell, w in zip(row, widths):
                self.cell(w, 6.5, str(cell), border=1)
            self.ln()
        self.ln(2)

    def figure(self, path, caption, width=130):
        if path.exists():
            x = (210 - width) / 2
            self.image(str(path), x=x, w=width)
            self.set_font("Helvetica", "I", 8.5)
            self.cell(0, 5, caption, align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)


def build():
    pdf = Report(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(18, 15, 18)

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "CO3117 - Part I Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "Nguyen Thien Nhan - 2452879 - HK261", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # 1. Problem & dataset
    pdf.h1("1. Problem and dataset")
    pdf.body(
        "Use case: predict a person's current physical activity (WALKING, "
        "WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) from "
        "smartphone accelerometer/gyroscope features. Dataset: UCI Human Activity "
        "Recognition Using Smartphones, original 2012 release (DOI 10.24432/C54S4K), "
        "frozen in protocol.yaml together with the archive SHA-256. Target: the 6-class "
        "activity label. Primary metric: Macro-F1 (accuracy and confusion matrix as "
        "secondary evidence). Leakage control: the UCI-provided train/test split is "
        "subject-disjoint (21 train subjects, 9 test subjects, no overlap, verified in "
        "src/data.py tests); the sealed test population is never touched before Part II; "
        "every validation/tuning split in this report is also subject-aware "
        "(GroupShuffleSplit on subject_train.txt), never a random row split."
    )

    # 2. Representations & protocol
    pdf.h1("2. Representations and protocol")
    pdf.body(
        "All models in this report use the same frozen 561-feature representation "
        "(UCI's precomputed time/frequency-domain features), no additional feature "
        "engineering. Every experiment uses the same fit/validation split "
        "(GroupShuffleSplit, test_size=0.25, random_state=2452879) carved out of the "
        "training population only. Seed 2452879 (student ID) is used consistently "
        "across all experiments per protocol.yaml. Environment: Python 3.13, "
        "scikit-learn, numpy, see requirements.txt; each experiment script under "
        "experiments/part1_pre_midterm/ is self-contained and reproducible by running "
        "it directly."
    )

    # 3. Model-family map
    pdf.h1("3. Model-family map (Part I)")
    pdf.table(
        ["Model", "Ch.", "Depth", "Code / own implementation"],
        [
            ["Perceptron (+ one-vs-rest)", "3", "A", "src/from_scratch/perceptron.py"],
            ["MLP (forward+backward)", "3", "A", "src/from_scratch/mlp.py"],
            ["Decision Tree split routine", "2", "B", "src/from_scratch/tree_split.py"],
            ["Naive Bayes (Gaussian+Multinomial)", "4", "A", "src/from_scratch/naive_bayes.py"],
            ["Genetic Algorithm (feature selection)", "5", "B", "src/from_scratch/genetic_algorithm.py"],
            ["Bayesian Network / TAN (theory)", "6 (Part I)", "C", "docs/weekly/w07-genetic-algorithm.md"],
        ],
        [78, 14, 14, 76],
    )
    pdf.body(
        "All Depth-A components pass their contract tests (tests/contract/) and match "
        "a trusted reference (sklearn) to tight numerical tolerance where applicable "
        "(Perceptron convergence test, MLP numerical-gradient check < 1e-5 relative "
        "error, Naive Bayes predict_log_proba matched to sklearn within 1e-4 to 1e-6). "
        "MLP is gradient-verified on synthetic data via the contract test but has not "
        "yet been trained end-to-end on HAR in this Part I window - noted as a known "
        "limitation below rather than claimed as benchmarked."
    )

    pdf.add_page()

    # 4. Results
    pdf.h1("4. Results")
    pdf.body(
        "Master comparison on the frozen validation split (subject-aware, seed "
        "2452879), Macro-F1 unless noted:"
    )
    pdf.table(
        ["Model", "Macro-F1", "Accuracy", "Notes"],
        [
            ["Perceptron (own, one-vs-rest)", "0.9633", "0.9636", "linear baseline"],
            ["Perceptron (sklearn)", "0.9734", "0.9738", "reference"],
            ["Decision Tree (sklearn, depth=8)", "0.858", "-", "catchup benchmark, best depth found"],
            ["Naive Bayes (own, var_smoothing=1e-9)", "0.7679", "0.7752", "matches sklearn 100%"],
            ["Naive Bayes (own, var_smoothing=1e-3)", "0.8411", "-", "tuned, best of sweep"],
            ["GA + NB feature selection (3 seeds)", "0.90-0.92", "-", "185-212/561 features, fitness incl. penalty"],
        ],
        [70, 24, 22, 66],
    )
    pdf.figure(
        FIG / "w05_perceptron_har_confusion.png",
        "Figure 1. Own Perceptron confusion matrix (validation).",
        width=110,
    )
    pdf.figure(
        FIG / "foundations_learning_curve.png",
        "Figure 2. Decision Tree capacity vs. train/validation Macro-F1 (Ch.1 diagnostic).",
        width=110,
    )

    # 5. Error/behavior analysis
    pdf.h1("5. Error and behavior analysis")
    pdf.body(
        "The Perceptron's 75/2060 validation errors (3.6%) concentrate on SITTING "
        "predicted as STANDING (34/75, ~45%). Naive Bayes shows the same pair but "
        "reversed and far worse: 463/2060 errors (22.5%), with STANDING predicted as "
        "SITTING accounting for 169/463 (~37%). Both models struggle hardest on the "
        "same pair of static postures, consistent with those two activities producing "
        "the most similar accelerometer/gyroscope signatures in this feature space - "
        "but Naive Bayes is hurt far more by it, because its conditional-independence "
        "assumption double-counts the many correlated features (e.g. multiple axes of "
        "the same sensor) that make SITTING and STANDING hard to tell apart, while the "
        "Perceptron's single linear boundary per class has no such assumption to "
        "violate. The Decision Tree's own capacity sweep shows the classic "
        "underfitting (shallow tree, both scores low) to overfitting (max_depth=20, "
        "train Macro-F1 to 1.0, validation plateaus/falls) transition, with the best "
        "validation score at max_depth=8 in the range tested."
    )

    # 6. Cross-model synthesis
    pdf.h1("6. Cross-model synthesis (Part I)")
    pdf.body(
        "Generative vs. discriminative is the clearest split in this Part I evidence: "
        "the discriminative Perceptron (no independence assumption, just a linear "
        "decision boundary) dominates the generative Naive Bayes (assumes conditional "
        "feature independence given the class) by a wide margin on this 561-correlated-"
        "feature representation. Interestingly, Depth-B feature selection (GA) nearly "
        "closes that gap for Naive Bayes - removing redundant/correlated features "
        "raises its fitness from 0.768 (all features) to 0.90-0.92 (185-212 selected "
        "features), suggesting the independence-violation failure mode is substantially "
        "driven by specific redundant feature groups rather than being an inherent "
        "ceiling for this dataset. The Decision Tree sits between the two: "
        "non-parametric and assumption-light like the Perceptron, but prone to the "
        "same kind of overfitting-with-capacity curve classic to Chapter 1's "
        "bias-variance discussion."
    )

    pdf.add_page()

    # 7. Ownership & learning reflection
    pdf.h1("7. Ownership and learning reflection")
    pdf.body(
        "Concrete ownership evidence from this Part I window: a real ReLU gradient-"
        "check failure in the MLP (W05) traced to a zero-bias-init + dead-ReLU "
        "interaction, fixed by switching to tanh; a Perceptron-rule vs. Delta-rule "
        "misconception corrected by reading ML-From-Scratch's reference code line by "
        "line; a GA mutation bug (`self.mutation_rate` vs. the resolved "
        "`self.mutation_rate_`) found only once the code actually ran against real "
        "data; and a genuine infrastructure failure (GA search crashing from low "
        "system memory) that was debugged and made resumable rather than just retried "
        "blindly. All Depth-A components were written and sanity-tested before any "
        "reference code or AI tool was consulted, per AI_USE.md and the two-state "
        "commit history visible in PROGRESS.md. Remaining uncertainty going into "
        "Part II: whether the GA's selected feature subsets are converging toward a "
        "similar subset across seeds or arriving at similar fitness via different "
        "subsets (not yet checked), and whether MLP would close more of the gap to the "
        "Perceptron once actually trained on HAR."
    )

    # 8. Reproducibility & references
    pdf.h1("8. Reproducibility and references")
    pdf.body(
        "Setup: python -m venv .venv; pip install -r requirements.txt; "
        "python tools/check.py. Every experiment script under "
        "experiments/part1_pre_midterm/ is runnable directly and reads the frozen "
        "split/seed from protocol.yaml / src/data.py. Full citations in REFERENCES.md; "
        "AI use disclosed per-episode in AI_USE.md (Socratic/verification/debugging use "
        "only, after each first-attempt commit, per assignment Section 12)."
    )

    pdf.output(str(OUT))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    build()
