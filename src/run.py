""" PyQt5 desktop GUI for the SA Language N-Gram Predictability Study.
    Load any plain-text corpus files, choose tokenizers and n-gram orders, run the analysis, inspect results in a table, and plot comparisons. """

import sys
import os

import matplotlib
matplotlib.use("Qt5Agg")                       
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QListWidget, QGroupBox,
    QCheckBox, QSpinBox, QDoubleSpinBox,
    QLabel, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QHeaderView,
    QTabWidget, QComboBox,
)
from PyQt5.QtCore import Qt

from preprocess import clean_text, split_data
from tokenizers import CharTokenizer, WordTokenizer, BPETokenizer
from language_model import build_unigram_model, build_ngram_model
from evaluate import cross_entropy_per_token, cross_entropy_per_char, perplexity
from visualize import ORDER_COLORS, TOK_COLORS, BPE_COLOR, _tok_color

# Main window

class LanguageComparisonGUI(QMainWindow):
    """ Main application window. """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SA Language N-Gram Predictability")
        self.setGeometry(100, 100, 1200, 720)

        self.languages   = {} # { file_path: {"name": str, "cleaned_text": str} }
        self.results_data = [] # list of result dicts populated by run_analysis()

        self._build_ui()
        self._centre_window()

    # UI construction
    def _build_ui(self):
    
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)

        root_layout.addLayout(self._build_left_panel(), stretch=1)
        root_layout.addLayout(self._build_right_panel(), stretch=4)

    def _build_left_panel(self):
        """Corpus file list with Add / Clear controls."""

        layout = QVBoxLayout()
        layout.addWidget(QLabel("<b>Loaded corpora</b>"))

        self.lang_list = QListWidget()
        layout.addWidget(self.lang_list)

        add_btn = QPushButton("Add corpus file(s)…")
        add_btn.setToolTip("Load one or more plain-text corpus files")
        add_btn.clicked.connect(self._add_languages)
        layout.addWidget(add_btn)

        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self._clear_languages)
        layout.addWidget(clear_btn)

        return layout

    def _build_right_panel(self):
        """Settings, action buttons, and tabbed output (table + chart)."""
        
        layout = QVBoxLayout()

        layout.addWidget(self._build_settings_group())
        layout.addLayout(self._build_action_buttons())
        layout.addWidget(self._build_output_tabs(), stretch=1)

        return layout

    def _build_settings_group(self):
        """Tokenizer checkboxes, n-gram order selector, smoothing α."""
        
        group  = QGroupBox("Analysis settings")
        grid   = QGridLayout()

        # Tokenizers 
        grid.addWidget(QLabel("Tokenizers:"), 0, 0)

        self.chk_char = QCheckBox("Character")
        self.chk_char.setChecked(True)
        self.chk_char.setToolTip("One token per character (~40–80 token vocabulary)")
        grid.addWidget(self.chk_char, 0, 1)

        self.chk_word = QCheckBox("Word")
        self.chk_word.setChecked(True)
        self.chk_word.setToolTip("One token per whitespace-separated word (large vocabulary)")
        grid.addWidget(self.chk_word, 0, 2)

        self.chk_bpe = QCheckBox("BPE subword")
        self.chk_bpe.setChecked(True)
        self.chk_bpe.setToolTip("Byte Pair Encoding — vocabulary size is configurable below")
        grid.addWidget(self.chk_bpe, 0, 3)

        grid.addWidget(QLabel("BPE vocab size:"), 0, 4)
        self.spin_bpe_vocab = QSpinBox()
        self.spin_bpe_vocab.setRange(50, 10_000)
        self.spin_bpe_vocab.setValue(500)
        self.spin_bpe_vocab.setToolTip("Number of subword types to learn for BPE")
        grid.addWidget(self.spin_bpe_vocab, 0, 5)

        # N-gram order
        grid.addWidget(QLabel("Max n-gram order:"), 1, 0)
        self.spin_max_n = QSpinBox()
        self.spin_max_n.setRange(1, 5)
        self.spin_max_n.setValue(5)
        self.spin_max_n.setToolTip("Evaluate all orders from unigram (1) up to this value")
        grid.addWidget(self.spin_max_n, 1, 1)

        # Smoothing alphe (α) 
        grid.addWidget(QLabel("Smoothing α:"), 1, 2)
        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.001, 10.0)
        self.spin_alpha.setValue(0.1)
        self.spin_alpha.setSingleStep(0.1)
        self.spin_alpha.setDecimals(3)
        self.spin_alpha.setToolTip("Add-alpha (Laplace) smoothing coefficient.\n Smaller values give less probability to unseen n-grams.")
        grid.addWidget(self.spin_alpha, 1, 3)

        group.setLayout(grid)
        return group

    def _build_action_buttons(self):
        """Run Analysis and Plot Comparison buttons."""

        layout = QHBoxLayout()

        run_btn = QPushButton("▶  Run analysis")
        run_btn.setToolTip("Train models and compute entropy / perplexity")
        run_btn.clicked.connect(self._run_analysis)
        layout.addWidget(run_btn)

        self.plot_btn = QPushButton("📊  Plot comparison")
        self.plot_btn.setToolTip("Show grouped bar charts in the Chart tab")
        self.plot_btn.setEnabled(False)
        self.plot_btn.clicked.connect(self._plot_results)
        layout.addWidget(self.plot_btn)

        layout.addStretch()
        return layout

    def _build_output_tabs(self):
        """Two-tab widget: raw results table and embedded chart."""
        self.tabs = QTabWidget()

        # Tab 1 results table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Language", "Tokenizer", "n",
            "H (bits/token)", "H (bits/char)",
            "Perplexity/token", "Perplexity/char",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.tabs.addTab(self.table, "Results table")

        # Tab 2 embedded matplotlib chart
        chart_container = QWidget()
        chart_layout    = QVBoxLayout(chart_container)

        # Controls above the chart
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Metric:"))
        self.combo_metric = QComboBox()
        self.combo_metric.addItems([
            "entropy_token", "entropy_char",
            "perplexity_token", "perplexity_char",
        ])
        self.combo_metric.setCurrentText("entropy_char")
        ctrl_layout.addWidget(self.combo_metric)

        ctrl_layout.addWidget(QLabel("Group by:"))
        self.combo_group = QComboBox()
        self.combo_group.addItems(["n-gram order", "tokenizer"])
        ctrl_layout.addWidget(self.combo_group)

        refresh_btn = QPushButton("Refresh chart")
        refresh_btn.clicked.connect(self._plot_results)
        ctrl_layout.addWidget(refresh_btn)
        ctrl_layout.addStretch()
        chart_layout.addLayout(ctrl_layout)

        # Matplotlib canvas
        self.figure = Figure(figsize=(9, 4.5))
        self.canvas = FigureCanvas(self.figure)
        chart_layout.addWidget(self.canvas)

        self.tabs.addTab(chart_container, "Chart")
        return self.tabs

    # Corpus loading
    
    def _add_languages(self):
        """Opens a file dialog, loads and cleans the selected corpus files."""
        
        files, _ = QFileDialog.getOpenFileNames(self, "Select corpus files", "","Text files (*.txt);;All files (*)")

        for fpath in files:
            if fpath in self.languages:
                continue   # already loaded
            try:
                with open(fpath, encoding="utf-8") as f:
                    raw = f.read()
                
                cleaned   = clean_text(raw)
                lang_name = os.path.splitext(os.path.basename(fpath))[0]
                
                self.languages[fpath] = {"name": lang_name, "cleaned_text": cleaned,}
                self.lang_list.addItem(lang_name)

            except Exception as exc:
                
                QMessageBox.warning(self, "Load error", f"Could not load {os.path.basename(fpath)}:\n{exc}")

    def _clear_languages(self):
        """Removes all loaded corpora."""

        self.languages.clear()
        self.lang_list.clear()
        self.results_data.clear()
        self.table.setRowCount(0)
        self.plot_btn.setEnabled(False)

    # Analysis
    def _run_analysis(self):
        """ For each loaded corpus build tokenizers, Encode the text and split 80/10/10, Train n-gram models for n = 1 .. max_n,
            Compute entropy and perplexity on the test split, Populate the results table. """
        
        if not self.languages:
            QMessageBox.information(self, "No corpora loaded", "Use 'Add corpus file(s)' to load at least one language.")
            return

        max_n = self.spin_max_n.value()
        alpha = self.spin_alpha.value()
        use_char = self.chk_char.isChecked()
        use_word = self.chk_word.isChecked()
        use_bpe = self.chk_bpe.isChecked()
        bpe_vocab = self.spin_bpe_vocab.value()

        if not (use_char or use_word or use_bpe):
            QMessageBox.warning(self, "No tokenizer selected", "Please select at least one tokenizer.")
            return

        # Disable the window while running to prevent double-clicks
        self.setEnabled(False)
        QApplication.processEvents()

        self.results_data = []
        self.table.setRowCount(0)

        try:
            for lang_info in self.languages.values():
                lang_name = lang_info["name"]
                text      = lang_info["cleaned_text"]

                # Build the requested tokenizers
                tokenizers = self._build_tokenizers(text, use_char, use_word, use_bpe, bpe_vocab, lang_name)
                
                if not tokenizers:
                    continue

                for tok in tokenizers:
                    
                    encoded = tok.encode(text)
                    train, val, test = split_data(encoded)

                    for n in range(1, max_n + 1):
                        if n == 1:
                            model = build_unigram_model(train, tok.vocab_size)
                        else:
                            model = build_ngram_model(train, n, tok.vocab_size)

                        h_tok = cross_entropy_per_token(test, model, alpha)
                        h_char = cross_entropy_per_char(test, model, tok, alpha)
                        ppl_tok = perplexity(h_tok)
                        ppl_char = perplexity(h_char)

                        self.results_data.append({
                            "lang":      lang_name,
                            "tokenizer": tok.name,
                            "n": n,
                            "entropy_token": h_tok,
                            "entropy_char": h_char,
                            "perplexity_token": ppl_tok,
                            "perplexity_char": ppl_char,
                        })

            self._populate_table()
            self.plot_btn.setEnabled(bool(self.results_data))

            if self.results_data:
                QMessageBox.information(self, "Done", f"Analysis complete — {len(self.results_data)} result rows.")

            else:

                QMessageBox.warning(self, "No results", "Analysis produced no results. Check your corpus files.")

        except Exception as exc:
            
            QMessageBox.critical(self, "Analysis error", f"Something went wrong:\n{exc}")

        finally:
            
            self.setEnabled(True)

    def _build_tokenizers(self, text, use_char, use_word, use_bpe, bpe_vocab, lang_name):
        """Constructs and returns the list of active tokenizer objects."""
        
        tokenizers = []

        if use_char:
            tokenizers.append(CharTokenizer(text))

        if use_word:
            tokenizers.append(WordTokenizer(text))

        if use_bpe:
            try:
                tokenizers.append(BPETokenizer(text, vocab_size=bpe_vocab))
            
            except Exception as exc:
                
                QMessageBox.warning(self, "BPE error", f"BPE training failed for '{lang_name}':\n{exc}\n BPE will be skipped for this corpus.")

        return tokenizers

    # Table
    def _populate_table(self):
        """Fills the results table from self.results_data."""
        
        self.table.setRowCount(len(self.results_data))

        order_names = {1: "Unigram", 2: "Bigram", 3: "Trigram", 4: "4-gram",  5: "5-gram",}

        for row, res in enumerate(self.results_data):

            self.table.setItem(row, 0, QTableWidgetItem(res["lang"]))
            self.table.setItem(row, 1, QTableWidgetItem(res["tokenizer"]))
            self.table.setItem(row, 2, QTableWidgetItem(order_names.get(res["n"], str(res["n"]))))
            self.table.setItem(row, 3, QTableWidgetItem(f"{res['entropy_token']:.4f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{res['entropy_char']:.4f}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{res['perplexity_token']:.2f}"))
            self.table.setItem(row, 6, QTableWidgetItem(f"{res['perplexity_char']:.2f}"))

        self.tabs.setCurrentIndex(0)   # switch to table tab

    # Charting
    def _plot_results(self):
        """ Draws a grouped bar chart in the Chart tab."""
    
        if not self.results_data:

            QMessageBox.information(self, "No results", "Run analysis first.")
            return

        metric = self.combo_metric.currentText()
        group_by = self.combo_group.currentText()

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if group_by == "n-gram order":
            self._draw_ngram_chart(ax, metric)
        
        else:
            self._draw_tokenizer_chart(ax, metric)

        self.canvas.draw()
        self.tabs.setCurrentIndex(1)   # switch to chart tab

    def _draw_ngram_chart(self, ax, metric):
        """ One group per language, one bar per n-gram order. Shows how entropy / perplexity changes with n. """

        langs = sorted({r["lang"]  for r in self.results_data})
        orders = sorted({r["n"]     for r in self.results_data})

        x = list(range(len(langs)))
        bar_width = 0.7 / len(orders)

        order_names = {1: "Unigram", 2: "Bigram", 3: "Trigram", 4: "4-gram",  5: "5-gram",}

        for idx, n in enumerate(orders):
            
            offset = (idx - len(orders) / 2 + 0.5) * bar_width
            vals = []
            
            for lang in langs:
                matches = [r[metric] for r in self.results_data if r["lang"] == lang and r["n"] == n]
                vals.append(matches[0] if matches else 0.0)

            bars = ax.bar(
                [xi + offset for xi in x], vals, bar_width,
                label=order_names.get(n, f"{n}-gram"),
                color=ORDER_COLORS.get(n, "#888888"),
            )
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([l.capitalize() for l in langs], fontsize=10)
        ax.set_ylabel(_metric_label(metric), fontsize=10)
        ax.set_title(f"{_metric_label(metric)} by N-Gram Order", fontweight="bold")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        self.figure.tight_layout()

    def _draw_tokenizer_chart(self, ax, metric):
        """ One group per language, one bar per tokenizer. Uses the highest n-gram order available for comparison. """
        langs = sorted({r["lang"]      for r in self.results_data})
        tokenizers = []

        for r in self.results_data:         # preserve insertion order
            if r["tokenizer"] not in tokenizers:
                tokenizers.append(r["tokenizer"])

        n_max = max(r["n"] for r in self.results_data)
        x = list(range(len(langs)))
        bar_width = 0.7 / len(tokenizers)

        for idx, tok in enumerate(tokenizers):
            offset = (idx - len(tokenizers) / 2 + 0.5) * bar_width
            vals   = []
            
            for lang in langs:
                matches = [r[metric] for r in self.results_data if r["lang"] == lang and r["tokenizer"] == tok and r["n"] == n_max]
                vals.append(matches[0] if matches else 0.0)

            bars = ax.bar(
                [xi + offset for xi in x], vals, bar_width,
                label=tok,
                color=_tok_color(tok),
            )
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([l.capitalize() for l in langs], fontsize=10)
        ax.set_ylabel(_metric_label(metric), fontsize=10)
        ax.set_title(
            f"{_metric_label(metric)} by Tokenizer  (n={n_max})",
            fontweight="bold"
        )
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        self.figure.tight_layout()

    # Window utilities
    def _centre_window(self):
        """Moves the window to the centre of the primary screen."""
        
        frame = self.frameGeometry()
        centre = QApplication.primaryScreen().availableGeometry().center()
        frame.moveCenter(centre)
        self.move(frame.topLeft())


# Helpers
def _metric_label(metric):
    """Readable axis label for a metric key."""
    return {
        "entropy_token": "Entropy (bits / token)",
        "entropy_char": "Entropy (bits / character)",
        "perplexity_token": "Perplexity (per token)",
        "perplexity_char": "Perplexity (per character)",
    }.get(metric, metric)


# Entry point
if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")      
    gui = LanguageComparisonGUI()
    gui.show()
    sys.exit(app.exec_())