from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QScrollArea,
    QVBoxLayout, QHBoxLayout, QTextEdit, QGroupBox, QComboBox,
    QRadioButton, QButtonGroup, QGridLayout, QDoubleSpinBox, QCheckBox)

from sqlalchemy import create_engine

import sys
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
import tempfile

from datetime import datetime
import os


# Wczytanie danych z pliku CSV.
def load_data(path):
    """
    Wczytuje dane z pliku CSV do dataframe.
    """
    try:
        data = pd.read_csv(path, delimiter=',')
        return data
    except FileNotFoundError:
        print('File not found')


# GUI: wybór pliku i filtrów.
class MainWindow(QWidget):
    """
    Główne okno aplikacji.
    Zawiera interfejs do ładowania danych, filtrowania, grupowania,
    generowania wykresów oraz tworzenia raportów.
    """

    def __init__(self):
        super().__init__()

        # Inicjalizacja stanu
        self.db_engine = None
        self.stats = None
        self.data_source = None
        self.grouping_columns = None
        self.applied_filters = None
        self.aggregation_functions = None
        self.data = None
        self.canvas = None
        self.current_figure = None
        self.current_plot_data = None

        # Etykiety i jednostki kolumn
        self.column_labels = {
            'Gender': 'Gender',
            'AGE': 'Age',
            'Urea': 'Urea',
            'Cr': 'Creatinine',
            'HbA1c': 'HbA1c',
            'Chol': 'Cholesterol',
            'TG': 'Triglycerides',
            'HDL': 'HDL',
            'LDL': 'LDL',
            'VLDL': 'VLDL',
            'BMI': 'BMI',
            'CLASS': 'Diabetes risk classification'
        }

        self.column_units = {
            'Urea': 'mmol/L',
            'AGE': 'years',
            'Cr': 'μmol/L',
            'HbA1c': '%',
            'Chol': 'mmol/L',
            'TG': 'mmol/L',
            'HDL': 'mmol/L',
            'LDL': 'mmol/L',
            'VLDL': 'mmol/L'
        }

        # Ustawienia okna
        self.setWindowTitle('Data analysis')
        self.setGeometry(100, 100, 1200, 700)

        #  Główne layouty
        self.main_layout = QVBoxLayout()
        self.center_layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        #  Tworzenie sekcji GUI
        self._create_left_panel()
        self._create_right_panel()
        self._create_bottom_panel()

        #  Składanie layoutów
        self.center_layout.addWidget(self.left_scroll_area, 1)
        self.center_layout.addWidget(self.right_box, 2)
        self.main_layout.addWidget(self.load_file_box)
        self.main_layout.addLayout(self.center_layout, 3)
        self.main_layout.addWidget(self.bottom_box, 1)

        #  Finalne ustawienia interfejsu
        self.initialize_ui_state(enabled=False)
        self.setLayout(self.main_layout)

    def _create_left_panel(self):
        """Tworzy lewą część interfejsu"""
        self.left_box = QWidget()
        self.left_box.setLayout(self.left_layout)
        self.left_box.setMinimumWidth(200)

        # Ładowanie plików
        self._create_load_file_section()

        # Filtry
        self._create_filter_section()

        # Grupowanie
        self._create_grouping_section()

        # Funkcje agregujące
        self._create_aggregation_section()

        # Opcje wykresu
        self._create_options_section()

        # Wykresy
        self._create_chart_section()

        # Raporty
        self._create_report_section()

        # Reset ustawień
        self.clear_filters_btn = QPushButton('❌ Reset settings')
        self.clear_filters_btn.clicked.connect(self.clear_filters)
        self.left_layout.addWidget(self.clear_filters_btn)

        # Scroll area
        self.left_scroll_area = QScrollArea()
        self.left_scroll_area.setWidgetResizable(True)
        self.left_scroll_area.setWidget(self.left_box)

    def _create_right_panel(self):
        """Panel z wizualizacjami (wykresami)."""
        self.right_box = QGroupBox('📊 Charts')
        self.right_box.setLayout(self.right_layout)

    def _create_bottom_panel(self):
        """Panel dolny (logi)."""
        self.bottom_box = QGroupBox('📝 Logs')
        bottom_layout = QVBoxLayout()
        self.label = QLabel('No file selected')
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFixedHeight(100)

        bottom_layout.addWidget(self.label)
        bottom_layout.addWidget(self.log_area)
        self.bottom_box.setLayout(bottom_layout)

    def _create_load_file_section(self):
        """Sekcja: Ładowanie danych."""
        self.load_file_box = QGroupBox('📁 Load file:')
        layout = QVBoxLayout()

        # Przyciski wyboru pliku
        file_button_layout = QHBoxLayout()
        self.csv_button = QPushButton('Select CSV File')
        self.csv_button.clicked.connect(self.select_csv_file)
        self.db_button = QPushButton('Connect with SQLite Database')
        self.db_button.clicked.connect(self.load_data_from_sqlite)
        file_button_layout.addWidget(self.csv_button)
        file_button_layout.addWidget(self.db_button)

        layout.addLayout(file_button_layout)
        self.load_file_box.setLayout(layout)

    def _create_filter_section(self):
        """Sekcja: Filtry."""
        self.filter_group_box = QGroupBox('🔍 Filter by:')
        layout = QVBoxLayout()

        self.filter_column_combo = QComboBox()
        self.filter_column_combo.currentIndexChanged.connect(self.update_filter_values)
        layout.addWidget(self.filter_column_combo)

        #  Filtry numeryczne
        self.filter_min_spinbox = QDoubleSpinBox()
        self.filter_max_spinbox = QDoubleSpinBox()
        for spin in (self.filter_min_spinbox, self.filter_max_spinbox):
            spin.setRange(0, 1000)
            spin.setDecimals(1)
        self.filter_min_spinbox.setPrefix('From: ')
        self.filter_max_spinbox.setPrefix('To: ')

        spin_layout = QHBoxLayout()
        spin_layout.addWidget(self.filter_min_spinbox)
        spin_layout.addWidget(self.filter_max_spinbox)
        layout.addLayout(spin_layout)

        self.filter_min_spinbox.hide()
        self.filter_max_spinbox.hide()

        #  Filtry kategoryczne
        self.category_combo_label = QLabel('Select value:')
        self.category_filter_combo = QComboBox()
        self.category_filter_combo.currentIndexChanged.connect(self.category_combo_changed)

        layout.addWidget(self.category_combo_label)
        layout.addWidget(self.category_filter_combo)
        self.category_combo_label.hide()
        self.category_filter_combo.hide()

        self.filter_group_box.setLayout(layout)
        self.left_layout.addWidget(self.filter_group_box)

    def _create_grouping_section(self):
        """Sekcja: Grupowanie."""
        self.grouping_group_box = QGroupBox('📑 Group by:')
        layout = QVBoxLayout()

        self.group_column_combo = QComboBox()
        self.group_column_combo.currentIndexChanged.connect(self.update_numeric_columns)
        self.agg_column_combo = QComboBox()

        self.raw_data_checkbox = QCheckBox('Use raw data (no aggregation)')
        self.raw_data_checkbox.setChecked(False)
        self.raw_data_checkbox.stateChanged.connect(self.update_ui)

        layout.addWidget(self.group_column_combo)
        layout.addWidget(self.agg_column_combo)
        layout.addWidget(self.raw_data_checkbox)

        self.grouping_group_box.setLayout(layout)
        self.left_layout.addWidget(self.grouping_group_box)

    def _create_aggregation_section(self):
        """Sekcja: Funkcje agregujące."""
        self.agg_func_groupbox = QGroupBox('Aggregate by:')
        layout = QGridLayout()

        self.agg_func_group = QButtonGroup(self)
        self.agg_func_buttons = {}

        functions = {
            'mean': 'Average values',
            'median': 'Median values',
            'count': 'Number of patients',
            'min': 'Minimum values',
            'max': 'Maximum values'
        }

        row = col = 0
        for key, label in functions.items():
            btn = QRadioButton(label)
            btn.toggled.connect(self.update_ui)
            layout.addWidget(btn, row, col)
            self.agg_func_group.addButton(btn)
            self.agg_func_buttons[key] = btn
            col += 1
            if col > 1:
                col = 0
                row += 1

        self.agg_func_groupbox.setLayout(layout)
        self.left_layout.addWidget(self.agg_func_groupbox)

    def _create_options_section(self):
        """Sekcja: Opcje wykresu."""
        self.gender_checkbox = QCheckBox('Show gender differences')
        self.bin_checkbox = QCheckBox('Show data in ranges')
        self.trendline_checkbox = QCheckBox("Show trend line")

        for cb in (self.bin_checkbox, self.trendline_checkbox, self.raw_data_checkbox):
            cb.stateChanged.connect(self.update_checkboxes_visibility)

        layout = QGridLayout()
        layout.addWidget(self.gender_checkbox, 0, 0)
        layout.addWidget(self.bin_checkbox, 0, 1)
        layout.addWidget(self.trendline_checkbox, 1, 0)

        self.options_box = QGroupBox('Options:')
        self.options_box.setLayout(layout)
        self.left_layout.addWidget(self.options_box)

    def _create_chart_section(self):
        """Sekcja: Wybór i generowanie wykresów."""
        self.chart_group_box = QGroupBox('📈 Chart:')
        layout = QVBoxLayout()

        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItem('Select chart type')
        self.chart_type_combo.addItems([
            'Bar Chart', 'Pie Chart', 'Line Chart',
            'Scatter Plot', 'Histogram', 'Heatmap'
        ])
        self.chart_type_combo.currentIndexChanged.connect(self.chart_type_changed)

        self.generate_chart_btn = QPushButton('Generate chart')
        self.generate_chart_btn.clicked.connect(self.update_chart)

        layout.addWidget(self.chart_type_combo)
        layout.addWidget(self.generate_chart_btn)

        self.chart_group_box.setLayout(layout)
        self.left_layout.addWidget(self.chart_group_box)

    def _create_report_section(self):
        """Sekcja: Raporty (CSV, PDF)."""
        self.report_box = QGroupBox('📄 Generate report')
        layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.generate_csv_btn = QPushButton('Generate CSV report')
        self.generate_pdf_btn = QPushButton('Generate PDF report')
        self.generate_csv_btn.clicked.connect(lambda: self.generate_report('csv'))
        self.generate_pdf_btn.clicked.connect(lambda: self.generate_report('pdf'))

        btn_layout.addWidget(self.generate_csv_btn)
        btn_layout.addWidget(self.generate_pdf_btn)
        layout.addLayout(btn_layout)

        self.report_box.setLayout(layout)
        self.left_layout.addWidget(self.report_box)

    def clear_filters(self):
        """
        Resetuje wszystkie filtry, grupowanie i typ wykresu oraz przywraca początkowy stan UI.
        """
        # Reset dynamicznych wartości filtrów i grupowania
        self.filter_column_combo.setCurrentIndex(0)
        self.filter_min_spinbox.setValue(self.filter_min_spinbox.minimum())
        self.filter_max_spinbox.setValue(self.filter_max_spinbox.maximum())
        self.filter_min_spinbox.hide()
        self.filter_max_spinbox.hide()
        self.category_filter_combo.setCurrentIndex(0)
        self.category_filter_combo.hide()
        self.category_combo_label.hide()
        self.group_column_combo.setCurrentIndex(0)
        self.agg_column_combo.setCurrentIndex(0)

        self.agg_func_group.setExclusive(False)
        for btn in self.agg_func_buttons.values():
            btn.setChecked(False)
        self.agg_func_group.setExclusive(True)

        self.chart_type_combo.setCurrentIndex(0)
        self.clear_right_panel()

        self.update_ui()

        self.log_area.append('Filters and grouping have been reset.')

    def clear_right_panel(self):
        """
        Usuwa wszystkie wykresy z prawego panelu (right_layout).
        """
        for i in reversed(range(self.right_layout.count())):
            item = self.right_layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                self.right_layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()

    # Funkcje dotyczące grupowania i filtrowania danych
    def get_filtered_data(self):
        """
        Zwraca dane przefiltrowane według aktualnie wybranych ustawień filtrowania.
        Obsługuje filtrowanie numeryczne i kategoryczne.
        """
        if self.data is None:
            return None

        filter_col = self.filter_column_combo.currentData()
        if not filter_col:
            return self.data.copy()

        col_data = self.data[filter_col]

        # Filtrowanie w kolumnach z wartościami numerycznymi
        if pd.api.types.is_numeric_dtype(col_data):
            min_val = self.filter_min_spinbox.value()
            max_val = self.filter_max_spinbox.value()
            filtered_df = self.data[(col_data >= min_val) & (col_data <= max_val)].copy()
            filter_description = f"{filter_col}: {min_val} - {max_val}"
        # Filtrowanie w kolumnach z wartościami nienumerycznymi
        elif self.category_filter_combo.isVisible():
            selected_val = self.category_filter_combo.currentData()
            if selected_val is not None:
                filtered_df = self.data[self.data[filter_col].astype(str) == str(selected_val)].copy()
                filter_description = f"{filter_col} = '{selected_val}'"
            else:
                filtered_df = self.data.copy()
                filter_description = f"{filter_col} - no value selected"
        else:
            filtered_df = self.data.copy()
            filter_description = "-"
        #  Zapis informacji o filtrze
        self.applied_filters = filter_description
        return filtered_df

    def get_selected_agg_func(self):
        """
        Zwraca klucz wybranej funkcji agregującej lub None.
        """
        for func_key, btn in self.agg_func_buttons.items():
            if btn.isChecked():
                return func_key
        return None

    def aggregate_grouped_data(self, df, group_cols, agg_col, agg_func):
        """
        Grupuje i agreguje dane według podanych kolumn i funkcji agregującej.
        """
        try:
            if agg_func == 'count':
                grouped = df.groupby(group_cols, observed=True).size().reset_index(name='Number of patients')
                y_col = 'Number of patients'
            else:
                grouped = df.groupby(group_cols, observed=True).agg({agg_col: agg_func}).reset_index()
                y_col = agg_col
            return grouped, y_col
        except Exception as e:
            self.log_area.append(f"Error during aggregation: {e}")
            return None, None

    def prepare_aggregated_data(self):
        """
        Zwraca dane po filtrowaniu, binowaniu i agregacji według wybranych parametrów.
        """
        # Filtrowanie
        df = self.get_filtered_data()
        if df is None or df.empty:
            return None
        # Pobranie ustawień agregacji
        group_col = self.group_column_combo.currentData()
        agg_func = self.get_selected_agg_func()
        agg_col = self.agg_column_combo.currentData()

        if not group_col or not agg_func or (agg_func != 'count' and not agg_col):
            return None
        # Grupowanie (uwzględnienie płci)
        group_cols = [group_col]
        if self.gender_checkbox.isChecked() and 'Gender' in df.columns:
            group_cols.append('Gender')
        # Binowanie kolumny, jeśli zaznaczone
        df, group_col_or_binned = self.bin_column_if_needed(df, group_col)
        if group_col_or_binned != group_col:
            group_cols[0] = group_col_or_binned
        # Agregacja
        grouped_df, y_col = self.aggregate_grouped_data(df, group_cols, agg_col, agg_func)

        # Dane do raportu
        self.grouping_columns = ", ".join(group_cols)
        if agg_func == 'count':
            self.aggregation_functions = "count (number of patients)"
        else:
            self.aggregation_functions = f"{agg_func} of '{agg_col}'"

        # Zaokrąglenie wartości tylko kolumn numerycznych
        for col in grouped_df.select_dtypes(include='number').columns:
            grouped_df[col] = grouped_df[col].round(2)
        return grouped_df

    def bin_column_if_needed(self, df, group_col):
        """
        Dodaje zbinowaną wersję kolumny grupującej, jeśli opcja binowania jest włączona.
        """
        if self.bin_checkbox.isChecked() and group_col in df.columns and pd.api.types.is_numeric_dtype(df[group_col]):
            binned_col_name = f'{group_col} in ranges'
            df[binned_col_name] = self.bin_numeric_column(df[group_col], column_name=group_col)
            return df, binned_col_name
        return df, group_col

    def bin_numeric_column(self, series, column_name=None):
        """
        Dzieli kolumnę numeryczną na przedziały z poprawnym uwzględnieniem wartości granicznych.
        """
        try:
            # Definicje przedziałów i etykiet dla poszczególnych kolumn
            bin_configs = {
                'BMI': {
                    'bins': [18.5, 24.9, 29.9, 34.9, 39.9, np.inf],
                    'labels': ['18.5-24.9', '25.0-29.9', '30.0-34.9', '35.0-39.9', '40.0+']
                },
                'AGE': {
                    'bins': [20, 30, 40, 50, 60, 70, np.inf],
                    'labels': ['20-29', '30-39', '40-49', '50-59', '60-69', '70+']
                },
                'Chol': {
                    'bins': [0, 4.99, 5.99, 6.99, np.inf],
                    'labels': ['<5', '5-5.99', '6-6.99', '7+']
                },
                'Cr': {
                    'bins': [0, 49.99, 99.99, 149.99, 199.99, np.inf],
                    'labels': ['<50', '50-99.99', '100-149.99', '150-199.99', '200+']
                },
                'HbA1c': {
                    'bins': [0, 5.6, 6.49, np.inf],
                    'labels': ['<5.7', '5.7-6.49', '6.5+']
                },
                'TG': {
                    'bins': [0, 1.69, 2.99, 3.99, 4.99, np.inf],
                    'labels': ['<1.7', '1.7-2.99', '3-3.99', '4-4.99', '5+']
                },
                'HDL': {
                    'bins': [0, 1.19, 2.19, 3.99, 4.99, np.inf],
                    'labels': ['<1.2', '1.2-2.19', '2.2-3.99', '4-4.99', '5+']
                },
                'LDL': {
                    'bins': [0, 2.99, 3.99, 4.99, 5.99, np.inf],
                    'labels': ['<3', '3-3.99', '4-4.99', '5-5.99', '6+']
                },
                'VLDL': {
                    'bins': [0, 0.79, 2.99, 4.99, np.inf],
                    'labels': ['<0.8', '0.8-2.99', '3-4.99', '5+']
                },
                'Urea': {
                    'bins': [0, 1.99, 2.99, 3.99, 4.99, 5.99, np.inf],
                    'labels': ['<2', '2-2.99', '3-3.99', '4-4.99', '5-5.99', '6+']
                }
            }

            if column_name in bin_configs:
                bins = bin_configs[column_name]['bins']
                labels = bin_configs[column_name]['labels']
                cat_type = CategoricalDtype(categories=labels, ordered=True)
                return pd.cut(series, bins=bins, labels=labels, right=False, include_lowest=True).astype(cat_type)
            else:
                # Dla pozostałych: niezmienione
                return series

        except Exception as e:
            self.log_area.append(f"Error while grouping values '{column_name}': {e}")
            return series

    # Funkcje generujące wykresy
    def update_chart(self):
        """
        Generuje wykres na podstawie aktualnych ustawień użytkownika.
        Obsługuje różne typy wykresów (w tym histogram i heatmapę).
        """
        # Czyszczenie poprzednich wykresów
        self.clear_right_panel()
        selected_chart = self.chart_type_combo.currentText()

        # Histogram i heatmapa
        if selected_chart in ['Histogram', 'Heatmap']:
            df = self.get_filtered_data()
            if df is None or df.empty:
                self.log_area.append('No data to display.')
                return
            self.current_plot_data = df

            if selected_chart == 'Histogram':
                filter_col = self.filter_column_combo.currentData()
                if not filter_col:
                    self.log_area.append('No filter column selected for histogram.')
                    return
                self.generate_hist(df, column=filter_col)
            else:
                self.generate_heatmap(df)
        # Pozostałe typy wykresów
        else:
            raw_mode = self.raw_data_checkbox.isChecked()
            self.handle_other_charts(raw_mode=raw_mode)

    def prepare_plot_params(self, x_col, y_col, agg_func=None, is_raw_mode=False):
        """
        Przygotowuje etykiety i tytuł wykresu na podstawie wybranych kolumn i trybu agregacji.
        Zwraca słownik z tytułem wykresu i opisami osi.
        """

        # Wyodrębnienie nazw bazowych kolumn
        x_base = x_col.replace(' in ranges', '').strip()
        y_base = y_col

        # Jednostki
        x_unit = self.column_units.get(x_base, '')
        y_unit = self.column_units.get(y_base, '') if agg_func != 'count' and y_base else ''

        # Etykiety osi
        x_label = self.column_labels.get(x_col, self.column_labels.get(x_base, x_col))
        y_label = self.column_labels.get(y_col, self.column_labels.get(y_base, y_col)) if y_col else ''

        # Tytuł wykresu
        if is_raw_mode:
            title = f'{y_label} by {x_label} (raw data)'
        elif agg_func == 'count':
            title = f'{self.agg_func_buttons[agg_func].text()} by {x_label}'
        elif agg_func:
            title = f'{self.agg_func_buttons[agg_func].text()} of {y_label} by {x_label}'
        else:
            title = f'{y_label} by {x_label}'

        return {
            'x_label': f'{x_label} ({x_unit})' if x_unit else x_label,
            'y_label': f'{y_label} ({y_unit})' if y_unit else y_label,
            'title': title
        }

    def handle_other_charts(self, raw_mode=False):
        """
        Generuje wykresy wymagające agregowanych danych lub surowych danych w zależności od trybu.
        """
        # Tryb surowych danych (bez agregacji)
        if raw_mode:
            df = self.get_filtered_data()
            if df is None or df.empty:
                self.log_area.append('No data to display.')
                return

            group_col = self.group_column_combo.currentData()
            if not group_col:
                self.log_area.append('No group column selected.')
                return

            x_col = group_col
            y_col = self.agg_column_combo.currentData()
            agg_func = None
            hue_col = None

            params = self.prepare_plot_params(x_col, y_col, is_raw_mode=True)

            self.generate_chart(
                data=df,
                x_col=x_col,
                y_col=y_col,
                x_label=params['x_label'],
                y_label=params['y_label'],
                title=params['title'],
                hue_col=hue_col,
                agg_func=agg_func,
                raw_mode=True
            )
            return

        # Tryb danych agregowanych
        grouped = self.prepare_aggregated_data()
        if grouped is None or grouped.empty:
            self.log_area.append('No data to display.')
            return

        group_col = self.group_column_combo.currentData()
        agg_func = self.get_selected_agg_func()
        if not agg_func:
            self.log_area.append('No aggregation function selected.')
            return

        x_col = f'{group_col} in ranges' if self.bin_checkbox.isChecked() else group_col
        y_col = grouped.columns[-1]

        params = self.prepare_plot_params(x_col, y_col, agg_func, is_raw_mode=False)

        hue_col = 'Gender' if self.gender_checkbox.isChecked() else None

        self.generate_chart(
            data=grouped,
            x_col=x_col,
            y_col=y_col,
            x_label=params['x_label'],
            y_label=params['y_label'],
            title=params['title'],
            hue_col=hue_col,
            agg_func=agg_func,
            raw_mode=False
        )

    def generate_hist(self, data, column):
        """
        Generuje histogram dla wybranej kolumny numerycznej z oznaczeniem średniej i mediany.
        """
        # Sprawdzenie kolumny
        if not column or column not in data.columns:
            self.log_area.append('Invalid column for histogram.')
            return

        if not pd.api.types.is_numeric_dtype(data[column]):
            self.log_area.append(f'Column "{column}" is not numeric and cannot be used for histogram.')
            return

        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        col_name = self.column_labels.get(column, column)
        col_unit = self.column_units.get(column, '')

        try:
            # Oblicz i zapisz statystyki
            self.stats = self.statistics(data, column)
            if self.stats is None:
                return

            # Rysuj histogram
            sns.histplot(
                data=data,
                x=column,
                bins=30,
                color='skyblue',
                edgecolor='black',
                ax=ax
            )

            # Dodaj linie pionowe dla średniej i mediany
            ax.axvline(self.stats['mean'], color='red', linestyle='--', linewidth=2,
                       label=f"Mean: {self.stats['mean']:.2f}")
            ax.axvline(self.stats['median'], color='green', linestyle=':', linewidth=2,
                       label=f"Median: {self.stats['median']:.2f}")

            # Dodaj tytuł, etykiety i legendę
            ax.set_title(f'Histogram of {col_name}')
            ax.set_xlabel(f'{col_name} ({col_unit})' if col_unit else f'{col_name}')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True)
            self.render_figure(fig)
            self.log_area.append(f'Histogram has been generated.')

        except Exception as e:
            self.log_area.append(f'Error generating histogram: {e}')

    def generate_heatmap(self, data):
        """
        Generuje heatmapę korelacji dla danych numerycznych.
        """
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        try:
            # Kolumny numeryczne
            numeric_data = data.select_dtypes(include='number')
            if numeric_data.shape[1] < 2:
                self.log_area.append('Not enough numeric columns for heatmap.')
                return

            corr = numeric_data.corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            ax.set_title('Heatmap of biochemical parameters')

            self.render_figure(fig)
            self.log_area.append('Heatmap has been generated.')
        except Exception as e:
            self.log_area.append(f'Error generating heatmap: {e}')

    def generate_chart(self, data, x_col, y_col, x_label, y_label, title, hue_col=None, agg_func=None, raw_mode=False):
        """
        Generowanie pozostałych wykresów na podstawie wybranego typu.
        """
        selected_chart = self.chart_type_combo.currentText()
        # Sprawdzenie typu wykresu
        if selected_chart == 'Select chart type':
            self.log_area.append('Please select a chart type before generating.')
            return

        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        try:
            other_plots = {
                'Bar Chart': self.bar_chart,
                'Line Chart': self.line_chart,
                'Scatter Plot': self.scatter_plot,
                'Pie Chart': self.pie_chart,
            }

            plot_func = other_plots.get(selected_chart)

            if not plot_func:
                return self.log_area.append(f'Unsupported chart type: {selected_chart}')

            # Wywołanie funkcji wykresu z obsługą sprawdzenia zwrotu
            if selected_chart == 'Pie Chart':
                plot = plot_func(ax, data, x_col, y_col, agg_func)
            elif selected_chart == 'Scatter Plot':
                plot = plot_func(ax, data, x_col, y_col, hue_col, raw_mode)
                if plot:
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.grid(True)
            else:
                plot = plot_func(ax, data, x_col, y_col, hue_col)
                if plot:
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.grid(True)

            if not plot:
                return

            full_title = ('Gender differences: ' if hue_col else '') + title
            ax.set_title(full_title, pad=20)

            self.render_figure(fig)
            self.log_area.append(f'{selected_chart} has been generated.')

        except Exception as e:
            self.log_area.append(f"Error while generating chart: {e}")

    def bar_chart(self, ax, data, x_col, y_col, hue_col):
        palette = {'F': '#fe46a5', 'M': '#82cafc'}
        data = data.copy()
        data[x_col] = data[x_col].astype(str)
        if hue_col:
            sns.barplot(data=data, x=x_col, y=y_col, hue=hue_col, palette=palette, ax=ax)
        else:
            sns.barplot(data=data, x=x_col, y=y_col, ax=ax)
        return True

    def line_chart(self, ax, data, x_col, y_col, hue_col):
        palette = {'F': '#fe46a5', 'M': '#82cafc'}
        if hue_col:
            sns.lineplot(data=data, x=x_col, y=y_col, hue=hue_col, palette=palette, ax=ax)
        else:
            sns.lineplot(data=data, x=x_col, y=y_col, ax=ax)
        return True

    def scatter_plot(self, ax, data, x_col, y_col, hue_col, raw_mode=False):
        palette = {'F': '#fe46a5', 'M': '#82cafc'}
        show_trend = self.trendline_checkbox.isChecked()

        if hue_col:
            sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col, palette=palette, ax=ax)

        elif raw_mode or not hue_col:
            sns.scatterplot(data=data, x=x_col, y=y_col, ax=ax)

        if show_trend:
            sns.regplot(
                data=data,
                x=x_col,
                y=y_col,
                scatter=False,
                ax=ax,
                line_kws={'color': 'red', 'linestyle': '--'},
                ci=None
            )

        return True

    def pie_chart(self, ax, data, x_col, y_col, agg_func):

        binning_enabled = self.bin_checkbox.isChecked()
        hue_col = self.gender_checkbox.isChecked()
        is_numeric_col = pd.api.types.is_numeric_dtype(data[x_col])
        agg_is_count = (agg_func == 'count')

        # Sprawdzanie ograniczeń
        if hue_col:
            self.log_area.append('Pie Chart does not support grouping by gender')
            return False

        if not agg_is_count:
            self.log_area.append('Pie Chart can only be generated with aggregation function "count".')
            return False

        if is_numeric_col and not binning_enabled:
            self.log_area.append('Pie Chart requires binning enabled for numeric columns.')
            return False
        try:
            values = data[y_col]
            labels = data[x_col]
            cmap = plt.get_cmap('tab20')
            colors = [cmap(i) for i in range(len(labels))]

            wedges, _, _ = ax.pie(
                values,
                labels=None,
                autopct='%1.1f%%',
                colors=colors,
                pctdistance=1.2
            )

            if x_col == 'CLASS':
                class_labels = {'Y': 'Diabetic', 'N': 'Non-Diabetic', 'P': 'Prediction\n of diabetes'}
                legend_labels = [class_labels.get(str(label), str(label)) for label in labels]
            else:
                legend_labels = list(labels)

            original_col = x_col.strip(' in ranges') if x_col.endswith(' in ranges') else x_col
            column_label = self.column_labels.get(original_col, original_col)
            unit = self.column_units.get(original_col, '')
            legend_title = f'Legend – {column_label}'

            # Tworzenie legendy
            ax.legend(
                wedges,
                legend_labels,
                title=f'{legend_title} ({unit})' if unit else legend_title,
                loc='center left',
                bbox_to_anchor=(1, 0.5)
            )
            ax.figure.subplots_adjust(left=0.05, right=0.6)
            ax.axis('equal')
            return True

        except Exception as e:
            self.log_area.append(f"Error in pie chart generation: {e}")
            return False

    def render_figure(self, fig):
        """
        Wyświetla wykres na panelu po prawej stronie.
        """
        # Usuwanie wszystkie poprzednie widgety z layoutu
        self.clear_right_panel()

        self.canvas = FigureCanvas(fig)
        self.right_layout.addWidget(self.canvas)

        # Aktualny wykres
        self.current_figure = fig

    def statistics(self, data, column):
        """
        Oblicza statystyki dla wybranej kolumny numerycznej
        """
        values = data[column].dropna()
        if values.empty:
            self.log_area.append('No data available for statistics.')
            return None

        col_label = self.column_labels.get(column, column)

        stats = {
            'number of rows': values.count(),
            'mean': values.mean(),
            'median': values.median(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max(),
            '25%': values.quantile(0.25),
            '75%': values.quantile(0.75)
        }
        self.log_area.append(f'--- Statistics for {col_label} ---')
        self.log_area.append(
            '\n'.join([
                f"Number of rows: {stats['number of rows']}",
                f"Mean: {stats['mean']:.2f}",
                f"Median: {stats['median']:.2f}",
                f"Std: {stats['std']:.2f}",
                f"Min: {stats['min']:.2f}",
                f"Max: {stats['max']:.2f}",
                f"25%: {stats['25%']:.2f}",
                f"75%: {stats['75%']:.2f}"
            ])
        )

        return stats

    # Dostosowywanie widoczności opcji interfejsu w zależności od typu wykresu i danych

    def chart_type_changed(self):
        """
        Reaguje na zmianę typu wykresu — odświeża dane filtrowania i UI.
        """
        # Sprawdzenie, czy dane są załadowane i dostępne
        if not hasattr(self, 'data') or self.data is None or self.data.empty:
            self.log_area.append("Data not loaded yet, skipping chart update.")
            return

        index = self.chart_type_combo.currentIndex()

        # Jeśli wybrano 'Select chart type'
        if index == 0:
            self.log_area.append("No chart type selected.")
            return

        chart_type = self.chart_type_combo.currentText()
        self.log_area.append(f"Chart type changed to: {chart_type}")

        self.update_filter_column_options()
        self.update_ui()

    def update_filter_column_options(self):
        """
        Aktualizuje dostępne kolumny w comboboxie filtrowania w zależności od typu wykresu.
        """
        if self.data is None:
            return

        selected_chart = self.chart_type_combo.currentText()

        # Zapamiętaj aktualnie wybraną kolumnę filtra
        current_filter_col = self.filter_column_combo.currentData()

        self.filter_column_combo.clear()
        self.filter_column_combo.addItem('Not selected', userData=None)

        # Wybór kolumn w zależności od typu wykresu
        if selected_chart == 'Histogram':
            # Dodaj tylko kolumny numeryczne
            columns_to_show = [
                col for col in self.data.columns
                if pd.api.types.is_numeric_dtype(self.data[col])
            ]
        else:
            # Dodaj wszystkie kolumny
            columns_to_show = self.data.columns

        # Dodaj kolumny do comboboxa z uwzględnieniem etykiet
        for col in columns_to_show:
            label = self.column_labels.get(col, col)
            self.filter_column_combo.addItem(label, userData=col)

        # Przywróć wybrany filtr, jeśli jest dostępny
        index = self.filter_column_combo.findData(current_filter_col)
        if index != -1:
            self.filter_column_combo.setCurrentIndex(index)
        else:
            self.filter_column_combo.setCurrentIndex(0)

        self.update_filter_values()

    def update_grouping_column_options(self):
        """
        Aktualizuje listy kolumn dostępnych do grupowania i agregacji po wczytaniu danych.
        Do grupowania: wszystkie kolumny zdefiniowane w słowniku etykiet.
        Do agregacji: tylko kolumny numeryczne zdefiniowane w słowniku etykiet.
        """
        if self.data is None:
            return
        # Reset comboboxów
        self.group_column_combo.clear()
        self.group_column_combo.addItem('Not selected', userData=None)
        self.group_column_combo.setCurrentIndex(0)

        self.agg_column_combo.clear()
        self.agg_column_combo.addItem('Not selected', userData=None)
        self.agg_column_combo.setCurrentIndex(0)

        # Kolumny dostępne w słowniku etykiet
        labeled_cols = [col for col in self.data.columns if col in self.column_labels]

        # Grupowanie – wszystkie kolumny z etykietą
        for col in labeled_cols:
            label = self.column_labels[col]
            self.group_column_combo.addItem(label, userData=col)

        # Agregacja – tylko kolumny numeryczne z etykietą
        numeric_cols = self.data.select_dtypes(include='number').columns
        numeric_labeled_cols = [col for col in numeric_cols if col in labeled_cols]

        for col in numeric_labeled_cols:
            label = self.column_labels[col]
            self.agg_column_combo.addItem(label, userData=col)

    def update_filter_values(self):
        """
        Aktualizuje widżety filtrowania na podstawie wybranej kolumny.
        Dla kolumn numerycznych (int, float) pokazuje spinboxy z zakresem.
        Dla kolumn kategorycznych pokazuje combobox z unikalnymi wartościami.
        """
        filter_col = self.filter_column_combo.currentData()

        if not filter_col or self.data is None:
            # Ukryj wszystkie powiązane widgety jeśli nic nie wybrano
            self.filter_min_spinbox.hide()
            self.filter_max_spinbox.hide()
            self.category_filter_combo.hide()
            self.category_combo_label.hide()
            return

        col_data = self.data[filter_col].dropna()
        # Spinbox dla kolumn z wartościami int
        if pd.api.types.is_integer_dtype(col_data):
            min_val = int(col_data.min())
            max_val = int(col_data.max())
            self.filter_min_spinbox.setDecimals(0)
            self.filter_max_spinbox.setDecimals(0)
            self.filter_min_spinbox.setSingleStep(1)
            self.filter_max_spinbox.setSingleStep(1)

            self.filter_min_spinbox.setRange(min_val, max_val)
            self.filter_max_spinbox.setRange(min_val, max_val)
            self.filter_min_spinbox.setValue(min_val)
            self.filter_max_spinbox.setValue(max_val)

            self.filter_min_spinbox.show()
            self.filter_max_spinbox.show()

            self.category_filter_combo.hide()
            self.category_combo_label.hide()
        # Spinbox dla kolumn z wartościami float
        elif pd.api.types.is_float_dtype(col_data):
            min_val = float(col_data.min())
            max_val = float(col_data.max())
            self.filter_min_spinbox.setDecimals(1)
            self.filter_max_spinbox.setDecimals(1)
            self.filter_min_spinbox.setSingleStep(0.1)
            self.filter_max_spinbox.setSingleStep(0.1)

            self.filter_min_spinbox.setRange(min_val, max_val)
            self.filter_max_spinbox.setRange(min_val, max_val)
            self.filter_min_spinbox.setValue(min_val)
            self.filter_max_spinbox.setValue(max_val)

            self.filter_min_spinbox.show()
            self.filter_max_spinbox.show()

            self.category_filter_combo.hide()
            self.category_combo_label.hide()

        else:
            # Kolumna nie jest numeryczna - combobox z wartościami tekstowymi
            unique_values = sorted(col_data.dropna().unique().tolist())
            self.category_filter_combo.clear()
            self.category_filter_combo.addItem('ALL', userData=None)
            for val in unique_values:
                self.category_filter_combo.addItem(str(val), userData=val)

            self.category_combo_label.show()
            self.category_filter_combo.show()

            self.filter_min_spinbox.hide()
            self.filter_max_spinbox.hide()

    def update_numeric_columns(self):
        """
        Aktualizuje dostępne opcje kolumny agregacji po wybraniu kolumny grupowania.
        """
        selected_group_col = self.group_column_combo.currentData()
        if not selected_group_col or self.data is None:
            return

        labeled_cols = [col for col in self.data.columns if col in self.column_labels]

        # tylko numeryczne kolumny do agregacji
        numeric_cols = self.data.select_dtypes(include='number').columns
        numeric_labeled_cols = [
            col for col in numeric_cols if col in labeled_cols and col != selected_group_col
        ]

        self.agg_column_combo.clear()
        self.agg_column_combo.addItem('Not selected', userData=None)
        for col in numeric_labeled_cols:
            label = self.column_labels[col]
            self.agg_column_combo.addItem(label, userData=col)
        self.agg_column_combo.setCurrentIndex(0)

        # Typy wykresów zależne od kolumny grupowania
        if not pd.api.types.is_numeric_dtype(self.data[selected_group_col]):
            allowed = ['Select chart type', 'Bar Chart', 'Pie Chart']
        else:
            allowed = ['Select chart type', 'Bar Chart', 'Scatter Plot', 'Line Chart', 'Pie Chart', 'Histogram',
                       'Heatmap']

        current_chart = self.chart_type_combo.currentText()
        self.chart_type_combo.blockSignals(True)
        self.chart_type_combo.clear()
        for chart in allowed:
            self.chart_type_combo.addItem(chart)
        if current_chart in allowed:
            self.chart_type_combo.setCurrentText(current_chart)
        else:
            self.chart_type_combo.setCurrentIndex(0)
            self.log_area.append(
                f"Chart type reset: '{current_chart}' not allowed for column '{selected_group_col}'."
            )
        self.chart_type_combo.blockSignals(False)

        self.generate_chart_btn.setVisible(True)
        self.update_ui()

    def update_checkboxes_visibility(self):
        """
        Aktualizuje widoczność, dostępność i tooltipy checkboxów
        na podstawie stanu danych, wybranego wykresu i opcji.
        """
        is_raw = self.raw_data_checkbox.isChecked()
        selected_chart = self.chart_type_combo.currentText()
        selected_group_col = self.group_column_combo.currentData()
        trendline_checked = self.trendline_checkbox.isChecked()

        # blokada raw data dla kolumn nienumerycznych
        if selected_group_col and not pd.api.types.is_numeric_dtype(self.data[selected_group_col]):
            self.raw_data_checkbox.setChecked(False)
            self.raw_data_checkbox.setEnabled(False)
            self.raw_data_checkbox.setToolTip(
                'Raw data mode requires a numeric group column.'
            )
        else:
            self.raw_data_checkbox.setEnabled(True)
            self.raw_data_checkbox.setToolTip('Enable raw data mode.')
        # W trybie raw - gender i bin są zawsze wyłączone i odznaczone
        if is_raw:
            self.gender_checkbox.setChecked(False)
            self.gender_checkbox.setEnabled(False)
            self.gender_checkbox.setToolTip('Not available in raw data mode.')

            self.bin_checkbox.setChecked(False)
            self.bin_checkbox.setEnabled(False)
            self.bin_checkbox.setToolTip('Not available in raw data mode.')
            return

        # Gender checkbox
        gender_available = self.gender_checkbox_available(selected_chart, selected_group_col, is_raw)
        self.gender_checkbox.setEnabled(gender_available)
        if not gender_available:
            self.gender_checkbox.setChecked(False)
            self.gender_checkbox.setToolTip(
                'Not available for the selected chart or when "Gender" is already used or in raw mode.'
            )
        else:
            self.gender_checkbox.setToolTip('Include gender-based comparison in the chart.')

        # Specjalne ustawienia dla Pie Chart
        if selected_chart == 'Pie Chart':
            if selected_group_col and pd.api.types.is_numeric_dtype(self.data[selected_group_col]):
                self.bin_checkbox.setEnabled(True)
                self.bin_checkbox.setChecked(True)
                self.bin_checkbox.setToolTip('Binning enabled for numeric group column in Pie Chart.')
            else:
                self.bin_checkbox.setChecked(False)
                self.bin_checkbox.setEnabled(False)
                self.bin_checkbox.setToolTip('Binning disabled for non-numeric group column in Pie Chart.')

            self.trendline_checkbox.setChecked(False)
            self.trendline_checkbox.setEnabled(False)
            self.trendline_checkbox.setToolTip('Trendline not available for Pie Chart.')

        else:
            # Standardowa logika binowania i trendline
            bin_available = self.binning_available(selected_chart, selected_group_col, is_raw, trendline_checked)
            trendline_available = (selected_chart == 'Scatter Plot') and (not self.bin_checkbox.isChecked())

            if trendline_checked and not bin_available:
                self.bin_checkbox.setChecked(False)

            if self.bin_checkbox.isChecked() and not trendline_available:
                self.trendline_checkbox.setChecked(False)

            self.bin_checkbox.setEnabled(bin_available)
            self.trendline_checkbox.setEnabled(trendline_available)

            self.bin_checkbox.setToolTip(
                'Enable binning for numeric data.'
                if bin_available
                else 'Binning is only available for numeric group columns, '
                     'compatible charts, not in raw mode, and not when trendline is active.'
            )

            self.trendline_checkbox.setToolTip(
                'Show regression trend line on scatter plot.'
                if trendline_available
                else 'Trendline is only available for Scatter Plot and when binning is not selected.'
            )

    def gender_checkbox_available(self, selected_chart, selected_group_col, is_raw):
        """
        Sprawdza, czy checkbox 'gender' może być aktywny
        według warunków dotyczących danych, wykresu i wybranych kolumn.
        """
        if self.data is None:
            return False
        if selected_chart in ['Pie Chart', 'Histogram', 'Heatmap']:
            return False
        if 'Gender' not in self.data.columns:
            return False
        if self.data['Gender'].nunique() < 2:
            return False
        if self.filter_column_combo.currentData() == 'Gender':
            return False
        if selected_group_col is None or selected_group_col == 'Gender':
            return False
        if is_raw:
            return False
        return True

    def binning_available(self, selected_chart, selected_group_col, is_raw, trendline_checked):
        """
        Sprawdza, czy checkbox dzielący na przedziały może być aktywny na podstawie typu wykresu, kolumny i innych opcji
        """
        if selected_chart in ['Heatmap', 'Histogram']:
            return False
        if self.data is None or selected_group_col is None:
            return False
        if not pd.api.types.is_numeric_dtype(self.data[selected_group_col]):
            return False
        if is_raw or trendline_checked:
            return False
        return True


    def initialize_ui_state(self, enabled=False):
        """
        Ustawia stan elementów UI
        """
        # Checkboxy
        self.gender_checkbox.setChecked(False)
        self.gender_checkbox.setEnabled(enabled)

        self.bin_checkbox.setChecked(False)
        self.bin_checkbox.setEnabled(enabled)

        self.trendline_checkbox.setChecked(False)
        self.trendline_checkbox.setEnabled(enabled)
        self.trendline_checkbox.setToolTip('')

        self.raw_data_checkbox.setChecked(False)
        self.raw_data_checkbox.setEnabled(enabled)

        # Filtr
        self.filter_column_combo.setVisible(True)
        self.filter_column_combo.setEnabled(enabled)

        # Grupowanie
        self.grouping_group_box.setVisible(True)
        self.group_column_combo.setVisible(True)
        self.group_column_combo.setEnabled(enabled)

        # Kolumna agregacji
        self.agg_column_combo.setVisible(True)
        self.agg_column_combo.setEnabled(enabled)

        # Przyciski funkcji agregujących
        for btn in self.agg_func_buttons.values():
            btn.setEnabled(enabled)
            btn.setChecked(False)

        # Combobox wyboru typu wykresu
        self.chart_type_combo.setEnabled(enabled)
        self.chart_type_combo.setToolTip('Select chart type.' if enabled else '')

        # Przyciski wczytywania danych zawsze aktywne
        self.csv_button.setEnabled(True)
        self.db_button.setEnabled(True)

        # Pozostałe przyciski
        self.generate_csv_btn.setEnabled(enabled)
        self.generate_chart_btn.setEnabled(enabled)
        self.generate_pdf_btn.setEnabled(enabled)
        self.clear_filters_btn.setEnabled(enabled)

    def update_ui(self):
        """
        Aktualizuje stan i widoczność elementów interfejsu użytkownika
        w zależności od trybu pracy, wybranego typu wykresu oraz funkcji agregującej.
        """

        is_raw = self.raw_data_checkbox.isChecked()
        selected_chart = self.chart_type_combo.currentText()
        agg_func = self.get_selected_agg_func()

        if selected_chart == 'Pie Chart':
            # Dla Pie Chart tylko 'count' aktywne i zaznaczone
            for key, btn in self.agg_func_buttons.items():
                if key == 'count':
                    btn.setEnabled(True)
                    btn.setChecked(True)
                else:
                    btn.setEnabled(False)
                    btn.setChecked(False)

            self.agg_column_combo.setEnabled(False)

            selected_group_col = self.group_column_combo.currentData()
            if selected_group_col and pd.api.types.is_numeric_dtype(self.data[selected_group_col]):
                self.bin_checkbox.setEnabled(True)
                self.bin_checkbox.setChecked(True)
                self.bin_checkbox.setToolTip('Binning enabled for numeric group column in Pie Chart.')
            else:
                self.bin_checkbox.setEnabled(False)
                self.bin_checkbox.setChecked(False)
                self.bin_checkbox.setToolTip('Binning disabled for non-numeric group column in Pie Chart.')

            self.trendline_checkbox.setEnabled(False)
            self.trendline_checkbox.setChecked(False)
            self.trendline_checkbox.setToolTip('Trendline not available for Pie Chart.')

            # Grupowanie aktywne
            self.group_column_combo.setEnabled(True)
            self.grouping_group_box.setVisible(True)

            self.raw_data_checkbox.setChecked(False)
            self.raw_data_checkbox.setEnabled(False)
            return

        if selected_chart in ['Histogram', 'Heatmap']:
            # Filtr aktywny
            self.filter_column_combo.setEnabled(True)

            # Kolumna grupowania i agregacji nieaktywna
            self.group_column_combo.setEnabled(False)
            self.agg_column_combo.setEnabled(False)

            # Przyciski wyłączone i odznaczone
            for btn in self.agg_func_buttons.values():
                btn.setEnabled(False)
                btn.setChecked(False)

            # Raw data wyłączone
            self.raw_data_checkbox.setChecked(False)
            self.raw_data_checkbox.setEnabled(False)
            return

        if is_raw:
            # Wymuszenie Scatter Plot
            if selected_chart != 'Scatter Plot':
                self.chart_type_combo.setCurrentText('Scatter Plot')
            self.chart_type_combo.setEnabled(False)
            self.chart_type_combo.setToolTip('Only Scatter Plot is available in raw data mode.')

            # Grupowanie widoczne i aktywne
            self.grouping_group_box.setVisible(True)
            self.group_column_combo.setVisible(True)
            self.group_column_combo.setEnabled(True)

            # Kolumna agregacji widoczna i aktywna
            self.agg_column_combo.setVisible(True)
            self.agg_column_combo.setEnabled(True)

            # Funkcje agregujące wyłączone i odznaczone
            for btn in self.agg_func_buttons.values():
                btn.setEnabled(False)
                btn.setChecked(False)

            # Trendline dostępny i aktywny
            self.trendline_checkbox.setEnabled(True)
            self.trendline_checkbox.setChecked(False)
            self.trendline_checkbox.setToolTip('Show regression trend line on scatter plot.')
            return

        # Pozostałe przypadki
        if agg_func == 'count':
            self.agg_column_combo.setEnabled(False)
        else:
            self.agg_column_combo.setEnabled(True)

        self.update_checkboxes_visibility()

    def category_combo_changed(self):
        """
        Wyświetla komunikat, jeśli użytkownik zmieni wybraną wartość filtra
        """
        selected = self.category_filter_combo.currentText()
        if selected:
            self.log_area.append(f'Selected value: {selected}')

    # Wczytanie pliku-CSV/SQLite
    def data_load_update(self):
        """
        Aktualizuje UI po wczytaniu danych z pliku CSV/SQLite.
        """
        try:
            if self.data is not None:
                self.log_area.append('Data has been processed.')
                self.update_filter_column_options()
                self.update_grouping_column_options()
                self.update_checkboxes_visibility()

                # Odblokowanie elementów po załadowaniu danych
                self.initialize_ui_state(enabled=True)

        except Exception as e:
            self.log_area.append(f"Error: {e}")
            self.initialize_ui_state(enabled=False)

    def select_csv_file(self):
        """
        Otwiera okno wyboru pliku i wczytuje dane z pliku CSV
        """
        path, _ = QFileDialog.getOpenFileName(self, 'Choose CSV file', '', 'CSV files (*.csv)')
        if not path:
            self.log_area.append('No file selected.')
            return

        self.log_area.append(f'File loaded: {path}')
        self.data = load_data(path)
        self.data_source = os.path.basename(path)
        self.data_load_update()

    def load_data_from_sqlite(self):
        """
        Wczytywanie danych z bazy SQLite
        """
        db_path = 'database.db'  # domyślna baza w katalogu projektu

        # Jeśli domyślna baza nie istnieje, użytkownik może wybrać plik
        if not os.path.exists(db_path):
            self.log_area.append('Default database not found. Please select a database file.')
            db_path, _ = QFileDialog.getOpenFileName(
                self,
                'Select SQLite database',
                '',
                'SQLite Database (*.db)'
            )
            if not db_path:
                self.log_area.append('No database selected. Operation cancelled.')
                return

        try:
            # połączenie z bazą SQLite
            self.db_engine = create_engine(f'sqlite:///{db_path}')

            # wczytanie danych z tabeli 'patients'
            self.data = pd.read_sql_table('patients', self.db_engine)

            # aktualizacja GUI
            self.log_area.append(f'Data loaded from table "patients" in "{os.path.basename(db_path)}".')
            self.data_source = f'{os.path.basename(db_path)} → patients'

            # aktualizacja danych w aplikacji
            self.data_load_update()

        except Exception as e:
            self.log_area.append(f'Error while loading data from database: {str(e)}')

    # Eksport wyników do pliku CSV.
    def generate_report(self, report_format):
        """
        Przygotowuje dane w zależności od trybu i wywołuje odpowiednią funkcję eksportu.
        """
        chart_type = self.chart_type_combo.currentText()
        if self.raw_data_checkbox.isChecked() or chart_type in ['Heatmap', 'Histogram']:
            df = self.get_filtered_data()
        else:
            df = self.prepare_aggregated_data()

        if df is None or df.empty:
            self.log_area.append('No data to export.')
            return

        if report_format == 'csv':
            dialog_title = 'Save CSV Report'
            file_filter = 'CSV Files (*.csv)'
        elif report_format == 'pdf':
            dialog_title = 'Save PDF Report'
            file_filter = 'PDF Files (*.pdf)'
        else:
            self.log_area.append(f'Unsupported report format: {report_format}')
            return

        path, _ = QFileDialog.getSaveFileName(
            self, dialog_title, '', file_filter
        )
        if not path:
            return

        try:
            if report_format == 'csv':
                self.generate_csv_report(df, path)
            else:
                self.generate_pdf_report(df, path)
        except Exception as e:
            self.log_area.append(f'Failed to save {report_format.upper()} report:\n{str(e)}')

    def generate_csv_report(self, df, path):
        """
        Eksportuje dane do pliku CSV.
        """
        try:
            df.to_csv(path, index=False)
            self.log_area.append(f'Report successfully saved as CSV:\n{path}')
        except Exception as e:
            self.log_area.append(f'Failed to save CSV report:\n{str(e)}')

    def generate_pdf_report(self, df, path):
        """
        Eksportuje dane i wykres do pliku PDF.
        """
        try:
            if not (hasattr(self, 'current_figure') and self.current_figure):
                self.log_area.append('No chart available to include in the report.')
                return

            # Zapisz wykres do pliku tymczasowego
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
                self.current_figure.savefig(tmpfile.name, bbox_inches='tight')
                chart_path = tmpfile.name

            c = canvas.Canvas(path, pagesize=A4)
            width, height = A4

            # Strona 1 - wykres i informacje
            c.setFont('Helvetica-Bold', 16)
            c.drawString(50, height - 50, 'Data Report')

            c.setFont('Helvetica', 10)
            info_lines = [
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Data source: {getattr(self, 'data_source', 'Unknown')}",
                f"Filter by: {getattr(self, 'applied_filters', 'None')}",
                f"Group by: {getattr(self, 'grouping_columns', 'None')}",
                f"Aggregation: {getattr(self, 'aggregation_functions', 'None')}"
            ]
            text_y = height - 80
            for line in info_lines:
                c.drawString(50, text_y, line)
                text_y -= 14

            # Wykres
            fig_width, fig_height = self.current_figure.get_size_inches()
            fig_dpi = self.current_figure.get_dpi()
            img_width_px = fig_width * fig_dpi
            img_height_px = fig_height * fig_dpi
            aspect_ratio = img_height_px / img_width_px

            chart_width = width - 100
            chart_height = chart_width * aspect_ratio

            # Pozycja wykresu
            chart_bottom = text_y - chart_height - 10
            c.drawImage(chart_path, 50, chart_bottom, width=chart_width, height=chart_height)

            stats = getattr(self, 'stats', None)
            if stats:
                c.setFont('Helvetica-Bold', 12)

                margin = 10
                header_y = chart_bottom - margin - 12
                c.drawString(50, header_y, 'Summary statistics:')

                c.setFont('Helvetica', 10)
                line_height = 14
                y_stat = header_y - line_height
                for key, value in stats.items():
                    text = f"{key.capitalize():<10}: {value:.2f}" if isinstance(value,
                                                                                (int, float)) else f"{key.capitalize()}: {value}"
                    c.drawString(70, y_stat, text)
                    y_stat -= line_height

            # Numeracja strony
            c.setFont('Helvetica', 9)
            c.drawRightString(width - 50, 30, "Page 1")
            c.showPage()

            # Strony z tabelą danych
            rows_per_page = 30
            data = [df.columns.tolist()] + df.values.tolist()
            total_pages = math.ceil(len(data[1:]) / rows_per_page)

            for page in range(total_pages):
                c.setFont('Helvetica-Bold', 16)
                c.drawString(50, height - 50, 'Dataset Table')

                c.setFont('Helvetica', 9)
                c.drawRightString(width - 50, 30, f"Page {page + 2}")

                start = page * rows_per_page + 1
                end = start + rows_per_page
                page_data = [data[0]] + data[start:end]

                table = Table(page_data, repeatRows=1)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ]))

                table_width, table_height = table.wrapOn(c, width - 100, height)
                table.drawOn(c, 50, height - 80 - table_height)

                c.showPage()

            c.save()
            os.remove(chart_path)
            self.log_area.append(f'PDF report saved successfully:\n{path}')

        except Exception as e:
            self.log_area.append(f'Failed to generate PDF report:\n{str(e)}')


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
