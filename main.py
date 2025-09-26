from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTextEdit, QGroupBox, QComboBox,
    QRadioButton, QButtonGroup, QGridLayout, QDoubleSpinBox, QCheckBox)
from matplotlib import pyplot as plt

from pandas.api.types import CategoricalDtype

import sys
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# Wczytanie danych z pliku CSV.
def load_data(path):
    try:
        data = pd.read_csv(path, delimiter=',')
        return data
    except FileNotFoundError:
        print('File not found')


# GUI: wybór pliku i filtrów.
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.data = None
        self.setWindowTitle('Data analysis')
        self.setGeometry(100, 100, 1200, 800)

        self.main_layout = QVBoxLayout()
        self.center_layout = QHBoxLayout()

        # Lewa część: przyciski filtrowania
        self.left_box = QGroupBox('🔍 Filters and grouping')
        self.left_layout = QVBoxLayout()

        self.left_box.setLayout(self.left_layout)
        self.left_box.setMinimumWidth(300)

        # Prawa część: Wizualizacje
        self.right_box = QGroupBox('📊 Charts')
        self.right_layout = QVBoxLayout()
        self.right_box.setLayout(self.right_layout)

        self.center_layout.addWidget(self.left_box, 1)
        self.center_layout.addWidget(self.right_box, 2)

        # Dolna część: Logi
        self.bottom_box = QGroupBox('📝 Logs')
        self.bottom_layout = QVBoxLayout()
        self.label = QLabel('No file selected')
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFixedHeight(150)

        self.bottom_layout.addWidget(self.label)
        self.bottom_layout.addWidget(self.log_area)
        self.bottom_box.setLayout(self.bottom_layout)

        # Przycisk wyboru pliku
        self.button = QPushButton('📂 Select CSV File')
        self.button.clicked.connect(self.on_file_select)

        # === LEWA CZĘŚĆ: FILTRY I GRUPOWANIE ===

        self.column_labels = {'Gender': 'Gender',
                              'AGE': 'Age',
                              'Urea': 'Urea level in blood',
                              'Cr': 'Creatinine ratio',
                              'HbA1c': 'HbA1c level',
                              'Chol': 'Cholesterol',
                              'TG': 'Triglycerides',
                              'HDL': 'HDL',
                              'LDL': 'LDL',
                              'VLDL': 'VLDL',
                              'BMI': 'BMI',
                              'CLASS': 'Classification'}
        # Opcje filtrowania
        self.left_layout.addWidget(QLabel('Filter by'))

        self.filter_column_combo = QComboBox()
        self.filter_column_combo.currentIndexChanged.connect(self.update_filter_values)
        self.left_layout.addWidget(self.filter_column_combo)

        # Dla kolumn z wartościami numerycznymi
        self.filter_min_spinbox = QDoubleSpinBox()
        self.filter_max_spinbox = QDoubleSpinBox()

        # Domyślny zakres na start, zostanie potem zmieniony dynamicznie
        self.filter_min_spinbox.setRange(0, 1000)
        self.filter_max_spinbox.setRange(0, 1000)

        self.filter_min_spinbox.setPrefix('From: ')
        self.filter_max_spinbox.setPrefix('To:')

        self.filter_min_spinbox.setDecimals(1)  # wyświetla 1 miejsce po przecinku
        self.filter_max_spinbox.setDecimals(1)

        spinbox_layout = QHBoxLayout()
        spinbox_layout.addWidget(self.filter_min_spinbox)
        spinbox_layout.addWidget(self.filter_max_spinbox)

        self.left_layout.addLayout(spinbox_layout)

        self.filter_min_spinbox.hide()
        self.filter_max_spinbox.hide()

        # Dla kolumn z wartościami nienumerycznymi
        self.category_combo_label = QLabel('Select value:')
        self.left_layout.addWidget(self.category_combo_label)
        self.category_combo_label.hide()

        self.category_filter_combo = QComboBox()
        self.category_filter_combo.currentIndexChanged.connect(self.on_category_combo_changed)
        self.left_layout.addWidget(self.category_filter_combo)
        self.category_filter_combo.hide()

        # Grupowanie: wybór kolumn

        self.grouping_section_label = QLabel('Group by:')
        self.left_layout.addWidget(self.grouping_section_label)

        self.group_column_combo = QComboBox()
        self.group_column_combo.currentIndexChanged.connect(self.update_numeric_columns)

        self.left_layout.addWidget(self.group_column_combo)

        self.agg_column_combo = QComboBox()
        self.agg_column_combo.setVisible(False)  # na początku ukryty
        self.left_layout.addWidget(self.agg_column_combo)

        self.agg_func_group = QButtonGroup(self)
        self.agg_func_buttons = {}

        agg_functions = {
            'mean': 'Mean',
            'median': 'Median',
            'count': 'Number of patients',
            'min': 'Min',
            'max': 'Max'
        }
        # Wyświetlanie przycisków w dwóch kolumnach
        agg_func_layout = QGridLayout()
        row = 0
        col = 0

        for i, (func_key, func_label) in enumerate(agg_functions.items()):
            btn = QRadioButton(func_label)
            btn.toggled.connect(self.on_agg_func_changed)
            agg_func_layout.addWidget(btn, row, col)
            self.agg_func_group.addButton(btn)
            self.agg_func_buttons[func_key] = btn

            # Przełączanie kolumny co drugi przycisk
            col += 1
            if col > 1:
                col = 0
                row += 1

        self.left_layout.addLayout(agg_func_layout)

        # Checkbox do wyświetlania różnic pomiędzy kobietami a mężczyznami

        self.gender_checkbox = QCheckBox('🎨 Show gender differences')
        self.gender_checkbox.setChecked(False)
        self.left_layout.addWidget(self.gender_checkbox)
        self.group_column_combo.currentIndexChanged.connect(self.update_gender_checkbox_visibility)
        self.filter_column_combo.currentIndexChanged.connect(self.update_gender_checkbox_visibility)

        self.gender_checkbox.hide()

        # Checkbox do wyświetlania danych w przedziałach
        self.bin_checkbox = QCheckBox('📦 Show data in ranges')
        self.bin_checkbox.setChecked(True)
        self.left_layout.addWidget(self.bin_checkbox)

        # Przyciski w dolnej części okna
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItem('Select chart type')
        self.chart_type_combo.addItems([
            'Bar Chart',
            'Pie Chart',
            'Line Chart',
            'Scatter Plot',
            'Histogram',
            'Heatmap'
        ])

        self.left_layout.addWidget(self.chart_type_combo)

        self.chart_type_combo.currentIndexChanged.connect(self.on_chart_type_changed)
        self.on_chart_type_changed()  # wywołaj raz na start, żeby stan UI był poprawny

        self.group_execute_btn = QPushButton('📈 Generate chart')
        self.group_execute_btn.clicked.connect(self.update_chart)
        self.group_execute_btn.setVisible(False)
        self.left_layout.addWidget(self.group_execute_btn)

        self.generate_report_btn = QPushButton('📄 Generate report')
        self.generate_report_btn.clicked.connect(self.generate_report)
        self.generate_report_btn.setVisible(False)  # Ukryty na start
        self.left_layout.addWidget(self.generate_report_btn)

        self.clear_filters_btn = QPushButton('❌ Reset filters and grouping')
        self.clear_filters_btn.clicked.connect(self.on_clear_filters)
        self.clear_filters_btn.setVisible(False)  # Ukryty na start
        self.left_layout.addWidget(self.clear_filters_btn)

        # Layout końcowy
        self.main_layout.addWidget(self.button)
        self.main_layout.addLayout(self.center_layout, 2)
        self.main_layout.addWidget(self.bottom_box, 1)

        self.setLayout(self.main_layout)

    def on_clear_filters(self):
        """
        Resetuje wybrane filtry i grupowanie – przywraca domyślne ustawienia.
        """
        # Reset filtra
        self.filter_column_combo.setCurrentIndex(0)
        self.filter_min_spinbox.setValue(self.filter_min_spinbox.minimum())
        self.filter_max_spinbox.setValue(self.filter_max_spinbox.maximum())
        self.filter_min_spinbox.hide()
        self.filter_max_spinbox.hide()

        self.category_filter_combo.setCurrentIndex(0)
        self.category_filter_combo.hide()
        self.category_combo_label.hide()

        # Reset grupowania
        self.group_column_combo.setCurrentIndex(0)
        self.agg_column_combo.setCurrentIndex(0)
        self.agg_column_combo.hide()

        # Reset funkcji agregujących
        self.agg_func_group.setExclusive(False)
        for btn in self.agg_func_buttons.values():
            btn.setChecked(False)
        self.agg_func_group.setExclusive(True)

        # Ukryj checkboxy
        self.gender_checkbox.setVisible(False)
        self.gender_checkbox.setChecked(False)

        self.bin_checkbox.setVisible(True)
        self.bin_checkbox.setChecked(True)

        self.log_area.append('Filters and grouping have been reset.')

    # Funkcje dotyczące grupowania i filtrowania danych

    def get_grouped_data(self):
        """
        Zwraca dane po przefiltrowaniu i ewentualnym grupowaniu/agregacji,
        zgodnie z aktualnymi ustawieniami interfejsu.
        """
        df = self.get_filtered_data()
        if df is None or df.empty:
            return None

        selected_chart = self.chart_type_combo.currentText()

        if selected_chart in ['Histogram', 'Heatmap']:
            return df  # Te wykresy nie wymagają agregacji, zwracamy surowe dane

        # Pozostałe wykresy – grupowanie i agregacja
        group_col = self.group_column_combo.currentData()
        agg_col = self.agg_column_combo.currentData()

        selected_func = None
        for func_key, btn in self.agg_func_buttons.items():
            if btn.isChecked():
                selected_func = func_key
                break

        if not group_col or not agg_col or not selected_func:
            return df  # Brakuje danych do agregacji – zwracamy przefiltrowany DataFrame

        # Grupowanie, z opcjonalnym podziałem na płeć
        grouping_cols = [group_col]

        if self.gender_checkbox.isChecked() and 'Gender' in df.columns:
            grouping_cols.append('Gender')

        try:
            grouped_df = df.groupby(grouping_cols)[agg_col].agg(selected_func).reset_index()
        except Exception as e:
            self.log_area.append(f'Error while grouping: {e}')
            return None

        return grouped_df

    def aggregate_data(self, df, group_keys, agg_func):
        """
        Grupuje dane i wykonuje agregację zgodnie z wybraną funkcją.
        """
        if agg_func == "count":
            grouped = df.groupby(group_keys, observed=True).size().reset_index(name='Number of patients')
            return grouped, 'Number of patients', 'Number of patients'
        else:
            agg_col = self.agg_column_combo.currentData()
            if not agg_col:
                self.log_area.append('Select a column to aggregate.')
                raise ValueError('Aggregation column not selected')
            grouped = df.groupby(group_keys, observed=True).agg({agg_col: agg_func}).reset_index()
            y_label = self.column_labels.get(agg_col, agg_col)
            return grouped, agg_col, y_label

    def get_filtered_data(self):
        """
        Filtrowanie danych według wybranych kryteriów.
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
            return self.data[
                (self.data[filter_col] >= min_val) & (self.data[filter_col] <= max_val)
                ].copy()
        # Filtrowanie w kolumnach z wartościami nienumerycznymi
        elif self.category_filter_combo.isVisible():
            selected_val = self.category_filter_combo.currentData()
            if selected_val is not None:  # Wartość inna niż ALL
                return self.data[self.data[filter_col].astype(str) == str(selected_val)].copy()

        return self.data.copy()

    def bin_numeric_column(self, series, column_name=None):
        """
        Dzieli kolumnę numeryczną na przedziały.
        """
        try:
            if column_name == 'BMI':
                bins = [0, 18.5, 24.9, 29.9, 34.9, 39.9, 100]
                labels = ['Underweight', 'Normal weight', 'Overweight', 'Obesity Class I',
                          'Obesity Class II', 'Obesity Class III']
                cat_type = CategoricalDtype(categories=labels, ordered=True)
                return pd.cut(series, bins=bins, labels=labels).astype(cat_type)

            elif column_name == 'AGE':
                bins = [0, 20, 30, 40, 50, 60, 70, 80]
                labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
                cat_type = CategoricalDtype(categories=labels, ordered=True)
                return pd.cut(series, bins=bins, labels=labels).astype(cat_type)

            elif column_name == 'Chol':
                bins = [0, 4.99, 5.99, 6.99, 12]
                labels = ['<5', '5-5.99', '6-6.99', '7≤ ']
                cat_type = CategoricalDtype(categories=labels, ordered=True)
                return pd.cut(series, bins=bins, labels=labels).astype(cat_type)

            elif column_name == 'Cr':
                bins = [0, 49.99, 99.99, 149.99, 199.99, 800]
                labels = ['<50', '50-99.99', '100-149.99', '150-199.99', '200≤ ']
                cat_type = CategoricalDtype(categories=labels, ordered=True)
                return pd.cut(series, bins=bins, labels=labels).astype(cat_type)

            elif column_name == 'HbA1c':
                bins = [0, 5.6, 6.49, 16]
                labels = ['<5.70', '5.7-6.49', '6.5≤']
                cat_type = CategoricalDtype(categories=labels, ordered=True)
                return pd.cut(series, bins=bins, labels=labels).astype(cat_type)

            elif column_name == 'TG':
                bins = [0, 1.69, 2.99, 3.99, 4.99, 14]
                labels = ['<1.70', '1.70-2.99', '3-3.99', '4-4.99', '5≤']
                cat_type = CategoricalDtype(categories=labels, ordered=True)
                return pd.cut(series, bins=bins, labels=labels).astype(cat_type)
            elif column_name == 'HDL':
                bins = [0, 1.19, 2.19, 3.99, 4.99, 10]
                labels = ['<1.2', '1.2-2.19', '2.20-3.99', '4-4.99', '5≤']
                cat_type = CategoricalDtype(categories=labels, ordered=True)
                return pd.cut(series, bins=bins, labels=labels).astype(cat_type)
            elif column_name == 'LDL':
                bins = [0, 2.99, 3.99, 4.99, 5.99, 10]
                labels = ['<3', '3-3.99', '4-4.99', '5-5.99', '6≤']
                cat_type = CategoricalDtype(categories=labels, ordered=True)
                return pd.cut(series, bins=bins, labels=labels).astype(cat_type)
            elif column_name == 'VLDL':
                bins = [0, 0.79, 2.99, 4.99, 35]
                labels = ['<0.8', '0.8-2.99', '3-4.99', '5≤']
                cat_type = CategoricalDtype(categories=labels, ordered=True)
                return pd.cut(series, bins=bins, labels=labels).astype(cat_type)
            elif column_name == 'Urea':
                bins = [0, 1.99, 2.99, 3.99, 4.99, 5.99, 40]
                labels = ['<2', '2-2.99', '3-3.99', '4-4.99', '5-5.99', '6≤']
                cat_type = CategoricalDtype(categories=labels, ordered=True)
                return pd.cut(series, bins=bins, labels=labels).astype(cat_type)
            else:
                # Dla pozostałych: niezmienione
                return series

        except Exception as e:
            self.log_area.append(f"Error while grouping values '{column_name}': {e}")
            return series

    # Funkcje generujące wykresy
    def update_chart(self):
        """Aktualizuje wykres na podstawie bieżących ustawień (filtry, grupowanie, typ wykresu)."""
        self.clear_right_panel()
        df = self.get_grouped_data()
        if df is None or df.empty:
            self.log_area.append('No data to display.')
            return

        selected_chart = self.chart_type_combo.currentText()

        if selected_chart == 'Histogram':
            filter_col = self.filter_column_combo.currentData()
            if not filter_col:
                self.log_area.append('No filter column selected for histogram.')
                return

            self.generate_hist(df, selected_chart, column=filter_col)

        elif selected_chart == 'Heatmap':
            self.generate_heatmap(df, selected_chart)

        else:
            self.handle_other_charts(df)

    def handle_other_charts(self, df):
        """
        Obsługuje generowanie wykresów z grupowaniem i funkcją agregującą (np. count, sum, mean).
        Uwzględnia podział na płeć i opcjonalne binowanie kolumn numerycznych.
        """

        group_col = self.group_column_combo.currentData()
        if not group_col:
            self.log_area.append('Select a grouping column.')
            return

        agg_func = None
        for func_key, btn in self.agg_func_buttons.items():
            if btn.isChecked():
                agg_func = func_key
                break

        if not agg_func:
            self.log_area.append('Select an aggregate function.')
            return

        use_gender = self.gender_checkbox.isChecked()
        group_keys = [group_col, 'Gender'] if use_gender else [group_col]  # klucze grupowania

        # Obsługa binowania
        if self.bin_checkbox.isChecked() and pd.api.types.is_numeric_dtype(df[group_col]):
            binned_col = self.bin_numeric_column(df[group_col], column_name=group_col)
            df['binned_group'] = binned_col
            group_keys = ['binned_group', 'Gender'] if use_gender else ['binned_group']

        try:
            grouped, y_col, y_label = self.aggregate_data(df, group_keys, agg_func)

            grouped = grouped.sort_values(by=group_keys)
            agg_label = self.agg_func_buttons[agg_func].text()
            x_label = self.column_labels.get(group_col, group_col)
            title = f'{agg_label} by {x_label.lower()}'
            y_axis_label = y_label

            x_col_name = group_keys[0] if 'binned_group' not in df.columns else 'binned_group'
            hue_col = 'Gender' if use_gender else None

            self.generate_chart(
                data=grouped,
                x_col=x_col_name,
                y_col=y_col,
                x_label=x_label,
                y_label=y_axis_label,
                title=title,
                hue_col=hue_col,
                agg_func=agg_func,
                binning_enabled=self.bin_checkbox.isChecked()
            )

        except Exception as e:
            self.log_area.append(f'Error while grouping: {e}')

    def generate_hist(self, data, selected_chart, column):
        """
        Generuje histogram
        """

        if selected_chart == 'Histogram':
            if not column or column not in data.columns:
                self.log_area.append('Invalid column selected for histogram.')
                return

            if not pd.api.types.is_numeric_dtype(data[column]):
                self.log_area.append(f'Column "{column}" is not numeric and cannot be used for histogram.')
                return

            values = data[column].dropna()
            if values.empty:
                self.log_area.append('No data available for histogram.')
                return

        fig = Figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        col_name = self.column_labels.get(column, column)

        try:
            if selected_chart == 'Histogram':
                # Sprawdź, czy podano kolumnę
                if not column or column not in data.columns:
                    self.log_area.append('Invalid column selected for histogram.')
                    return

                values = data[column].dropna()
                if values.empty:
                    self.log_area.append('No data available for histogram.')
                    return

                # Oblicz statystyki
                mean_val = values.mean()
                median_val = values.median()
                std_val = values.std()
                min_val = values.min()
                max_val = values.max()

                # Rysuj histogram
                ax.hist(values, bins=30, color='skyblue', edgecolor='black')

                # Dodaj linie pionowe
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f"Mean: {mean_val:.2f}")
                ax.axvline(median_val, color='green', linestyle='-.', linewidth=2, label=f"Median: {median_val:.2f}")

                # Dodaj tytuł, etykiety i legendę
                ax.set_title(f'Histogram of {col_name.lower()}')
                ax.set_xlabel(col_name)
                ax.set_ylabel('Frequency')

                ax.legend(loc='upper right', title=f'Std: {std_val:.2f}'
                                                   f'\nMax: {max_val:.2f}'
                                                   f'\nMin:{min_val:.2f}')
                ax.grid(True)
                self.render_figure(fig)
                self.log_area.append(f'{selected_chart} has been generated.')
        except Exception as e:
            self.log_area.append(f'Error generating {selected_chart}: {e}')

    def generate_heatmap(self, data, selected_chart):
        """
        Generuje heatmapę
        """
        fig = Figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        try:
            if selected_chart == 'Heatmap':
                numeric_data = data.select_dtypes(include='number')
                if numeric_data.shape[1] < 2:
                    self.log_area.append('Not enough numeric columns for heatmap.')
                    return

                corr = numeric_data.corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
                ax.set_title('Heatmap of biochemical parameters')

            else:
                self.log_area.append(f"Unsupported distribution chart type: {selected_chart}")
                return

            self.render_figure(fig)
            self.log_area.append(f'{selected_chart} has been generated.')
        except Exception as e:
            self.log_area.append(f'Error generating {selected_chart}: {e}')

    def generate_chart(self, data, x_col, y_col, x_label, y_label, title, hue_col=None, agg_func=None,
                       binning_enabled=False):
        """
        Generowanie pozostałych wykresów na podstawie wybranego typu.
        """
        selected_chart = self.chart_type_combo.currentText()

        if selected_chart == 'Select chart type':
            self.log_area.append('Please select a chart type before generating.')
            return

        fig = Figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        try:
            plot_dispatch = {
                'Bar Chart': self.plot_bar_chart,
                'Line Chart': self.plot_line_chart,
                'Scatter Plot': self.plot_scatter_plot,
                'Pie Chart': self.plot_pie_chart,
            }

            plot_func = plot_dispatch.get(selected_chart)

            if not plot_func:
                return self.log_area.append(f'Unsupported chart type: {selected_chart}')

            plot_func(ax, data, x_col, y_col, hue_col, agg_func, binning_enabled)

            # Ustawienia końcowe (wykresy inne niż Pie Chart)
            if selected_chart != 'Pie Chart':
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.grid(True)

            full_title = f'{title} (gender differences)' if hue_col else title
            ax.set_title(full_title)

            self.render_figure(fig)
            self.log_area.append('The chart has been generated.')

        except Exception as e:
            self.log_area.append(f"Error while generating chart: {e}")

    def plot_bar_chart(self, ax, data, x_col, y_col, hue_col, agg_func=None, binning_enabled=False):
        palette = {'F': '#fe46a5', 'M': '#82cafc'}
        if hue_col:
            sns.barplot(data=data, x=x_col, y=y_col, hue=hue_col, palette=palette, ax=ax)
        else:
            sns.barplot(data=data, x=x_col, y=y_col, ax=ax)
        return True

    def plot_line_chart(self, ax, data, x_col, y_col, hue_col, agg_func, binning_enabled):
        palette = {'F': '#fe46a5', 'M': '#82cafc'}
        if hue_col:
            sns.lineplot(data=data, x=x_col, y=y_col, hue=hue_col, palette=palette, ax=ax)
        else:
            sns.lineplot(data=data, x=x_col, y=y_col, ax=ax)
        return True

    def plot_scatter_plot(self, ax, data, x_col, y_col, hue_col, agg_func, binning_enabled):
        palette = {'F': '#fe46a5', 'M': '#82cafc'}
        if hue_col:
            sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col, palette=palette, ax=ax)
        else:
            sns.scatterplot(data=data, x=x_col, y=y_col, ax=ax)
        return True

    def plot_pie_chart(self, ax, data, x_col, y_col, hue_col, agg_func, binning_enabled):
        if hue_col:
            self.log_area.append('Pie Chart does not support hue (gender-based) data.')
            return False

        if x_col != 'CLASS' and not (binning_enabled or agg_func == 'count'):
            self.log_area.append('Pie Chart is available only for "CLASS" or when binning is '
                                 'enabled or aggregation is "Number of patietns.')
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
                pctdistance=1.1
            )

            if x_col == 'CLASS':
                class_labels = {'Y': 'Diabetic', 'N': 'Non-Diabetic', 'P': 'Prediction of diabetes'}
                legend_labels = [class_labels.get(str(label), str(label)) for label in labels]
            else:
                legend_labels = list(labels)

            ax.legend(wedges, legend_labels, title='Legend', loc='best')
            ax.axis('equal')

            return True

        except Exception as e:
            self.log_area.append(f"Error in pie chart generation: {e}")
            return False

    def render_figure(self, fig):
        """
        Wyświetla wykres na panelu po prawej stronie.
        """
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        self.right_layout.addWidget(canvas)

    # Dostosowywanie widoczności i dostępnych opcji interfejsu w zależności od typu wykresu i danych
    def on_chart_type_changed(self):
        selected_chart = self.chart_type_combo.currentText()
        disable_grouping = selected_chart in ["Histogram", "Heatmap"]

        # Ukryj/pokaż"Group by:"
        self.grouping_section_label.setVisible(not disable_grouping)

        # Ukryj/pokaż combobox do kolumny grupowania
        self.group_column_combo.setVisible(not disable_grouping)

        # Ukryj/pokaż combobox do kolumny agregacji
        self.agg_column_combo.setVisible(not disable_grouping)

        # Ukryj/pokaż przyciski funkcji agregujących
        self.agg_func_group.setExclusive(False)
        for btn in self.agg_func_buttons.values():
            btn.setVisible(not disable_grouping)
        self.agg_func_group.setExclusive(True)

        # Checkboxy jeśli masz np. płeć, binowanie itp.
        self.gender_checkbox.setVisible(not disable_grouping)
        self.bin_checkbox.setVisible(not disable_grouping)
        # Aktualizacja listy kolumn filtrowania (na podstawie wybranego wykresu)
        self.update_filter_column_options()

    def on_agg_func_changed(self):
        selected_func = None
        for func_key, btn in self.agg_func_buttons.items():
            if btn.isChecked():
                selected_func = func_key
                break

        # Ukryj pole, jeśli nie wybrano funkcji lub wybrano count
        if selected_func in [None, 'count']:
            self.agg_column_combo.setVisible(False)
        else:
            self.agg_column_combo.setVisible(True)

    def update_filter_column_options(self):
        if self.data is None:
            return

        selected_chart = self.chart_type_combo.currentText()
        self.filter_column_combo.clear()
        self.filter_column_combo.addItem('Not selected', userData=None)

        if selected_chart == "Histogram":
            # Dodaj tylko kolumny numeryczne
            numeric_columns = [col for col in self.data.columns if pd.api.types.is_numeric_dtype(self.data[col])]
            for col in numeric_columns:
                label = self.column_labels.get(col, col)
                self.filter_column_combo.addItem(label, userData=col)
        else:
            # Dodaj wszystkie kolumny
            for col in self.data.columns:
                label = self.column_labels.get(col, col)
                self.filter_column_combo.addItem(label, userData=col)

        self.update_filter_values()

    def update_grouping_column_options(self):
        """
        Aktualizuje kolumny dostępne do grupowania i agregacji po wczytaniu pliku.
        """
        if self.data is None:
            return

        self.group_column_combo.clear()
        self.group_column_combo.addItem('Not selected', userData=None)

        self.agg_column_combo.clear()
        self.agg_column_combo.addItem('Not selected', userData=None)

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
        """Aktualizuje opcje filtrowania"""
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
        Aktualizuje opcje kolumny agregacji po wybraniu kolumny grupowania.
        Ukrywa lub blokuje niepasujące funkcje agregujące.
        """
        selected_group_col = self.group_column_combo.currentData()
        if not selected_group_col or self.data is None:
            return

        # Kolumny z etykietą
        labeled_cols = [col for col in self.data.columns if col in self.column_labels]

        # Kolumny numeryczne z etykietą (dla agregacji)
        numeric_cols = self.data.select_dtypes(include='number').columns
        numeric_labeled_cols = [col for col in numeric_cols if col in labeled_cols]

        # Usuń kolumnę grupującą z kolumn do agregacji
        numeric_labeled_cols = [col for col in numeric_labeled_cols if col != selected_group_col]

        self.agg_column_combo.clear()
        self.agg_column_combo.addItem('Not selected', userData=None)

        for col in numeric_labeled_cols:
            label = self.column_labels[col]
            self.agg_column_combo.addItem(label, userData=col)

        # Obsługa dostępności funkcji agregujących
        if selected_group_col in numeric_cols:
            for func_key, btn in self.agg_func_buttons.items():
                btn.setEnabled(True)
        else:
            for func_key, btn in self.agg_func_buttons.items():
                if func_key == 'count':
                    btn.setEnabled(True)
                    btn.setChecked(True)
                else:
                    btn.setEnabled(False)

        # Zsynchronizuj widoczność pola agregacji z funkcją agregującą
        self.on_agg_func_changed()

        # Resetuj wybór kolumny agregacji (na „Not selected”)
        self.agg_column_combo.setCurrentIndex(0)

        # Pokaż przycisk wykonania grupowania
        self.group_execute_btn.setVisible(True)

        # Pokaż lub ukryj checkbox "Show data in ranges"
        if pd.api.types.is_numeric_dtype(self.data[selected_group_col]):
            self.bin_checkbox.setVisible(True)
        else:
            self.bin_checkbox.setVisible(False)

    def update_gender_checkbox_visibility(self):
        """
        Pokazuje checkbox do porównania płci tylko jeśli
        użytkownik NIE grupuje ani NIE filtruje po kolumnie 'Gender'.
        """
        filter_col = self.filter_column_combo.currentData()
        group_col = self.group_column_combo.currentData()

        show_gender_checkbox = (
                self.data is not None and
                'Gender' in self.data.columns and
                self.data['Gender'].nunique() >= 2 and
                filter_col != 'Gender' and
                group_col != 'Gender'
        )

        self.gender_checkbox.setVisible(show_gender_checkbox)

        # Jeśli checkbox nie powinien być widoczny, to go też odznacz
        if not show_gender_checkbox:
            self.gender_checkbox.setChecked(False)

    def on_category_combo_changed(self):
        """
        Wyświetla komunikat, jeśli użytkownik zmieni wybraną wartość filtra
        """
        selected = self.category_filter_combo.currentText()
        if selected:
            self.log_area.append(f'Selected value: {selected}')

    def clear_right_panel(self):
        for i in reversed(range(self.right_layout.count())):
            widget_to_remove = self.right_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

    # Wczytanie pliku CSV
    def on_file_select(self):
        """
        Wybór pliku
        """
        path, _ = QFileDialog.getOpenFileName(self, 'Choose CSV file', '', 'CSV files (*.csv)')
        if not path:
            self.log_area.append('No file selected.')
            return

        self.label.setText(f'Selected: {path}')
        self.log_area.append(f'File loaded: {path}')
        self.data = load_data(path)

        if self.data is not None:
            self.log_area.append('Data has been processed.')
            self.update_filter_column_options()
            self.update_grouping_column_options()
            # Pokaż przyciski po załadowaniu danych
            self.group_execute_btn.setVisible(True)
            self.generate_report_btn.setVisible(True)
            self.clear_filters_btn.setVisible(True)
        else:
            self.log_area.append('An error occurred while processing data.')
            # Ukryj przyciski, jeśli dane się nie wczytały
            self.group_execute_btn.setVisible(False)
            self.generate_report_btn.setVisible(False)
            self.clear_filters_btn.setVisible(False)

    # Eksport wyników do pliku CSV.
    def generate_report(self):
        """
            Generuje raport CSV na podstawie danych aktualnie widocznych na wykresie
            (z uwzględnieniem filtrowania i grupowania).
            """
        df = self.get_grouped_data()

        if df is None or df.empty:
            self.log_area.append('No data to export.')
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            'Save CSV Report',
            '',
            'CSV Files (*.csv);;All Files (*)',
        )

        if file_path:
            try:
                df.to_csv(file_path, index=False)
                self.log_area.append(f'The report was successfully saved:\n{file_path}')
            except Exception as e:
                self.log_area.append(f'Failed to save the report:\n{str(e)}')


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
