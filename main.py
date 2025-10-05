from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTextEdit, QGroupBox, QComboBox,
    QRadioButton, QButtonGroup, QGridLayout, QDoubleSpinBox, QCheckBox)
import sys
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
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
        self.current_plot_data = None
        self.setWindowTitle('Data analysis')
        self.setGeometry(100, 100, 1200, 800)

        self.main_layout = QVBoxLayout()
        self.center_layout = QHBoxLayout()

        # Lewa część: filtrowanie i grupowanie
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
        self.button.clicked.connect(self.select_file)

        # LEWA CZĘŚĆ: FILTRY I GRUPOWANIE ===

        self.column_labels = {'Gender': 'Gender',
                              'AGE': 'Age',
                              'Urea': 'Urea level in blood',
                              'Cr': 'Creatinine',
                              'HbA1c': 'HbA1c level',
                              'Chol': 'Cholesterol',
                              'TG': 'Triglycerides',
                              'HDL': 'HDL',
                              'LDL': 'LDL',
                              'VLDL': 'VLDL',
                              'BMI': 'BMI',
                              'CLASS': 'Diabetes risk classification'}
        self.column_units = {'Urea': 'mmol/L',
                             'AGE': 'years',
                             'Cr': 'μmol/L',
                             'HbA1c': '%',
                             'Chol': 'mmol/L',
                             'TG': 'mmol/L',
                             'HDL': 'mmol/L',
                             'LDL': 'mmol/l',
                             'VLDL': 'mmol/L'}
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
        self.category_filter_combo.currentIndexChanged.connect(self.category_combo_changed)
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
            btn.toggled.connect(self.agg_func_changed)
            agg_func_layout.addWidget(btn, row, col)
            self.agg_func_group.addButton(btn)
            self.agg_func_buttons[func_key] = btn

            # Przełączanie kolumny co drugi przycisk
            col += 1
            if col > 1:
                col = 0
                row += 1

        self.left_layout.addLayout(agg_func_layout)
        # Przyciski w dolnej części okna

        # Checkbox do wyświetlania różnic pomiędzy kobietami a mężczyznami

        self.gender_checkbox = QCheckBox('🎨 Show gender differences')
        self.gender_checkbox.setChecked(False)
        self.left_layout.addWidget(self.gender_checkbox)
        self.filter_column_combo.currentIndexChanged.connect(self.update_checkboxes_visibility)
        self.group_column_combo.currentIndexChanged.connect(self.update_checkboxes_visibility)
        self.gender_checkbox.setVisible(False)

        # Checkbox do wyświetlania danych w przedziałach
        self.bin_checkbox = QCheckBox('📦 Show data in ranges')
        self.bin_checkbox.setChecked(True)
        self.left_layout.addWidget(self.bin_checkbox)
        self.bin_checkbox.setVisible(False)

        # Przyciski do generowania wykresów
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

        self.chart_type_combo.currentIndexChanged.connect(self.chart_type_changed)
        self.chart_type_changed()

        self.group_execute_btn = QPushButton('📈 Generate chart')
        self.group_execute_btn.clicked.connect(self.update_chart)
        self.group_execute_btn.setVisible(False)
        self.left_layout.addWidget(self.group_execute_btn)

        # Dodatkowe funkcje

        self.generate_report_btn = QPushButton('📄 Generate report')
        self.generate_report_btn.clicked.connect(self.generate_report)
        self.generate_report_btn.setVisible(False)  # Ukryty na start
        self.left_layout.addWidget(self.generate_report_btn)

        self.clear_filters_btn = QPushButton('❌ Reset filters and grouping')
        self.clear_filters_btn.clicked.connect(self.clear_filters)
        self.clear_filters_btn.setVisible(False)  # Ukryty na start
        self.left_layout.addWidget(self.clear_filters_btn)

        # Layout końcowy
        self.main_layout.addWidget(self.button)
        self.main_layout.addLayout(self.center_layout, 2)
        self.main_layout.addWidget(self.bottom_box, 1)

        self.setLayout(self.main_layout)

    def clear_filters(self):
        """
        Resetuje wybrane filtry i grupowanie – przywraca domyślne ustawienia.
        """
        # Reset wybranej kolumny do filtrowania
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

    def clear_right_panel(self):
        """
        Usuwa wszystkie wykresy z prawego panelu (right_layout).
        """
        for i in reversed(range(self.right_layout.count())):
            widget_to_remove = self.right_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

    # Funkcje dotyczące grupowania i filtrowania danych
    def get_filtered_data(self):
        """
        Zwraca dane przefiltrowane według wybranej kolumny i kryteriów.
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
        # Filtrowanie w kolumnach z wartościami nienumerycznymi
        elif self.category_filter_combo.isVisible():
            selected_val = self.category_filter_combo.currentData()
            if selected_val is not None:
                filtered_df = self.data[self.data[filter_col].astype(str) == str(selected_val)].copy()
            else:
                filtered_df = self.data.copy()
        else:
            filtered_df = self.data.copy()

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
        df = self.get_filtered_data()
        if df is None or df.empty:
            return None

        group_col = self.group_column_combo.currentData()
        agg_func = self.get_selected_agg_func()
        agg_col = self.agg_column_combo.currentData()

        if not group_col or not agg_func or (agg_func != 'count' and not agg_col):
            return None

        group_cols = [group_col]
        if self.gender_checkbox.isChecked() and 'Gender' in df.columns:
            group_cols.append('Gender')

        df, group_col_or_binned = self.bin_column_if_needed(df, group_col)
        if group_col_or_binned != group_col:
            group_cols[0] = group_col_or_binned

        grouped_df, y_col = self.aggregate_grouped_data(df, group_cols, agg_col, agg_func)

        return grouped_df

    def bin_column_if_needed(self, df, group_col):
        """
        Zwraca DataFrame z dodaną kolumną z binowaniem, jeśli włączone.
        """
        if self.bin_checkbox.isChecked() and pd.api.types.is_numeric_dtype(df[group_col]):
            binned_col_name = f'binned_{group_col}'
            df[binned_col_name] = self.bin_numeric_column(df[group_col], column_name=group_col)
            return df, binned_col_name
        return df, group_col

    import numpy as np
    from pandas.api.types import CategoricalDtype

    def bin_numeric_column(self, series, column_name=None):
        """
        Dzieli kolumnę numeryczną na przedziały z poprawnym uwzględnieniem wartości granicznych.
        """
        try:
            # Definicje przedziałów i etykiet dla poszczególnych kolumn
            bin_configs = {
                'BMI': {
                    'bins': [0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf],
                    'labels': ['<18.5', '18.5-24.9', '25.0-29.9', '30.0-34.9', '35.0-39.9', '40.0+']
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
        Główna funkcja wywołująca generowanie wykresów na podstawie aktualnych ustawień.
        """
        self.clear_right_panel()
        selected_chart = self.chart_type_combo.currentText()

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
        else:
            self.handle_other_charts()

    def handle_other_charts(self):
        """
        Generuje wykresy, które wymagają agregowanych danych.
        """
        grouped = self.prepare_aggregated_data()
        if grouped is None or grouped.empty:
            self.log_area.append('No data to display.')
            return

        group_col = self.group_column_combo.currentData()
        agg_func = self.get_selected_agg_func()
        if not agg_func:
            self.log_area.append('No aggregation function selected.')
            return

        x_col = f'binned_{group_col}' if self.bin_checkbox.isChecked() else group_col
        y_col = grouped.columns[-1]
        x_unit = self.column_units.get(group_col, '')
        y_unit = self.column_units.get(y_col, '') if agg_func != 'count' else ''

        x_label = self.column_labels.get(group_col, group_col)
        y_label = self.column_labels.get(y_col, y_col)
        if agg_func == 'count':
            title = f'{self.agg_func_buttons[agg_func].text()} by {x_label}'
        else:
            title = f'{self.agg_func_buttons[agg_func].text()} values of {y_label} by {x_label}'
        hue_col = 'Gender' if self.gender_checkbox.isChecked() else None

        self.generate_chart(
            data=grouped,
            x_col=x_col,
            y_col=y_col,
            x_label=f'{x_label} ({x_unit})' if x_unit else x_label,
            y_label=f'{y_label} ({y_unit})' if y_unit else y_label,
            title=title,
            hue_col=hue_col,
            agg_func=agg_func,
        )

    def generate_hist(self, data, column):
        """
        Generuje histogram dla wybranej kolumny numerycznej z oznaczeniem średniej i mediany.
        """

        if not column or column not in data.columns:
            self.log_area.append('Invalid column selected for histogram.')
            return

        if not pd.api.types.is_numeric_dtype(data[column]):
            self.log_area.append(f'Column "{column}" is not numeric and cannot be used for histogram.')
            return

        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        col_name = self.column_labels.get(column, column)

        try:
            # Oblicz statystyki
            stats = self.statistics(data, column)

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
            ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2,
                       label=f"Mean: {stats['mean']:.2f}")
            ax.axvline(stats['median'], color='green', linestyle=':', linewidth=2,
                       label=f"Median: {stats['median']:.2f}")

            # Dodaj tytuł, etykiety i legendę
            ax.set_title(f'Histogram of {col_name}')
            ax.set_xlabel(col_name)
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

    def generate_chart(self, data, x_col, y_col, x_label, y_label, title, hue_col=None, agg_func=None):
        """
        Generowanie pozostałych wykresów na podstawie wybranego typu.
        """
        selected_chart = self.chart_type_combo.currentText()

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
                success = plot_func(ax, data, x_col, y_col, agg_func)
            else:
                success = plot_func(ax, data, x_col, y_col, hue_col)
                if success:
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.grid(True)

            if not success:
                # Jeśli wykres się nie wygenerował, zakończ i nie renderuj
                return

            full_title = title + (' (gender differences)' if hue_col else '')
            ax.set_title(full_title, pad=20)

            self.render_figure(fig)
            self.log_area.append(f'{selected_chart} has been generated.')

        except Exception as e:
            self.log_area.append(f"Error while generating chart: {e}")

    def bar_chart(self, ax, data, x_col, y_col, hue_col):
        palette = {'F': '#fe46a5', 'M': '#82cafc'}
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

    def scatter_plot(self, ax, data, x_col, y_col, hue_col):
        palette = {'F': '#fe46a5', 'M': '#82cafc'}
        if hue_col:
            sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col, palette=palette, ax=ax)
        else:
            sns.scatterplot(data=data, x=x_col, y=y_col, ax=ax)
        return True

    def pie_chart(self, ax, data, x_col, y_col, agg_func):

        binning_enabled = self.bin_checkbox.isChecked()
        hue_col = self.gender_checkbox.isChecked()
        is_numeric_col = pd.api.types.is_numeric_dtype(data[x_col])
        agg_is_count = (agg_func == 'count')
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

            original_col = x_col.replace('binned_', '') if x_col.startswith('binned_') else x_col
            column_label = self.column_labels.get(original_col, original_col)
            unit = self.column_units.get(original_col, '')
            legend_title = f'Legend – {column_label} ({unit})' if unit else f'Legend – {column_label}'

            # Tworzenie legendy
            ax.legend(
                wedges,
                legend_labels,
                title=legend_title,
                loc='center left',
                bbox_to_anchor=(1, 0.5)
            )
            ax.figure.subplots_adjust(right=0.75)
            ax.axis('equal')
            return True

        except Exception as e:
            self.log_area.append(f"Error in pie chart generation: {e}")
            return False

    def render_figure(self, fig):
        """
        Wyświetla wykres na panelu po prawej stronie.
        """
        canvas = FigureCanvas(fig)
        self.right_layout.addWidget(canvas)

    def statistics(self, data, column):
        """
        Oblicza i loguje statystyki (mean, median, std, min, max) dla wybranej kolumny
        """
        values = data[column].dropna()
        if values.empty:
            self.log_area.append('No data available for statistics.')
            return None

        stats = {
            'mean': values.mean(),
            'median': values.median(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max()
        }
        self.log_area.append('--- Statistics ---')
        self.log_area.append(
            '\n'.join([
                f"Mean: {stats['mean']:.2f}",
                f"Median: {stats['median']:.2f}",
                f"Std: {stats['std']:.2f}",
                f"Min: {stats['min']:.2f}",
                f"Max: {stats['max']:.2f}",
            ])
        )
        return stats

    # Dostosowywanie widoczności opcji interfejsu w zależności od typu wykresu i danych
    def chart_type_changed(self):
        selected_chart = self.chart_type_combo.currentText()
        disable_grouping = selected_chart in ['Histogram', 'Heatmap']
        disable_agg = selected_chart in ['Pie chart']

        # Ukryj/pokaż "Group by:"
        self.grouping_section_label.setVisible(not disable_grouping)

        # Ukryj/pokaż combobox do kolumny grupowania
        self.group_column_combo.setVisible(not disable_grouping)

        # Ukryj/pokaż combobox do kolumny agregacji
        self.agg_column_combo.setVisible(not disable_grouping and not disable_agg)

        # Przyciski funkcji agregujących
        self.agg_func_group.setExclusive(False)
        for name, btn in self.agg_func_buttons.items():
            if selected_chart == "Pie Chart":
                # Dla PieChart tylko Count jest widoczny i zaznaczony
                if name.lower() == "count":
                    btn.setVisible(True)
                    btn.setChecked(True)
                    btn.setEnabled(True)
                else:
                    btn.setVisible(False)
                    btn.setChecked(False)
                    btn.setEnabled(False)
            else:
                # Dla pozostałych wykresów widoczne, jeśli grupowanie włączone
                visible = not disable_grouping
                btn.setVisible(visible)
                btn.setEnabled(visible)
                if not visible:
                    btn.setChecked(False)
        self.agg_func_group.setExclusive(True)

        # Aktualizacja listy kolumn filtrowania (na podstawie wybranego wykresu)
        self.update_filter_column_options()

        # Widoczność checkboxów
        self.update_checkboxes_visibility()

    def agg_func_changed(self):
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

        # Zapamiętaj aktualnie wybraną kolumnę filtra (userData)
        current_filter_col = self.filter_column_combo.currentData()

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

        # Przywróć wybrany filtr, jeśli jest dostępny
        index = self.filter_column_combo.findData(current_filter_col)
        if index != -1:
            self.filter_column_combo.setCurrentIndex(index)
        else:
            self.filter_column_combo.setCurrentIndex(0)

        self.update_filter_values()

    def update_grouping_column_options(self):
        """
        Aktualizuje kolumny dostępne do grupowania i agregacji po wczytaniu pliku.
        """
        if self.data is None:
            return

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
        self.agg_func_changed()

        # Resetuj wybór kolumny agregacji (na „Not selected”)
        self.agg_column_combo.setCurrentIndex(0)

        # Pokaż przycisk wykonania grupowania
        self.group_execute_btn.setVisible(True)

        # Pokaż lub ukryj checkbox "Show data in ranges"
        self.update_checkboxes_visibility()

    def update_checkboxes_visibility(self):
        """
        Ustawia widoczność checkboxów w zależności od:
        - wybranego wykresu,
        - danych,
        - typu kolumny grupującej (dla bin),
        - wybranych filtrów.
        """

        selected_chart = self.chart_type_combo.currentText()
        selected_group_col = self.group_column_combo.currentData()

        # --- GENDER checkbox ---
        show_gender = (
                self.data is not None and
                selected_chart not in ['Pie Chart', 'Histogram', 'Heatmap'] and
                'Gender' in self.data.columns and
                self.data['Gender'].nunique() >= 2 and
                self.filter_column_combo.currentData() != 'Gender' and
                selected_group_col != 'Gender'
        )

        # --- BIN checkbox ---
        # Wykluczamy Heatmap i Histogram
        chart_allows_bin = selected_chart not in ['Heatmap', 'Histogram']

        group_col_is_numeric = (
                self.data is not None and
                selected_group_col is not None and
                pd.api.types.is_numeric_dtype(self.data[selected_group_col])
        )

        show_bin = chart_allows_bin and group_col_is_numeric

        # --- Widoczność i ustawienia checkboxów ---
        self.gender_checkbox.setVisible(show_gender)
        self.gender_checkbox.setChecked(False)  # zawsze odznaczony

        self.bin_checkbox.setVisible(show_bin)
        self.bin_checkbox.setChecked(show_bin)  # zaznaczony tylko jeśli widoczny

    def category_combo_changed(self):
        """
        Wyświetla komunikat, jeśli użytkownik zmieni wybraną wartość filtra
        """
        selected = self.category_filter_combo.currentText()
        if selected:
            self.log_area.append(f'Selected value: {selected}')

    # Wczytanie pliku CSV
    def select_file(self):
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
            self.gender_checkbox.setVisible(False)
            self.bin_checkbox.setVisible(False)

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
        Generuje raport CSV z danych użytych do wykresu (z filtrowaniem, grupowaniem, binowaniem, płcią).
        """
        if self.chart_type_combo.currentText() in ['Heatmap', 'Histogram']:
            df = self.get_filtered_data()
        else:
            df = self.prepare_aggregated_data()

        if df is None or df.empty:
            self.log_area.append('No data to export.')
            return

        path, _ = QFileDialog.getSaveFileName(
            self, 'Save CSV Report', '', 'CSV Files (*.csv);;All Files (*)'
        )

        if path:
            try:
                df.to_csv(path, index=False)
                self.log_area.append(f'Report successfully saved:\n{path}')
            except Exception as e:
                self.log_area.append(f'Failed to save the report:\n{str(e)}')


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
