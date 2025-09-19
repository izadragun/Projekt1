from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTextEdit, QGroupBox, QComboBox,
    QRadioButton, QButtonGroup, QGridLayout, QDoubleSpinBox, QCheckBox)

from pandas.api.types import CategoricalDtype

import sys
import pandas as pd

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# Wczytanie danych z pliku CSV.
def load_data(path):
    try:
        data = pd.read_csv(path, delimiter=',')
        return data
    except FileNotFoundError:
        print("Nie odnaleziono pliku")


def group_data(data, by_column, agg_column, agg_func):
    """
    Grupuje dane na podstawie wybranej kolumny
    """
    if by_column not in data.columns or agg_column not in data.columns:
        raise ValueError("Nieprawidłowe kolumny.")

    return data.groupby(by_column, observed=True)[agg_column].agg(agg_func).reset_index()


# GUI: wybór pliku i filtrów.
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.data = None
        self.setWindowTitle("Analiza danych")
        self.setGeometry(100, 100, 1200, 800)

        self.main_layout = QVBoxLayout()
        self.center_layout = QHBoxLayout()

        # Lewa część: przyciski filtrowania
        self.left_box = QGroupBox("🔍 Filtry i grupowanie")
        self.left_layout = QVBoxLayout()

        self.left_box.setLayout(self.left_layout)
        self.left_box.setMinimumWidth(300)

        # Prawa część: Wizualizacje
        self.right_box = QGroupBox("📊 Wizualizacje")
        self.right_layout = QVBoxLayout()
        self.right_box.setLayout(self.right_layout)

        self.center_layout.addWidget(self.left_box, 1)
        self.center_layout.addWidget(self.right_box, 2)

        # Dolna część: Logi
        self.bottom_box = QGroupBox("📝 Komunikaty")
        self.bottom_layout = QVBoxLayout()
        self.label = QLabel("Nie wybrano pliku")
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFixedHeight(150)

        self.bottom_layout.addWidget(self.label)
        self.bottom_layout.addWidget(self.log_area)
        self.bottom_box.setLayout(self.bottom_layout)

        # Przycisk wyboru pliku
        self.button = QPushButton("📂 Wybierz plik CSV")
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
        self.left_layout.addWidget(QLabel("Filtruj wg:"))

        self.filter_column_combo = QComboBox()
        self.filter_column_combo.currentIndexChanged.connect(self.update_filter_values)
        self.left_layout.addWidget(self.filter_column_combo)

        # Dla kolumn z wartościami numerycznymi
        self.filter_min_spinbox = QDoubleSpinBox()
        self.filter_max_spinbox = QDoubleSpinBox()

        # Domyślny zakres na start, zostanie potem zmieniony dynamicznie
        self.filter_min_spinbox.setRange(0, 1000)
        self.filter_max_spinbox.setRange(0, 1000)

        self.filter_min_spinbox.setPrefix("Od: ")
        self.filter_max_spinbox.setPrefix("Do: ")

        self.filter_min_spinbox.setDecimals(1)  # wyświetla 1 miejsce po przecinku
        self.filter_max_spinbox.setDecimals(1)

        spinbox_layout = QHBoxLayout()
        spinbox_layout.addWidget(self.filter_min_spinbox)
        spinbox_layout.addWidget(self.filter_max_spinbox)

        self.left_layout.addLayout(spinbox_layout)

        self.filter_min_spinbox.hide()
        self.filter_max_spinbox.hide()

        # Dla kolumn z wartościami nienumerycznymi
        self.category_combo_label = QLabel("Wybierz wartość:")
        self.left_layout.addWidget(self.category_combo_label)
        self.category_combo_label.hide()

        self.category_filter_combo = QComboBox()
        self.category_filter_combo.currentIndexChanged.connect(self.on_category_combo_changed)
        self.left_layout.addWidget(self.category_filter_combo)
        self.category_filter_combo.hide()

        # Odstęp przed "Grupuj wg"
        self.left_layout.addSpacing(20)

        # Grupowanie: wybór kolumn
        self.grouping_section_label = QLabel("Grupuj wg kolumny:")
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
            "mean": "Średnia",
            "median": "Mediana",
            "count": "Ilość pacjentów",
            "min": "Min",
            "max": "Max"
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

        self.bin_checkbox = QCheckBox("📦 Pokaż przedziały")
        self.bin_checkbox.setChecked(True)
        self.left_layout.addWidget(self.bin_checkbox)

        # Przyciski w dolnej części okna
        self.group_execute_btn = QPushButton("📈 Generuj wykres")
        self.group_execute_btn.clicked.connect(self.perform_grouping)
        self.group_execute_btn.setVisible(False)
        self.left_layout.addWidget(self.group_execute_btn)

        self.generate_report_btn = QPushButton("📄 Generuj raport")
        self.generate_report_btn.clicked.connect(self.generate_report)
        self.generate_report_btn.setVisible(False)  # Ukryty na start
        self.left_layout.addWidget(self.generate_report_btn)

        self.clear_filters_btn = QPushButton("❌ Usuń filtry")
        self.clear_filters_btn.clicked.connect(self.on_clear_filters)
        self.clear_filters_btn.setVisible(False)  # Ukryty na start
        self.left_layout.addWidget(self.clear_filters_btn)

        # Layout końcowy
        self.main_layout.addWidget(self.button)
        self.main_layout.addLayout(self.center_layout, 2)
        self.main_layout.addWidget(self.bottom_box, 1)

        self.setLayout(self.main_layout)

    # Eksport wyników do pliku CSV.
    def generate_report(self):
        pass

    def on_clear_filters(self):
        """
        Resetuje wybrane filtry i przywraca domyślne ustawienia.
        """
        self.filter_column_combo.setCurrentIndex(0)  # Reset wyboru kolumny

        # Reset wartości spinboxów
        self.filter_min_spinbox.setValue(self.filter_min_spinbox.minimum())
        self.filter_max_spinbox.setValue(self.filter_max_spinbox.maximum())

        # Reset kategorii
        self.category_filter_combo.setCurrentIndex(0)

        # Ukryj wszystkie widżety filtrowania
        self.filter_min_spinbox.hide()
        self.filter_max_spinbox.hide()
        self.category_filter_combo.hide()
        self.category_combo_label.hide()

        self.log_area.append("Filtry zostały zresetowane.")

    def perform_grouping(self):
        """
        Grupowanie na podstawie wybranych kryteriów.
        """

        self.clear_logs()
        self.clear_right_panel()

        df = self.get_filtered_data()
        if df is None or df.empty:
            self.log_area.append("Brak danych do wyświetlenia.")
            return

        group_col = self.group_column_combo.currentData()

        # Pobierz zaznaczoną funkcję agregującą
        agg_func = None
        for func_key, btn in self.agg_func_buttons.items():
            if btn.isChecked():
                agg_func = func_key
                break

        if not group_col:
            self.log_area.append("Wybierz kolumnę do grupowania.")
            return

        if not agg_func:
            self.log_area.append("Wybierz funkcję agregującą.")
            return

        # Sprawdź, czy użytkownik chce widzieć dane w przedziałach
        if self.bin_checkbox.isChecked() and pd.api.types.is_numeric_dtype(df[group_col]):
            binned_col = self.bin_numeric_column(df[group_col], column_name=group_col)
            df["_binned_group"] = binned_col
            group_key = "_binned_group"
        else:
            group_key = group_col

        try:
            if agg_func == "count":
                grouped = df.groupby(group_key, observed=True).size().reset_index(name="Liczba rekordów")
                grouped = grouped.sort_values(by=group_key)
                y_values = grouped["Liczba rekordów"]
                y_label = "Liczba rekordów"
            else:
                agg_col = self.agg_column_combo.currentData()
                if not agg_col:
                    self.log_area.append("Wybierz kolumnę do agregacji.")
                    return

                grouped = group_data(df, group_key, agg_col, agg_func)
                grouped = grouped.sort_values(by=group_key)
                y_values = grouped[agg_col]
                y_label = self.column_labels.get(agg_col, agg_col)

            agg_label = self.agg_func_buttons[agg_func].text()
            x_label = self.column_labels.get(group_col, group_col)
            title = f"{agg_label} by {x_label}"
            y_axis_label = f"{agg_label} {y_label}"

            self.generate_chart(grouped[group_key], y_values, x_label, y_axis_label, title)

            self.log_area.append("Wygenerowano wykres.")

        except Exception as e:
            self.log_area.append(f"Błąd podczas grupowania: {e}")

    def generate_chart(self, x_values, y_values, x_label, y_label, title):
        """
        Generowanie wykresu
        """
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        # Wykres kołowy tylko dla CLASS
        if self.group_column_combo.currentData() == "CLASS":
            wedges, texts, autotexts = ax.pie(
                y_values,
                labels=x_values,
                autopct='%1.1f%%',
                startangle=90,
                colors=['pink', 'orange', 'mediumseagreen']
            )
            ax.set_title("Klasyfikacja pacjentów pod kątem wystepowania cukrzycy")
            ax.axis('equal')

            # Dodajemy legendę
            class_labels = {
                'Y': 'Diabetic',
                'N': 'Non-Diabetic',
                'P': 'Prediction of diabetes'
            }

            # Dopasuj etykiety do objaśnień
            legend_labels = [class_labels.get(str(label), str(label)) for label in x_values]
            ax.legend(wedges, legend_labels, title="Legenda", loc="best")

        else:
            ax.bar(x_values, y_values, color='mediumseagreen')
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True)

        fig.tight_layout()
        canvas = FigureCanvas(fig)
        self.right_layout.addWidget(canvas)

    def on_agg_func_changed(self):
        selected_func = None
        for func_key, btn in self.agg_func_buttons.items():
            if btn.isChecked():
                selected_func = func_key
                break

        if selected_func == "count":
            self.agg_column_combo.setVisible(False)
        else:
            self.agg_column_combo.setVisible(True)

    def update_filter_column_options(self):
        if self.data is not None:
            self.filter_column_combo.clear()
            self.filter_column_combo.addItem("Nie wybrano", userData=None)
            for col in self.data.columns:
                if col in self.column_labels:
                    label = self.column_labels[col]
                    self.filter_column_combo.addItem(label, userData=col)
            self.update_filter_values()

    def update_grouping_column_options(self):
        """
        Aktualizuje kolumny dostępne do grupowania i agregacji po wczytaniu pliku.
        """
        if self.data is None:
            return

        self.group_column_combo.clear()
        self.group_column_combo.addItem("Nie wybrano", userData=None)

        self.agg_column_combo.clear()
        self.agg_column_combo.addItem("Nie wybrano", userData=None)

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

    def bin_numeric_column(self, series, column_name=None):
        """
        Dzieli kolumnę numeryczną na przedziały.
        """
        try:
            if column_name == "BMI":
                bins = [0, 18.5, 24.9, 29.9, 34.9, 100]
                labels = ["Niedowaga", "Norma", "Nadwaga", "Otyłość I", "Otyłość II+"]
                cat_type = CategoricalDtype(categories=labels, ordered=True)
                return pd.cut(series, bins=bins, labels=labels).astype(cat_type)

            elif column_name == "AGE":
                bins = [0, 20, 30, 40, 50, 60, 70, 80]
                labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
                cat_type = CategoricalDtype(categories=labels, ordered=True)
                return pd.cut(series, bins=bins, labels=labels).astype(cat_type)

            elif column_name == "Chol":
                bins = [0, 4.99, 5.99, 6.99, 12]
                labels = ['<5', '5-5.99', '6-6.99', '7≤ ']
                cat_type = CategoricalDtype(categories=labels, ordered=True)
                return pd.cut(series, bins=bins, labels=labels).astype(cat_type)

            elif column_name == 'Cr':
                bins = [0, 49.99, 99.99, 149.99, 199.99, 800]
                labels = ['<50', '50-99.99', '100-149.99', '150-199.99' '200≤ ']
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
            self.log_area.append(f"Błąd podziału na kolumny '{column_name}': {e}")
            return series

    def update_filter_values(self):
        """Aktualizuje opcje filtrowania"""
        filter_col = self.filter_column_combo.currentData()
        if not filter_col or self.data is None:
            return

        col_data = self.data[filter_col].dropna()

        if pd.api.types.is_integer_dtype(col_data):
            min_val = int(col_data.min())
            max_val = int(col_data.max())
            self.filter_min_spinbox.setDecimals(0)
            self.filter_max_spinbox.setDecimals(0)
            self.filter_min_spinbox.setSingleStep(1)
            self.filter_max_spinbox.setSingleStep(1)
        elif pd.api.types.is_float_dtype(col_data):
            min_val = float(col_data.min())
            max_val = float(col_data.max())
            self.filter_min_spinbox.setDecimals(1)
            self.filter_max_spinbox.setDecimals(1)
            self.filter_min_spinbox.setSingleStep(0.1)
            self.filter_max_spinbox.setSingleStep(0.1)
        else:
            return  # np. kolumny nie numeryczne - nie pokazuj spinboxów

        self.filter_min_spinbox.setRange(min_val, max_val)
        self.filter_max_spinbox.setRange(min_val, max_val)

        self.filter_min_spinbox.setValue(min_val)
        self.filter_max_spinbox.setValue(max_val)

        self.filter_min_spinbox.show()
        self.filter_max_spinbox.show()

        # Ukryj kategoryczne
        self.category_filter_combo.hide()
        self.category_combo_label.hide()

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

        if pd.api.types.is_numeric_dtype(col_data):
            min_val = self.filter_min_spinbox.value()
            max_val = self.filter_max_spinbox.value()
            return self.data[
                (self.data[filter_col] >= min_val) & (self.data[filter_col] <= max_val)
                ].copy()

        elif self.category_filter_combo.isVisible():
            selected_val = self.category_filter_combo.currentData()
            if selected_val is not None:
                return self.data[self.data[filter_col].astype(str) == str(selected_val)].copy()

        return self.data.copy()

    def update_numeric_columns(self):
        """
        Aktualizuje opcje kolumny agregacji po wybraniu kolumny grupowania.
        Ukrywa lub blokuje niepasujące funkcje agregujące.
        """
        selected_group_col = self.group_column_combo.currentData()
        if not selected_group_col or self.data is None:
            return

        # Zaktualizuj listę kolumn agregacji (jak wcześniej)
        numeric_cols = self.data.select_dtypes(include='number').columns.tolist()

        if selected_group_col in numeric_cols:
            if self.agg_column_combo.count() == 0:  # Jeśli pusta, uzupełnij
                labeled_cols = [col for col in self.data.columns if col in self.column_labels]
                numeric_labeled_cols = [col for col in numeric_cols if col in labeled_cols]
                self.agg_column_combo.clear()
                self.agg_column_combo.addItem("Nie wybrano", userData=None)
                for col in numeric_labeled_cols:
                    label = self.column_labels[col]
                    self.agg_column_combo.addItem(label, userData=col)

            # Wszystkie przyciski agregujące dostępne
            for func_key, btn in self.agg_func_buttons.items():
                btn.setEnabled(True)

            self.agg_column_combo.setVisible(True)
        else:
            # Jeśli kolumna grupująca NIE jest numeryczna:
            self.agg_column_combo.setVisible(False)

            # Tylko "count" może być zaznaczone
            for func_key, btn in self.agg_func_buttons.items():
                if func_key == "count":
                    btn.setEnabled(True)
                    btn.setChecked(True)
                else:
                    btn.setEnabled(False)

        self.group_execute_btn.setVisible(True)
        # Pokaż lub ukryj checkbox "Pokaż przedziały"
        if pd.api.types.is_numeric_dtype(self.data[selected_group_col]):
            self.bin_checkbox.setVisible(True)
        else:
            self.bin_checkbox.setVisible(False)

    def on_category_combo_changed(self):
        """
        Wyświetla komunikat, jeśli użytkownik zmieni wybraną wartość filtra
        """
        selected = self.category_filter_combo.currentText()
        if selected:
            self.log_area.append(f"Wybrano wartość: {selected}")

    def on_file_select(self):
        """
        Wybór pliku
        """
        path, _ = QFileDialog.getOpenFileName(self, "Wybierz plik CSV", "", "Pliki CSV (*.csv)")
        if not path:
            self.log_area.append("Nie wybrano pliku.")
            return

        self.label.setText(f"Wybrano: {path}")
        self.log_area.append(f"Załadowano plik: {path}")
        self.data = load_data(path)

        if self.data is not None:
            self.log_area.append("Dane zostały przetworzone.")
            self.update_filter_column_options()
            self.update_grouping_column_options()
            # Pokaż przyciski po załadowaniu danych
            self.group_execute_btn.setVisible(True)
            self.generate_report_btn.setVisible(True)
            self.clear_filters_btn.setVisible(True)
        else:
            self.log_area.append("Błąd podczas przetwarzania danych.")
            # Ukryj przyciski, jeśli dane się nie wczytały
            self.group_execute_btn.setVisible(False)
            self.generate_report_btn.setVisible(False)
            self.clear_filters_btn.setVisible(False)

    def clear_logs(self):
        self.log_area.clear()

    def clear_right_panel(self):
        for i in reversed(range(self.right_layout.count())):
            widget_to_remove = self.right_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
