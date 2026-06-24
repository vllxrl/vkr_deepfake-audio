# app/report.py  —  Генерация PDF-отчёта по результатам анализа

from __future__ import annotations
import io
from datetime import datetime

import matplotlib.pyplot as plt
from fpdf import FPDF


class ReportPDF(FPDF):
    """PDF-отчёт со встроенными графиками."""

    def __init__(self, font_dir: str | None = None):
        super().__init__()
        import os
        if font_dir is None:
            font_dir = os.path.join(os.path.dirname(__file__), "fonts")

        regular = os.path.join(font_dir, "ALS_Sector-Regular.otf")
        bold    = os.path.join(font_dir, "ALS_Sector-Bold.otf")

        self.add_font("ALSSector", "",  regular, uni=True)
        self.add_font("ALSSector", "B", bold,    uni=True)
        self._font_family = "ALSSector"

    # колонтитулы 
    def header(self):
        self.set_font("ALSSector", "B", 11)
        self.cell(0, 8, "Детектор дипфейк-аудио | Отчёт",
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("ALSSector", "", 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 10, f"Страница {self.page_no()}/{{nb}}", align="C")

    # ── Утилиты ──────────────────────────────────────────────────────────────
    def section_title(self, text: str):
        self.set_font("ALSSector", "B", 13)
        self.set_text_color(30, 30, 30)
        self.ln(4)
        self.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("ALSSector", "", 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def key_value(self, key: str, value: str):
        self.set_font("ALSSector", "B", 10)
        self.set_text_color(50, 50, 50)
        self.cell(55, 6, f"{key}:")
        self.set_font("ALSSector", "", 10)
        self.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")

    def add_figure(self, fig: plt.Figure, w: int = 180):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
        buf.seek(0)
        if self.get_y() + 80 > 270:
            self.add_page()
        self.image(buf, x=(210 - w) / 2, w=w)
        self.ln(4)
        buf.close()


# сборка отчёта
# параметры
def build_analysis_report(
    filename: str,
    label: str,
    confidence: float,
    prob_real: float,
    prob_fake: float,
    fig_wave: plt.Figure,
    fig_mel: plt.Figure,
    fig_grad: plt.Figure,
    fig_conf_bar: plt.Figure,
    models_names: list[str] | None = None,      # имена моделей
    models_acc: list[float] | None = None,      # accuracy
    models_f1: list[float] | None = None,       # F1-мера
    best_model: str | None = None,              # лучшая модель
) -> bytes:
    """
    Формирует PDF-отчёт по одному аудиофайлу.
    Возвращает байты готового PDF.
    """
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # шапка
    pdf.section_title("1. Общие сведения")
    pdf.key_value("Файл", filename)
    pdf.key_value("Дата анализа", datetime.now().strftime("%d.%m.%Y  %H:%M:%S"))
    pdf.key_value("Результат", label)
    pdf.key_value("Уверенность", f"{confidence:.1%}")
    pdf.key_value("P(настоящая)", f"{prob_real:.4f}")
    pdf.key_value("P(дипфейк)", f"{prob_fake:.4f}")
    pdf.ln(2)

    # шкала вероятностей
    pdf.section_title("2. Распределение вероятностей")
    pdf.add_figure(fig_conf_bar, w=170)

    # форма волны 
    pdf.section_title("3. Форма волны")
    pdf.add_figure(fig_wave, w=170)

    # мел-спектрограмма
    pdf.section_title("4. Лог мел-спектрограмма")
    pdf.add_figure(fig_mel, w=170)

    # Grad-CAM
    pdf.section_title("5. Grad-CAM — тепловая карта внимания модели")
    pdf.body_text(
        "Ниже отображены области спектрограммы, которые оказали "
        "наибольшее влияние на решение классификатора. "
        "Яркие зоны соответствуют высокой активации."
    )
    pdf.add_figure(fig_grad, w=170)

    if models_names and models_acc and models_f1 and best_model:
        pdf.section_title("6. Сравнение всех моделей")
        # таблица 
        pdf.set_font("ALSSector", "B", 10)
        pdf.cell(70, 8, "Модель", border=1, align='C')
        pdf.cell(45, 8, "Accuracy", border=1, align='C')
        pdf.cell(45, 8, "F1-мера", border=1, align='C')
        pdf.ln()
        # строки с данными
        pdf.set_font("ALSSector", "", 10)
        for name, acc, f1 in zip(models_names, models_acc, models_f1):
            pdf.cell(70, 8, name, border=1)
            pdf.cell(45, 8, f"{acc:.3f}", border=1, align='C')
            pdf.cell(45, 8, f"{f1:.3f}", border=1, align='C')
            pdf.ln()
        pdf.ln(4)
        pdf.body_text(f"Лучшая модель по F1-мере: {best_model}")
        pdf.ln(2)


    # интерпретация
    pdf.section_title("7. Интерпретация")
    if "Дипфейк" in label:
        pdf.body_text(
            "Модель классифицировала данную запись как синтетическую (дипфейк). "
            "На тепловой карте Grad-CAM обычно выделяются нерегулярные паттерны "
            "в высокочастотном диапазоне или неестественная периодичность, "
            "характерная для артефактов синтеза речи."
        )
    else:
        pdf.body_text(
            "Модель классифицировала данную запись как настоящую. "
            "Активации на тепловой карте распределены относительно равномерно, "
            "что указывает на естественную структуру речевого сигнала."
        )

    # сборка
    return bytes(pdf.output())  # bytearray -> bytes