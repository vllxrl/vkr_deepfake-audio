# app/main.py  —  Детектор дипфейк-аудио · Streamlit UI

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import config
import predict as pred_module

from report import build_analysis_report

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Детектор дипфейк-аудио",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Максимальная ширина — графики не растягиваются на весь монитор */
    .block-container {
        padding-top: 2rem;
        max-width: 900px;
        margin: 0 auto;
    }

    .result-card {
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .card-real { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .card-fake { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .card-stub { background: #fff3cd; color: #856404; border: 1px solid #ffeeba; }

    .result-label { font-size: 2.4rem; font-weight: 700; margin: 0; }
    .result-conf  { font-size: 1.1rem; margin-top: 0.3rem; opacity: 0.85; }

    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #888;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def fig_waveform(waveform: np.ndarray) -> plt.Figure:
    t = np.linspace(0, config.DURATION, len(waveform))
    fig, ax = plt.subplots(figsize=(7, 1.8))
    ax.plot(t, waveform, color="#4C9BE8", linewidth=0.6, alpha=0.85)
    ax.axhline(0, color="#ccc", linewidth=0.5)
    ax.set_xlim(0, config.DURATION)
    ax.set_xlabel("Время (с)", fontsize=8)
    ax.set_ylabel("Амплитуда", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("Форма волны", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_melspec(melspec: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 2.6))
    img = ax.imshow(
        melspec, aspect="auto", origin="lower",
        cmap="magma",
        extent=[0, config.DURATION, 0, config.N_MELS],
    )
    fig.colorbar(img, ax=ax, format="%+2.0f дБ", pad=0.01)
    ax.set_xlabel("Время (с)", fontsize=8)
    ax.set_ylabel("Мел-бин", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("Лог мел-спектрограмма", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_gradcam(melspec: np.ndarray, gradcam: np.ndarray, is_fake: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 2.6))

    ax.imshow(melspec, aspect="auto", origin="lower", cmap="gray",
              extent=[0, config.DURATION, 0, config.N_MELS], alpha=0.6)

    cmap    = "Reds" if is_fake else "Blues"
    overlay = ax.imshow(gradcam, aspect="auto", origin="lower",
                        cmap=cmap, alpha=0.55,
                        extent=[0, config.DURATION, 0, config.N_MELS],
                        vmin=0, vmax=1)

    fig.colorbar(overlay, ax=ax, label="Активация", pad=0.01)
    ax.set_xlabel("Время (с)", fontsize=8)
    ax.set_ylabel("Мел-бин", fontsize=8)
    ax.tick_params(labelsize=7)
    verdict = "дипфейк" if is_fake else "настоящая"
    ax.set_title(f"Grad-CAM — области, определившие решение «{verdict}»",
                 fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig


def confidence_bar(prob_real: float, prob_fake: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 0.55))
    ax.barh(0, prob_real, color="#2ecc71", height=0.6, label="Настоящая")
    ax.barh(0, prob_fake, left=prob_real, color="#e74c3c", height=0.6, label="Дипфейк")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.5)
    fig.tight_layout(pad=0.3)
    return fig


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_demo, tab_results, tab_about = st.tabs([
    "🎤  Анализ",
    "📊  Результаты модели",
    "ℹ️  О проекте",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Анализ
# ══════════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown("## 🎙️ Детектор дипфейк-аудио")
    st.markdown("Загрузите файл `.wav` или `.mp3`, чтобы определить — это **настоящий** голос человека "
                "или **синтетическая / дипфейк** запись.")

    uploaded = st.file_uploader(
        label="Перетащите аудиофайл сюда или нажмите для выбора",
        # type=["wav","mp3"],
        type=None,
        label_visibility="collapsed",
    )

    if uploaded is not None:
        # проверка формата
        if not uploaded.name.lower().endswith(('.wav', '.mp3')):
            st.error("Неверный формат файла. Пожалуйста, загрузите файл в формате .wav или .mp3.")
            st.stop()

        with st.spinner("Анализируем запись…"):
            result = pred_module.predict(uploaded)

        label   = result["label"]
        conf    = result["confidence"]
        is_stub = result["stub_mode"]

        if is_stub:
            card_class = "card-stub"
            label      = "⚠️ Модель не загружена"
            conf_text  = "Поместите cnn_deepfake_detector.pt в папку /models"
        elif "Real" in label:
            card_class = "card-real"
            label      = "Настоящая запись"
            conf_text  = f"Уверенность: {conf:.1%}"
        else:
            card_class = "card-fake"
            label      = "Дипфейк"
            conf_text  = f"Уверенность: {conf:.1%}"

        st.markdown(f"""
        <div class="result-card {card_class}">
            <p class="result-label">{label}</p>
            <p class="result-conf">{conf_text}</p>
        </div>
        """, unsafe_allow_html=True)

        if not is_stub:
            st.markdown('<p class="section-title">Распределение вероятностей</p>',
                        unsafe_allow_html=True)
            st.pyplot(confidence_bar(result["prob_real"], result["prob_fake"]),
                      use_container_width=True)

        st.markdown("---")
        col_w, col_m = st.columns(2)

        with col_w:
            st.markdown('<p class="section-title">Форма волны</p>',
                        unsafe_allow_html=True)
            st.pyplot(fig_waveform(result["waveform"]), use_container_width=True)

        with col_m:
            st.markdown('<p class="section-title">Лог мел-спектрограмма</p>',
                        unsafe_allow_html=True)
            st.pyplot(fig_melspec(result["melspec"]), use_container_width=True)

        st.markdown("---")
        is_fake = not is_stub and "Дипфейк" in label
        st.markdown('<p class="section-title">Почему модель приняла такое решение?</p>',
                    unsafe_allow_html=True)

        if is_stub:
            st.caption("⚠️ Режим заглушки — тепловая карта ниже не является реальным предсказанием.")

        st.pyplot(
            fig_gradcam(result["melspec"], result["gradcam"], is_fake=is_fake),
            use_container_width=True,
        )

        with st.expander("Как читать этот график"):
            st.markdown("""
**Grad-CAM** (Gradient-weighted Class Activation Mapping) выделяет
временно-частотные области мел-спектрограммы, которые **сильнее всего повлияли**
на решение модели.

- **Горизонтальная ось** — время (секунды)
- **Вертикальная ось** — мел-частотные бины (низкие частоты снизу)
- **Яркий / тёплый цвет** — области, которые модель посчитала наиболее важными

Синтетическая речь часто проявляет нерегулярные паттерны энергии в **высокочастотном**
диапазоне (верхняя половина графика) или **неестественную периодичность** в отдельных
временных окнах. Настоящая речь, как правило, даёт более плавные и равномерные активации.
            """)

        st.markdown("---")
        uploaded.seek(0)
        file_type = uploaded.type  # 'audio/wav' или 'audio/mpeg'
        st.audio(uploaded.read(), format=file_type)

        # экспорт отчёта
        st.markdown("---")
        st.markdown('<p class="section-title">Экспорт отчёта</p>',
                    unsafe_allow_html=True)
        
        models_names = []
        models_acc = []
        models_f1 = []
        best_model = ""
        COMPARISON_FILE = "weights/comparison.npz"
        if os.path.exists(COMPARISON_FILE):
            data = np.load(COMPARISON_FILE, allow_pickle=True)
            models_names = [str(n) for n in data["model_names"]]
            models_acc = data["accuracy"].tolist()
            models_f1 = data["f1"].tolist()
            best_model = str(data["best_model"])

        if st.button("📄 Сформировать PDF-отчёт"):
            with st.spinner("Генерация PDF…"):
                # создаём фигуры заново
                _fig_wave     = fig_waveform(result["waveform"])
                _fig_mel      = fig_melspec(result["melspec"])
                _is_fake      = not is_stub and "Дипфейк" in label
                _fig_grad     = fig_gradcam(
                    result["melspec"], result["gradcam"], is_fake=_is_fake
                )
                _fig_conf_bar = confidence_bar(
                    result["prob_real"], result["prob_fake"]
                )

                pdf_bytes = build_analysis_report(
                    filename=uploaded.name,
                    label=label,
                    confidence=conf,
                    prob_real=result["prob_real"],
                    prob_fake=result["prob_fake"],
                    fig_wave=_fig_wave,
                    fig_mel=_fig_mel,
                    fig_grad=_fig_grad,
                    fig_conf_bar=_fig_conf_bar,
                    models_names=models_names, 
                    models_acc=models_acc,       
                    models_f1=models_f1,         
                    best_model=best_model,
                )

                # Освобождаем память
                for f in [_fig_wave, _fig_mel, _fig_grad, _fig_conf_bar]:
                    plt.close(f)

            st.download_button(
                label="⬇️ Скачать PDF",
                data=pdf_bytes,
                file_name=f"deepfake_report_{uploaded.name}.pdf",
                mime="application/pdf",
            )

    else:
        st.info("👆 Загрузите файл `.wav` или `.mp3` выше, чтобы начать анализ.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Сравнение моделей
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:
    import pandas as pd

    st.markdown("## 📊 Сравнение моделей")

    COMPARISON_FILE = "weights/comparison.npz"

    if os.path.exists(COMPARISON_FILE):
        data = np.load(COMPARISON_FILE, allow_pickle=True)

        names     = [str(n) for n in data["model_names"]]
        acc_vals  = data["accuracy"].tolist()
        f1_vals   = data["f1"].tolist()
        prec_vals = data["precision"].tolist()
        rec_vals  = data["recall"].tolist()
        auc_vals  = data["roc_auc"].tolist()
        best_name = str(data["best_model"])
        best_idx  = names.index(best_name) if best_name in names else int(np.argmax(f1_vals))

        # ── Таблица метрик ────────────────────────────────────────────────────
        st.markdown('<p class="section-title">Метрики на тестовой выборке</p>',
                    unsafe_allow_html=True)

        rows = []
        for i, name in enumerate(names):
            rows.append({
                "Модель":    ("🏆 " if name == best_name else "") + name,
                "Accuracy":  acc_vals[i],
                "F1-мера":   f1_vals[i],
                "Precision": prec_vals[i],
                "Recall":    rec_vals[i],
                "AUC-ROC":   auc_vals[i],
            })
        df = pd.DataFrame(rows)

        # Цветовая шкала по F1 — сразу видно кто лучше
        st.dataframe(
            df.style.background_gradient(
                subset=["Accuracy", "F1-мера", "Precision", "Recall", "AUC-ROC"],
                cmap="Greens", vmin=0.5, vmax=1.0,
            ).format({
                "Accuracy": "{:.3f}", "F1-мера": "{:.3f}",
                "Precision": "{:.3f}", "Recall": "{:.3f}", "AUC-ROC": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Лучшая модель по F1: **{best_name}**  "
                   f"(Accuracy {acc_vals[best_idx]:.3f} · "
                   f"F1 {f1_vals[best_idx]:.3f} · "
                   f"AUC {auc_vals[best_idx]:.3f})")

        st.markdown("---")

        # ── Grouped bar chart ─────────────────────────────────────────────────
        st.markdown('<p class="section-title">Визуальное сравнение</p>',
                    unsafe_allow_html=True)

        metrics_to_plot = {
            "Accuracy":  acc_vals,
            "F1":        f1_vals,
            "Precision": prec_vals,
            "Recall":    rec_vals,
        }
        x      = np.arange(len(names))
        width  = 0.2
        colors = ["#4C9BE8", "#2ecc71", "#e67e22", "#9b59b6"]

        fig, ax = plt.subplots(figsize=(8, 3.8))
        for i, (label, vals) in enumerate(metrics_to_plot.items()):
            bars = ax.bar(x + i * width, vals, width,
                          label=label, color=colors[i], alpha=0.85)
            # Подпись значений над столбцами лучшей модели
            for j, (bar, v) in enumerate(zip(bars, vals)):
                if j == best_idx:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{v:.2f}", ha="center", va="bottom",
                            fontsize=6.5, fontweight="bold", color="#333")

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(names, fontsize=8.5)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Значение", fontsize=9)
        ax.set_title("Сравнение метрик — все модели", fontweight="bold")
        ax.legend(fontsize=8, ncol=4, loc="upper left")
        ax.axvline(best_idx + width * 1.5, color="#e74c3c",
                   lw=1.2, ls="--", alpha=0.5, label="_nolegend_")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.markdown("---")

        # ── Матрица ошибок + ROC лучшей модели ───────────────────────────────
        st.markdown(f'<p class="section-title">Детальный анализ лучшей модели — {best_name}</p>',
                    unsafe_allow_html=True)

        col_cm, col_roc = st.columns(2)

        with col_cm:
            if "confusion_matrix" in data:
                cm = data["confusion_matrix"]
                fig, ax = plt.subplots(figsize=(4, 3.5))
                im = ax.imshow(cm, cmap="Blues")
                fig.colorbar(im, ax=ax)
                ax.set_xticks([0, 1]); ax.set_xticklabels(["Дипфейк", "Настоящая"])
                ax.set_yticks([0, 1]); ax.set_yticklabels(["Дипфейк", "Настоящая"])
                ax.set_xlabel("Предсказано", fontsize=9)
                ax.set_ylabel("Реально", fontsize=9)
                ax.set_title("Матрица ошибок", fontweight="bold")
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, str(int(cm[i, j])),
                                ha="center", va="center", fontsize=16,
                                color="white" if cm[i, j] > cm.max() / 2 else "black")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)

                # TN / FP / FN / TP расшифровка
                tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
                st.caption(
                    f"TP {tp} · TN {tn} · FP {fp} (ложная тревога) · FN {fn} (пропущен дипфейк)"
                )

        with col_roc:
            if "roc_fpr" in data and "roc_tpr" in data:
                fpr = data["roc_fpr"]
                tpr = data["roc_tpr"]
                roc_auc = auc_vals[best_idx]
                fig, ax = plt.subplots(figsize=(4, 3.5))
                ax.plot(fpr, tpr, color="#4C9BE8", lw=2,
                        label=f"AUC = {roc_auc:.3f}")
                ax.fill_between(fpr, tpr, alpha=0.08, color="#4C9BE8")
                ax.plot([0, 1], [0, 1], "k--", lw=1, label="Случайный классификатор")
                ax.set_xlabel("Доля ложных срабатываний (FPR)", fontsize=8)
                ax.set_ylabel("Доля верных срабатываний (TPR)", fontsize=8)
                ax.set_title("ROC-кривая", fontweight="bold")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)

    else:
        st.info("Результаты появятся здесь после обучения и оценки всех моделей.")
        st.markdown("""
**Шаги для получения результатов:**
```bash
cd app
python train.py       # обучение всех 6 моделей
python evaluate.py    # оценка → weights/comparison.npz
```

| # | Модель | Признаки | Тип |
|---|--------|----------|-----|
| 1 | RandomForest | Усреднённые MFCC | Бейзлайн |
| 2 | SVM (RBF) | Усреднённые MFCC | Бейзлайн |
| 3 | CNN | Мел-спектрограмма | Нейросеть |
| 4 | BiLSTM | Последовательность MFCC | Нейросеть |
| 5 | CNN + LSTM | Мел-спектрограмма | Нейросеть |
| 6 | wav2vec 2.0 | Сырой waveform | Предобученная |
        """)



# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — О проекте
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("## ℹ️ О проекте")

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("""
### Что такое дипфейк-аудио?

Дипфейк-аудио — это синтетическая речь, сгенерированная системами искусственного
интеллекта: системами синтеза речи (TTS) или моделями конверсии голоса, способными
убедительно имитировать голос конкретного человека.
Такие технологии несут риски в сфере мошенничества, дезинформации и кражи личных данных.

### Как работает детектор?

1. **Предобработка** — аудио ресэмплируется до 16 кГц и обрезается / дополняется до
   фиксированного окна в 3 секунды.
2. **Извлечение признаков** — вычисляется логарифмическая мел-спектрограмма (128 мел-бинов).
   Она представляет аудио в виде двумерного изображения, кодирующего частотное содержимое во времени.
3. **Классификация** — лёгкая двумерная свёрточная нейросеть (CNN) обрабатывает
   спектрограмму и выдаёт вероятность для каждого класса (*настоящая* / *дипфейк*).

### Архитектура модели

| Блок           | Размер выхода |
|----------------|---------------|
| Вход           | (1, 128, 94)  |
| ConvBlock × 4  | (256, 8, 6)   |
| GlobalAvgPool  | (256,)        |
| FC → ReLU      | (128,)        |
| FC (выход)     | (2,)          |

Обучаемых параметров: **~600 К**
        """)

    with col_b:
        st.markdown("""
### Параметры

| Параметр         | Значение  |
|------------------|-----------|
| Частота дискр.   | 16 000 Гц |
| Длина окна       | 3 с       |
| Мел-бины         | 128       |
| Размер FFT       | 1024      |
| Шаг (hop)        | 512       |

### Стек технологий

- **Python** 3
- **PyTorch**
- **librosa**
- **Streamlit**
- **Docker**
        """)

    st.markdown("---")
    st.caption("Детектор дипфейк-аудио · Дипломная работа")
