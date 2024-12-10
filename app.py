import streamlit as st
import pandas as pd
import numpy as np
import json
import torch
from io import StringIO
import tempfile
import os

from transformers import AutoTokenizer, AutoModelForCausalLM

# Импортируем наши модули генерации данных
from modules.text_generator import generate_text_data
from modules.table_generator import generate_table_data_advanced
from modules.timeseries_generator import generate_timeseries

# Настройки страницы
st.set_page_config(page_title="Генератор синтетических данных", layout="wide")

st.title("Генератор синтетических данных")

@st.cache_resource
def load_gpt2_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_gpt2_model()

# Инициализация session_state для сохранения конфигураций табличных данных
if "table_config" not in st.session_state:
    st.session_state.table_config = {
        "num_samples_table": 10,
        "locale_table": "en_US",
        "num_columns": 3,
        "columns_config": [],
        "correlations": None
    }

# Вкладки приложения
tab_text, tab_table, tab_timeseries, tab_sample, tab_config, tab_about = st.tabs([
    "Текстовые данные",
    "Табличные данные",
    "Временные ряды",
    "Имитация по образцу",
    "Конфигурации",
    "О программе"
])

###############################################
# Вкладка Текстовых данных
###############################################
with tab_text:
    st.header("Генерация текстовых данных")
    st.markdown("""
    Здесь вы можете сгенерировать текстовые данные при помощи:
    - **Faker**: простая генерация фейковых предложений, абзацев, имен и адресов.
    - **GPT-2**: генерация текста с помощью предобученной модели.
    """)

    generation_type = st.selectbox(
        "Способ генерации:", 
        ["Faker", "GPT-2 (ML-модель)"], 
        help="Выберите, использовать ли простую фейковую генерацию или ML-модель GPT-2."
    )
    
    if generation_type == "Faker":
        st.markdown("**Параметры Faker**")
        num_samples_text = st.number_input("Количество строк для генерации", min_value=1, value=10, help="Сколько текстовых строк будет сгенерировано.")
        locale = st.selectbox("Локаль", ["en_US", "ru_RU", "fr_FR"], help="Локаль определяет язык и формат фейковых данных.")
        text_type = st.selectbox("Тип текста", ["sentence", "paragraph", "name", "address"], help="Выберите тип фейкового текста.")
        
        st.markdown("---")
        if st.button("Сгенерировать текстовые данные"):
            df_text = generate_text_data(num_samples=num_samples_text, locale=locale, text_type=text_type)
            st.write("**Пример сгенерированных данных:**")
            st.dataframe(df_text)
            
            csv_text = df_text.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Скачать данные в CSV",
                data=csv_text,
                file_name='synthetic_text_data.csv',
                mime='text/csv',
                help="Скачать сгенерированные данные"
            )
    else:
        st.markdown("**Параметры GPT-2**")
        prompt = st.text_area("Начальная фраза (prompt)", value="Once upon a time", help="Введите начальный текст (промпт).")
        max_length = st.slider("Максимальная длина", 10, 200, 50, help="Максимальное число токенов.")
        temperature = st.slider("Температура", 0.0, 1.5, 1.0, help="Контролирует креативность модели.")
        top_k = st.slider("top_k", 0, 100, 50, help="Ограничивает выбор следующих токенов.")
        top_p = st.slider("top_p", 0.0, 1.0, 0.9, help="Ограничивает выбор токенов по совокупной вероятности.")
        
        st.markdown("---")
        if st.button("Сгенерировать текст (GPT-2)"):
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            with torch.no_grad():
                output = model.generate(
                    input_ids, 
                    max_length=max_length, 
                    temperature=temperature, 
                    top_k=top_k, 
                    top_p=top_p,
                    do_sample=True
                )
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            st.write("**Сгенерированный текст:**")
            st.write(generated_text)

            df_gpt2 = pd.DataFrame([generated_text], columns=["generated_text"])
            csv_gpt2 = df_gpt2.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Скачать сгенерированный текст",
                data=csv_gpt2,
                file_name='synthetic_text_gpt2.csv',
                mime='text/csv'
            )

###############################################
# Вспомогательные функции для Табличных данных
###############################################

def get_normal_numeric_columns(columns_config):
    return [c for c in columns_config if c["type"] == "numeric" and c["dist"] == "normal"]

def generate_large_table_csv(num_samples, columns_config, locale, correlations=None, chunk_size=100000):
    """
    Генерирует большой датасет в CSV с помощью чанков.
    Возвращает путь к временному файлу CSV.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    temp_path = temp_file.name
    temp_file.close()

    num_chunks = (num_samples // chunk_size) + (1 if num_samples % chunk_size != 0 else 0)
    progress_bar = st.progress(0)

    samples_generated = 0
    for chunk_idx in range(num_chunks):
        rows_in_this_chunk = min(chunk_size, num_samples - samples_generated)
        if rows_in_this_chunk <= 0:
            break

        df_chunk = generate_table_data_advanced(
            num_samples=rows_in_this_chunk,
            columns_config=columns_config,
            locale=locale,
            correlations=correlations
        )
        df_chunk.to_csv(temp_path, mode='a', index=False, header=(chunk_idx == 0))

        samples_generated += rows_in_this_chunk
        progress_bar.progress((chunk_idx+1)/num_chunks)

    return temp_path

def convert_np_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_np_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_to_list(x) for x in obj]
    return obj


###############################################
# Табличные данные
###############################################
with tab_table:
    st.header("Генерация табличных данных (расширенная конфигурация)")
    st.markdown("""
    Здесь вы можете сгенерировать табличные данные с различными типами столбцов, задавать распределения, категории и корреляции.
    Можно также включить режим для больших объёмов данных, чтобы генерировать их порционно.
    """)

    big_data_mode = st.checkbox("Включить режим больших данных", help="При включении данные генерируются порционно и сразу пишутся в файл.")

    num_samples_table = st.number_input("Количество строк", min_value=1, value=st.session_state.table_config["num_samples_table"], help="Сколько строк будет в итоговой таблице.")
    locale_table = st.selectbox("Локаль для категориальных данных", ["en_US", "ru_RU", "fr_FR"], index=["en_US","ru_RU","fr_FR"].index(st.session_state.table_config["locale_table"]))

    num_columns = st.number_input("Количество столбцов", min_value=1, value=st.session_state.table_config["num_columns"], help="Сколько столбцов будет в сгенерированной таблице.")

    # session_state
    st.session_state.table_config["num_samples_table"] = num_samples_table
    st.session_state.table_config["locale_table"] = locale_table
    st.session_state.table_config["num_columns"] = num_columns

    st.markdown("---")
    st.markdown("**Настройки столбцов**")

    # Инициализация столбцов, если не была произведена
    for i in range(num_columns):
        col_conf_key = f"col_conf_{i}"
        if col_conf_key not in st.session_state:
            st.session_state[col_conf_key] = {
                "name": f"col_{i+1}",
                "type": "numeric",
                "dist": "normal",
                "mean": 0.0,
                "std": 1.0
            }

    # Рендерим настройки столбцов
    columns_config = []
    for i in range(num_columns):
        col_conf_key = f"col_conf_{i}"
        with st.expander(f"Столбец {i+1}", expanded=True):
            c = st.session_state[col_conf_key]
            c["name"] = st.text_input("Имя столбца", value=c["name"], key=f"col_name_{i}")
            c["type"] = st.selectbox("Тип столбца", ["numeric","categorical"], index=["numeric","categorical"].index(c["type"]), key=f"col_type_{i}")

            if c["type"] == "numeric":
                c["dist"] = st.selectbox("Распределение", ["normal","uniform"], index=["normal","uniform"].index(c.get("dist","normal")), key=f"dist_{i}")
                if c["dist"] == "normal":
                    c["mean"] = st.number_input("Среднее", value=c.get("mean",0.0), key=f"mean_{i}")
                    c["std"] = st.number_input("Стд. отклонение", value=c.get("std",1.0), key=f"std_{i}")
                    # Удаляем возможные ключи от uniform
                    c.pop("min_val", None)
                    c.pop("max_val", None)
                else:
                    c["min_val"] = st.number_input("Минимум", value=c.get("min_val",0.0), key=f"min_val_{i}")
                    c["max_val"] = st.number_input("Максимум", value=c.get("max_val",1.0), key=f"max_val_{i}")
                    # Удаляем ключи mean/std
                    c.pop("mean", None)
                    c.pop("std", None)
            else:
                c["faker_type"] = st.selectbox("Тип фейковых данных", ["name", "city", "company", "custom_categories"], index=["name","city","company","custom_categories"].index(c.get("faker_type","name")), key=f"faker_{i}")
                if c["faker_type"] == "custom_categories":
                    cat_str = st.text_area("Категории (через запятую)", value=",".join(c.get("categories",["cat1","cat2"])), key=f"cats_{i}")
                    categories = [x.strip() for x in cat_str.split(",")]
                    c["categories"] = categories
                else:
                    c.pop("categories", None)

            # Обновляем session_state
            st.session_state[col_conf_key] = c
            columns_config.append(c)

    # Если есть несколько нормальных числовых столбцов, предложим ввести корреляции
    normal_numeric_columns = get_normal_numeric_columns(columns_config)
    correlations = None
    if len(normal_numeric_columns) > 1:
        st.markdown("**Настройка корреляций для нормальных числовых столбцов:**")
        n = len(normal_numeric_columns)
        corr_inputs = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(1.0)
                elif j > i:
                    val = st.number_input(
                        f"Corr({normal_numeric_columns[i]['name']}, {normal_numeric_columns[j]['name']})", 
                        min_value=-1.0, max_value=1.0, value=0.0, step=0.1,
                        key=f"corr_{i}_{j}"
                    )
                    row.append(val)
                else:
                    row.append(None)
            corr_inputs.append(row)

        # Заполняем симметрично
        corr_matrix = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr_matrix[i,j] = 1.0
                elif j > i:
                    corr_matrix[i,j] = corr_inputs[i][j]
                else:
                    corr_matrix[i,j] = corr_matrix[j,i]

        correlations = corr_matrix
    else:
        correlations = None

    st.session_state.table_config["columns_config"] = columns_config
    st.session_state.table_config["correlations"] = correlations

    if big_data_mode:
        chunk_size = st.number_input("Размер чанка (число строк)", min_value=1000, value=100000)
    
    if st.button("Сгенерировать табличные данные"):
        if big_data_mode:
            csv_path = generate_large_table_csv(
                num_samples=st.session_state.table_config["num_samples_table"], 
                columns_config=st.session_state.table_config["columns_config"], 
                locale=st.session_state.table_config["locale_table"],
                correlations=st.session_state.table_config["correlations"],
                chunk_size=chunk_size
            )
            
            st.success("Генерация завершена.")
            with open(csv_path, 'rb') as f:
                st.download_button("Скачать CSV", data=f, file_name='big_synthetic_data.csv', mime='text/csv')
            os.remove(csv_path)
        else:
            df_table = generate_table_data_advanced(
                num_samples=st.session_state.table_config["num_samples_table"], 
                columns_config=st.session_state.table_config["columns_config"], 
                locale=st.session_state.table_config["locale_table"],
                correlations=st.session_state.table_config["correlations"]
            )
            st.write("**Пример сгенерированных данных (первые 50 строк):**")
            st.dataframe(df_table.head(50))
            
            csv_table = df_table.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Скачать данные в CSV",
                data=csv_table,
                file_name='synthetic_table_data_advanced.csv',
                mime='text/csv'
            )

###############################################
# Временные ряды
###############################################
with tab_timeseries:
    st.header("Генерация временного ряда")
    st.markdown("""
    Сгенерируйте синтетический временной ряд, настраивая тренд, шум и сезонность.
    """)
    num_points = st.number_input("Количество точек", min_value=1, value=50, help="Длина временного ряда.")
    trend = st.slider("Сила тренда", 0.0, 5.0, 1.0, help="Наклон линейного тренда.")
    noise_level = st.slider("Уровень шума", 0.0, 1.0, 0.1, help="Стандартное отклонение случайного шума.")
    seasonality = st.slider("Амплитуда сезонности", 0.0, 5.0, 0.0, help="Амплитуда синусоидальной сезонности.")
    
    if st.button("Сгенерировать временной ряд"):
        df_ts = generate_timeseries(num_points=num_points, trend=trend, noise_level=noise_level, seasonality=seasonality)
        st.write("**Сгенерированный временной ряд:**")
        st.line_chart(df_ts.set_index("time"))
        
        csv_ts = df_ts.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Скачать данные в CSV",
            data=csv_ts,
            file_name='synthetic_timeseries_data.csv',
            mime='text/csv'
        )

###############################################
# Имитации по образцу
###############################################
with tab_sample:
    st.header("Имитация данных по образцу")
    st.markdown("""
    Загрузите свой CSV-файл, и приложение проанализирует его, определит типы столбцов,
    рассчитает статистику и поможет сгенерировать схожий синтетический датасет.
    """)

    uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.subheader("Анализ загруженного датасета:")
        st.write(df_input.head())
        
        columns_analysis = []
        for col in df_input.columns:
            col_data = df_input[col]
            if pd.api.types.is_numeric_dtype(col_data):
                mean = col_data.mean()
                std = col_data.std(ddof=1) if len(col_data)>1 else 0
                min_val = col_data.min()
                max_val = col_data.max()
                columns_analysis.append({
                    "name": col,
                    "detected_type": "numeric",
                    "mean": mean,
                    "std": std,
                    "min_val": min_val,
                    "max_val": max_val
                })
            else:
                unique_vals = col_data.unique()
                # Если уникальных не слишком много, считаем категориальным
                if len(unique_vals) < 50:
                    columns_analysis.append({
                        "name": col,
                        "detected_type": "categorical",
                        "categories": unique_vals.tolist()
                    })
                else:
                    # Много уникальных - тоже категор. но большое количество
                    columns_analysis.append({
                        "name": col,
                        "detected_type": "categorical",
                        "categories": unique_vals[:50].tolist()
                    })

        st.subheader("Предложенные параметры:")
        synthetic_config = []
        for i, c in enumerate(columns_analysis):
            st.write(f"**Столбец: {c['name']}**")
            if c['detected_type'] == "numeric":
                dist = st.selectbox(f"Распределение для {c['name']}", ["normal", "uniform"], key=f"sample_dist_{i}")
                if dist == "normal":
                    mean = st.number_input(f"Среднее для {c['name']}", value=float(c['mean']), key=f"sample_mean_{i}")
                    std = st.number_input(f"Стд. отклонение для {c['name']}", value=float(c['std']), key=f"sample_std_{i}")
                    synthetic_config.append({
                        "name": c['name'],
                        "type": "numeric",
                        "dist": dist,
                        "mean": mean,
                        "std": std
                    })
                else:
                    min_val = st.number_input(f"Минимум для {c['name']}", value=float(c['min_val']), key=f"sample_min_{i}")
                    max_val = st.number_input(f"Максимум для {c['name']}", value=float(c['max_val']), key=f"sample_max_{i}")
                    synthetic_config.append({
                        "name": c['name'],
                        "type": "numeric",
                        "dist": dist,
                        "min_val": min_val,
                        "max_val": max_val
                    })
            else:
                cat_str = st.text_area(f"Категории для {c['name']} (через запятую)", value=", ".join(map(str,c['categories'])), key=f"sample_cats_{i}")
                categories = [x.strip() for x in cat_str.split(",")]
                synthetic_config.append({
                    "name": c['name'],
                    "type": "categorical",
                    "categories": categories
                })

        num_synthetic_rows = st.number_input("Количество строк в синтетическом датасете", min_value=1, value=len(df_input))
        locale_sample = st.selectbox("Локаль для категориальных данных", ["en_US", "ru_RU", "fr_FR"], key="locale_sample")
        
        if st.button("Сгенерировать синтетические данные на основе образца"):
            df_synth = generate_table_data_advanced(num_samples=num_synthetic_rows, columns_config=synthetic_config, locale=locale_sample)
            st.dataframe(df_synth.head(50))
            
            csv_synth = df_synth.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Скачать синтетический датасет",
                data=csv_synth,
                file_name='synthetic_from_sample.csv',
                mime='text/csv'
            )

###############################################
# Конфигурации
###############################################
with tab_config:
    st.header("Сохранение и загрузка конфигураций")
    st.markdown("""
    Здесь вы можете сохранить текущие настройки генерации данных в файл (JSON), а затем загрузить их позже.
    """)

    # Сохранение конфигурации
    if st.button("Сохранить текущую конфигурацию"):
        config_to_save = {
            "table_config": st.session_state.table_config
        }
        config_to_save = convert_np_to_list(config_to_save)
        config_json = json.dumps(config_to_save, indent=4)
        st.download_button(
            "Скачать конфигурацию",
            data=config_json.encode('utf-8'),
            file_name="config.json",
            mime="application/json"
        )

    # Загрузка конфигурации
    uploaded_config = st.file_uploader("Загрузить конфигурацию (JSON)", type=["json"])
    if uploaded_config is not None:
        config_data = json.load(uploaded_config)
        st.session_state.table_config = config_data.get("table_config", st.session_state.table_config)
        st.success("Конфигурация загружена! Нажмите 'Применить конфигурацию' для обновления интерфейса.")

        if st.button("Применить конфигурацию"):
            st.experimental_rerun()

###############################################
# О программе
###############################################
with tab_about:
    st.header("О программе")
    st.markdown("""
    Это приложение генерирует синтетические данные для обучения и тестирования моделей машинного обучения.
    
    **Основные возможности:**
    - Генерация текстовых данных (Faker, GPT-2)
    - Генерация табличных данных с гибкой настройкой и возможностью задать корреляции между нормальными числовыми столбцами
    - Генерация временных рядов с параметрами (тренд, шум, сезонность)
    - Имитация данных по образцу из загруженного CSV
    - Сохранение и загрузка конфигураций в/из JSON
    - Оптимизация для больших объёмов данных (генерация по чанкам)
    
    """)

