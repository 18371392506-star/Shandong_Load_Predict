# -*- coding: utf-8 -*-
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from datetime import timedelta, datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

import matplotlib
from matplotlib import font_manager

# ============= Streamlit 页面基本设置 =============
st.set_page_config(
    page_title="山东省电力负荷预测（XGBoost+节假日修正）",
    layout="wide",
)

st.title("山东省电力负荷预测平台")
st.markdown("""
本工具对**增量客户+存量客户**的总用电量进行小时级预测：
- 自动汇总多用户小时用电量，与气象数据对齐；
- 预测特征包含：日期偏移、星期、小时、月份、年中第几天、周末、**各法定节假日独立标识**、归一化温度；
- **春节修正**：先用初步模型预测春节期间负荷并替换原始数据，再重新训练，消除春节异常波动影响；
- 提供测试集评估与未来预测结果导出；
- **重要**：最终预测未来时，模型会使用全部历史数据（包括测试集）进行训练，确保最近趋势被捕捉。
""")

# ============= Matplotlib 中文字体 =============
def set_matplotlib_chinese():
    candidates = [
        "SimHei", "Microsoft YaHei", "Microsoft JhengHei",
        "Noto Sans CJK SC", "Noto Sans CJK", "Noto Sans SC",
        "PingFang SC", "Heiti SC", "STHeiti",
        "WenQuanYi Zen Hei", "Arial Unicode MS"
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break
    if chosen is None:
        matplotlib.rcParams["axes.unicode_minus"] = False
        return
    matplotlib.rcParams["font.sans-serif"] = [chosen]
    matplotlib.rcParams["axes.unicode_minus"] = False

set_matplotlib_chinese()

# =========================
# 1. 用户上传区
# =========================
st.sidebar.header("数据上传")

uploaded_power_files = st.sidebar.file_uploader(
    "1) 上传多个用户小时级用电量 Excel（可多选）",
    type=["xlsx", "xls"],
    accept_multiple_files=True
)

uploaded_weather = st.sidebar.file_uploader(
    "2) 上传气象数据文档（含历史与未来，表头：record_time, value）",
    type=["xlsx", "xls", "csv"]
)

test_days = st.sidebar.number_input(
    "测试集天数（最近 N 天作为测试集，仅用于评估，不影响最终未来预测模型）",
    min_value=1,
    max_value=30,
    value=5,
    step=1
)

run_button = st.sidebar.button("开始处理 & 训练 & 预测")

# =========================
# 2. 公共工具函数
# =========================

HOUR_COLUMNS = [
    '01:00', '02:00', '03:00', '04:00', '05:00', '06:00',
    '07:00', '08:00', '09:00', '10:00', '11:00', '12:00',
    '13:00', '14:00', '15:00', '16:00', '17:00', '18:00',
    '19:00', '20:00', '21:00', '22:00', '23:00', '24:00'
]

def read_power_file(file):
    """读取单个用户用电 Excel，自动识别工作表名（客户详细用电量 / 用户详细用电量）"""
    xl = pd.ExcelFile(file)
    sheet_name = None
    for s in xl.sheet_names:
        if "客户详细用电量" in s or "用户详细用电量" in s:
            sheet_name = s
            break
    if sheet_name is None:
        raise ValueError("文件中未找到名为“客户详细用电量”或“用户详细用电量”的工作表")
    df = pd.read_excel(xl, sheet_name=sheet_name)
    if "日期" not in df.columns or "用户名称" not in df.columns:
        raise ValueError("工作表中缺少“日期”或“用户名称”列")
    return df

def parse_weather_file(file):
    """读取新的气象文件格式：record_time (2026/1/1 0:00:00), value"""
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df.columns = df.columns.str.strip()
    if "record_time" not in df.columns or "value" not in df.columns:
        raise ValueError("气象文件必须包含 'record_time' 和 'value' 列")
    df["record_time"] = pd.to_datetime(df["record_time"], format="%Y/%m/%d %H:%M:%S")
    df["date"] = df["record_time"].dt.normalize()
    df["hour"] = df["record_time"].dt.hour
    df = df.rename(columns={"value": "temperature"})
    return df[["date", "hour", "temperature"]]

def split_weather_by_power_dates(df_weather, power_start_date, power_end_date):
    """
    根据电量数据的起止日期划分天气数据：
    - 训练集天气：电量最早日期 至 电量最后日期
    - 预测集天气：训练集截止日之后的所有天气数据
    """
    train_start_date = power_start_date
    train_end_date = power_end_date
    train_weather = df_weather[(df_weather["date"] >= train_start_date) & (df_weather["date"] <= train_end_date)].copy()
    future_weather = df_weather[df_weather["date"] > train_end_date].copy()
    return train_weather, future_weather

def process_power_files_get_total(file_list):
    """
    处理所有用电文件，仅保留最后一天出现的客户，汇总所有小时用电量。
    返回：
    - total_power_hourly: DataFrame，总用电小时数据（时间列，用电量列）
    - power_date_range: (最早日期, 最晚日期)
    - last_day_customers: 最后一天出现的所有用户名称列表
    """
    all_data = []
    for file in file_list:
        df = read_power_file(file)
        df["日期"] = pd.to_datetime(df["日期"])
        df["用户名称"] = df["用户名称"].astype(str).str.strip()
        all_data.append(df)

    if not all_data:
        return pd.DataFrame(), (None, None), []

    combined_df = pd.concat(all_data, ignore_index=True)
    power_start_date = combined_df["日期"].min()
    power_end_date = combined_df["日期"].max()
    st.write(f"电量数据日期范围：{power_start_date.date()} 至 {power_end_date.date()}")

    # 找出最后一天的所有用户
    last_day_df = combined_df[combined_df["日期"] == power_end_date]
    last_day_customers = last_day_df["用户名称"].unique().tolist()
    st.write(f"最后一天有电量的客户数量：{len(last_day_customers)}")

    # 仅保留这些客户的所有历史数据
    combined_df = combined_df[combined_df["用户名称"].isin(last_day_customers)]

    # 汇总所有用户的小时用电量
    grouped = combined_df.groupby("日期")[HOUR_COLUMNS].sum()
    rows = []
    for date, row in grouped.iterrows():
        for hour_col in HOUR_COLUMNS:
            hour = int(hour_col.split(":")[0])
            if hour == 24:   # 电量中的24:00 -> 下一天0:00
                actual_date = date + timedelta(days=1)
                actual_hour = 0
            else:
                actual_date = date
                actual_hour = hour
            time_str = f"{actual_date.strftime('%Y/%m/%d')} {actual_hour:02d}:00"
            rows.append({"时间": time_str, "用电量": row[hour_col]})
    df_out = pd.DataFrame(rows)
    df_out["时间_排序"] = pd.to_datetime(df_out["时间"], format="%Y/%m/%d %H:%M")
    df_out = df_out.sort_values("时间_排序").reset_index(drop=True)
    return df_out[["时间", "用电量"]], (power_start_date, power_end_date), last_day_customers

def build_train_df(weather_df, power_hourly_df, col_name="total_load"):
    """
    输入：
    - weather_df: 包含 date, hour, temperature
    - power_hourly_df: 包含 时间, 用电量，小时已对齐为0-23
    - col_name: 电量列的名称
    返回 DataFrame 包含 date, hour, temperature, 以及指定的电量列
    """
    if power_hourly_df.empty:
        df = weather_df[["date", "hour", "temperature"]].copy()
        df[col_name] = 0.0
        return df

    df_power = power_hourly_df.copy()
    df_power["时间_dt"] = pd.to_datetime(df_power["时间"], format="%Y/%m/%d %H:%M")
    df_power["date"] = df_power["时间_dt"].dt.normalize()
    df_power["hour"] = df_power["时间_dt"].dt.hour
    df_power = df_power[["date", "hour", "用电量"]].rename(columns={"用电量": col_name})

    df = weather_df.merge(df_power, on=["date", "hour"], how="left")
    df[col_name] = df[col_name].fillna(0.0)
    return df

# =========================
# 3. 手动节假日定义 (2026年) + 节后天数特征
# =========================
HOLIDAYS_2026 = {
    '元旦': ('2026-01-01', '2026-01-03'),
    '春节': ('2026-02-15', '2026-02-23'),
    '清明节': ('2026-04-04', '2026-04-06'),
    '劳动节': ('2026-05-01', '2026-05-05'),
    '端午节': ('2026-06-19', '2026-06-21'),
    '中秋节': ('2026-09-25', '2026-09-27'),
    '国庆节': ('2026-10-01', '2026-10-07'),
}

def add_manual_holiday_features(df):
    """
    为DataFrame添加周末特征、各节假日独立标识列，以及距最近节假日结束的天数
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # 周末特征
    df["is_weekend"] = (df["date"].dt.weekday >= 5).astype(int)

    # 初始化各节假日列
    for holiday_name in HOLIDAYS_2026.keys():
        df[f"is_{holiday_name}"] = 0

    # 标记各节假日区间，并记录所有区间用于后续计算
    holiday_ranges = []
    for holiday_name, (start_str, end_str) in HOLIDAYS_2026.items():
        start = pd.to_datetime(start_str)
        end = pd.to_datetime(end_str)
        holiday_ranges.append((start, end))
        mask = (df["date"] >= start) & (df["date"] <= end)
        df.loc[mask, f"is_{holiday_name}"] = 1

    # 计算距最近节假日结束的天数（如果当天正在放假，则为0；如果从未放假，则为一个大数）
    df["days_after_holiday"] = 30  # 默认视为超过一个月无影响
    for idx, row in df.iterrows():
        d = row["date"]
        # 检查是否处于任何节假日内
        in_holiday = False
        min_days = 999
        for start, end in holiday_ranges:
            if start <= d <= end:
                in_holiday = True
                break
            elif end < d:
                days = (d - end).days
                if days < min_days:
                    min_days = days
        if in_holiday:
            df.at[idx, "days_after_holiday"] = 0
        elif min_days != 999:
            df.at[idx, "days_after_holiday"] = min_days
        # else 保持默认30

    return df

# =========================
# 4. XGBoost 模型类（含春节修正 + 全量数据训练）
# =========================

class CorrectedXGBoost:
    """
    使用 XGBoost 对小时级总负荷进行预测。
    支持先用初步模型预测春节期间负荷，替换原始数据中的真实值，再重新训练。
    特征：day_idx, dow, hour, month, day_of_year, is_weekend, 各节假日标识, days_after_holiday, temp_norm
    """
    def __init__(self, target_col='total_load', test_days=5):
        self.target_col = target_col
        self.test_days = test_days
        self.df_raw = None          # 原始训练数据（已对齐气象和电量）
        self.df_corrected = None    # 春节修正后的数据（全量）
        self.hours = np.arange(24)
        self.model = None           # 用于评估的模型（基于训练集）
        self.final_model = None     # 用于未来预测的最终模型（基于全量数据）
        self.feat_cols = []         # 特征列名列表
        self.min_date = None        # 用于计算 day_idx
        self.temp_stats = (0, 1)    # (mean, std) 用于归一化温度
        self.spring_range = (pd.to_datetime('2026-02-15'), pd.to_datetime('2026-02-23'))

    @staticmethod
    def _normalize_columns(df):
        df = df.copy()
        df.columns = df.columns.str.strip()
        if "data" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"data": "date"})
        return df

    def prepare_data(self, df):
        """准备数据：保留必要列，排序，存储为 self.df_raw"""
        df = self._normalize_columns(df)
        if self.target_col not in df.columns:
            raise ValueError(f"原始数据中缺少指定的电量列: '{self.target_col}'")

        required = ["date", "hour", self.target_col]
        miss = [c for c in required if c not in df.columns]
        if miss:
            raise ValueError(f"原始数据缺少必要列: {miss}")

        cols_to_keep = required.copy()
        if "temperature" in df.columns:
            cols_to_keep.append("temperature")

        df = df.loc[:, cols_to_keep].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "hour"]).reset_index(drop=True)
        self.df_raw = df
        return self

    def _build_features(self, df):
        """为DataFrame添加所有特征列"""
        df = df.copy()
        df = add_manual_holiday_features(df)
        df["day_idx"] = (df["date"] - self.min_date).dt.days
        df["dow"] = df["date"].dt.weekday
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear

        if "temperature" in df.columns:
            mean_t, std_t = self.temp_stats
            df["temp_norm"] = (df["temperature"] - mean_t) / std_t
        else:
            df["temp_norm"] = 0.0
        return df

    def _fit_model(self, df_feat, train_dates):
        """使用指定训练日期训练XGBoost模型"""
        df_train = df_feat[df_feat["date"].isin(train_dates)]
        if df_train.empty:
            st.warning("训练集为空，无法训练模型。")
            return None
        X_train = df_train[self.feat_cols].values
        y_train = df_train[self.target_col].values
        model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

    def _prepare_feature_columns(self, df_feat):
        """确定特征列名（基于构建后的DataFrame）"""
        holiday_cols = [f"is_{h}" for h in HOLIDAYS_2026.keys()]
        self.feat_cols = ["day_idx", "dow", "hour", "month", "day_of_year",
                          "is_weekend", "days_after_holiday"] + holiday_cols + ["temp_norm"]
        # 确保所有特征列都存在
        for col in self.feat_cols:
            if col not in df_feat.columns:
                df_feat[col] = 0.0
        return df_feat

    def correct_spring_festival(self):
        """
        使用全量数据（self.df_raw）进行春节修正，并将修正后的数据存入 self.df_corrected
        """
        df_raw = self.df_raw.copy()
        self.min_date = df_raw["date"].min()
        if "temperature" in df_raw.columns:
            mean_t = df_raw["temperature"].mean()
            std_t = df_raw["temperature"].std() + 1e-6
            self.temp_stats = (mean_t, std_t)
        else:
            self.temp_stats = (0, 1)

        df_feat = self._build_features(df_raw)
        df_feat = self._prepare_feature_columns(df_feat)

        # 训练初步模型（使用全量数据，但排除春节日期？实际我们先用全量数据训练一个初步模型，
        # 因为春节数据本身就是异常，用全量数据训练会让模型学到春节低负荷，替换后再训练即可。
        # 这里为了简单，直接用全量数据训练初步模型）
        unique_dates = sorted(df_feat["date"].unique())
        prelim_model = self._fit_model(df_feat, unique_dates)
        if prelim_model is None:
            st.error("初步模型训练失败，无法进行春节修正")
            self.df_corrected = df_raw
            return

        # 找出春节日期（全部，因为修正针对全量数据）
        spring_mask = (df_feat["date"] >= self.spring_range[0]) & (df_feat["date"] <= self.spring_range[1])
        spring_indices = df_feat.index[spring_mask]
        if len(spring_indices) == 0:
            st.info("春节日期不在数据范围内，无需修正。")
            self.df_corrected = df_raw
            return

        # 预测春节期间负荷
        X_spring = df_feat.loc[spring_indices, self.feat_cols].values
        pred_spring = prelim_model.predict(X_spring)

        # 替换原始数据中的春节负荷
        df_corrected = df_raw.copy()
        df_corrected.loc[spring_indices, self.target_col] = pred_spring
        self.df_corrected = df_corrected
        st.success(f"已用预测值替换 {len(spring_indices)} 条春节小时负荷数据，并完成修正。")

    def evaluate_with_test_split(self):
        """
        基于修正后的全量数据 (self.df_corrected)，划分训练/测试集，训练评估模型，
        返回测试集真实值、预测值、日期等，用于展示评估指标和图表。
        """
        if self.df_corrected is None:
            st.error("请先进行春节修正。")
            return None

        df = self.df_corrected.copy()
        # 重新计算特征（min_date 不变）
        df_feat = self._build_features(df)
        df_feat = self._prepare_feature_columns(df_feat)

        unique_dates = sorted(df_feat["date"].unique())
        if self.test_days is None or self.test_days <= 0 or len(unique_dates) <= self.test_days:
            train_dates = unique_dates
            test_dates = []
        else:
            train_dates = unique_dates[:-self.test_days]
            test_dates = unique_dates[-self.test_days:]

        self.model = self._fit_model(df_feat, train_dates)
        if self.model is None:
            return None

        # 测试集预测
        if not test_dates:
            st.warning("没有测试日期。")
            return None

        df_test = df_feat[df_feat["date"].isin(test_dates)].copy()
        X_test = df_test[self.feat_cols].values
        y_true = df_test[self.target_col].values
        y_pred = self.model.predict(X_test)

        df_test["y_true"] = y_true
        df_test["y_pred"] = y_pred

        piv_true = df_test.pivot(index="date", columns="hour", values="y_true").reindex(columns=self.hours).fillna(0.0)
        piv_pred = df_test.pivot(index="date", columns="hour", values="y_pred").reindex(columns=self.hours).fillna(0.0)

        dates_out = list(piv_true.index)
        return {
            "true_mat": piv_true.values,
            "pred_mat": piv_pred.values,
            "dates": dates_out,
            "hours": self.hours
        }

    def fit_final_model(self):
        """
        使用修正后的全量数据训练最终模型，用于未来预测。
        """
        if self.df_corrected is None:
            st.error("没有修正后的数据，无法训练最终模型。")
            return

        df = self.df_corrected.copy()
        df_feat = self._build_features(df)
        df_feat = self._prepare_feature_columns(df_feat)

        unique_dates = sorted(df_feat["date"].unique())
        self.final_model = self._fit_model(df_feat, unique_dates)
        if self.final_model is not None:
            st.success("最终预测模型已基于全部历史数据训练完成（包含测试集时间段）。")

    def predict_future_curve(self, df_pred, return_long=True):
        """
        对未来气象数据进行预测（使用 final_model）
        df_pred: 包含 date, hour, temperature 的 DataFrame
        """
        if self.final_model is None:
            st.warning("最终模型未训练，请先调用 fit_final_model()。")
            return {
                "dates": pd.Index([]),
                "hours": self.hours,
                "X_load_pred": np.empty((0, len(self.hours))),
                "df_curve_pred_wide": pd.DataFrame(),
                "df_curve_pred_long": pd.DataFrame(),
            }

        df_pred = self._normalize_columns(df_pred)
        df_pred["date"] = pd.to_datetime(df_pred["date"])
        df_pred = df_pred.sort_values(["date", "hour"]).reset_index(drop=True)

        # 添加特征
        df_pred = add_manual_holiday_features(df_pred)
        df_pred["day_idx"] = (df_pred["date"] - self.min_date).dt.days
        df_pred["dow"] = df_pred["date"].dt.weekday
        df_pred["month"] = df_pred["date"].dt.month
        df_pred["day_of_year"] = df_pred["date"].dt.dayofyear

        mean_t, std_t = self.temp_stats
        if "temperature" in df_pred.columns:
            df_pred["temp_norm"] = (df_pred["temperature"] - mean_t) / std_t
        else:
            df_pred["temp_norm"] = 0.0

        # 确保所有特征列都存在
        for col in self.feat_cols:
            if col not in df_pred.columns:
                df_pred[col] = 0.0

        X_pred = df_pred[self.feat_cols].values
        y_pred = self.final_model.predict(X_pred)
        df_pred["y_pred"] = y_pred

        piv_pred = df_pred.pivot(index="date", columns="hour", values="y_pred").reindex(columns=self.hours).fillna(0.0)
        future_dates = piv_pred.index
        X_load_pred = piv_pred.values

        result = {
            "dates": future_dates,
            "hours": self.hours,
            "X_load_pred": X_load_pred
        }

        if return_long:
            df_curve_pred_wide = pd.DataFrame(X_load_pred, index=future_dates, columns=self.hours).reset_index()
            df_curve_pred_wide = df_curve_pred_wide.rename(columns={"index": "date"})
            df_curve_pred_long = df_curve_pred_wide.melt(
                id_vars="date",
                var_name="hour",
                value_name=f"{self.target_col}_pred"
            )
            result["df_curve_pred_wide"] = df_curve_pred_wide
            result["df_curve_pred_long"] = df_curve_pred_long

        return result

# =========================
# 5. 可视化函数
# =========================

def plot_test_comparison(true_mat, pred_mat, dates, hours, title_prefix="测试集"):
    n_plot = min(5, true_mat.shape[0])
    for i in range(n_plot):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(hours, true_mat[i], marker="o", label="真实负荷", color='green', alpha=0.7)
        ax.plot(hours, pred_mat[i], marker="x", linestyle="--", label="预测负荷", color='red')
        ax.set_xlabel("小时 (0-23)")
        ax.set_ylabel("负荷")
        ax.set_title(f"{title_prefix} - {pd.Timestamp(dates[i]).date()}")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        if len(hours) > 0:
            ax.set_xticks(np.arange(0, max(hours) + 1, 2))
        st.pyplot(fig)

# =========================
# 6. 主按钮逻辑
# =========================

if run_button:
    if not uploaded_power_files:
        st.error("请至少上传一个用电量 Excel 文件。")
    elif uploaded_weather is None:
        st.error("请上传气象数据文档。")
    else:
        # ---------- 读取气象数据 ----------
        with st.spinner("正在解析气象数据..."):
            try:
                df_weather_all = parse_weather_file(uploaded_weather)
            except Exception as e:
                st.error(f"气象数据解析失败：{e}")
                st.stop()
            st.success("气象数据解析成功。")
            st.write(f"气象数据日期范围：{df_weather_all['date'].min().date()} 至 {df_weather_all['date'].max().date()}")

        # ---------- 处理用电数据（汇总总负荷）----------
        with st.spinner("正在处理用电数据并汇总总负荷..."):
            total_power_hourly, (power_start_date, power_end_date), last_day_customers = process_power_files_get_total(uploaded_power_files)

            if power_start_date is None:
                st.error("用电数据为空，请检查上传文件。")
                st.stop()

            st.subheader("总负荷预览（小时级）")
            if not total_power_hourly.empty:
                st.dataframe(total_power_hourly.head(20))
            else:
                st.write("无数据")

        # ---------- 划分天气数据 ----------
        with st.spinner("根据电量数据日期范围划分天气数据..."):
            train_weather, future_weather = split_weather_by_power_dates(
                df_weather_all, power_start_date, power_end_date
            )
            st.write(f"训练气象数据（对应电量日期范围）：{train_weather['date'].min().date()} 至 {train_weather['date'].max().date()}，共 {len(train_weather)} 条")
            st.write(f"预测气象数据：{future_weather['date'].min().date()} 至 {future_weather['date'].max().date()}，共 {len(future_weather)} 条")

            if future_weather.empty:
                st.warning("没有未来气象数据，无法进行未来预测。")

        # ---------- 构造训练数据框（全量历史数据）----------
        st.subheader("构造训练数据集（全量历史）")
        df_train_total = build_train_df(train_weather, total_power_hourly, col_name="total_load")

        with st.expander("训练数据示例（总负荷）"):
            st.dataframe(df_train_total.head(20))

        # ---------- 初始化模型并进行春节修正（基于全量数据）----------
        st.subheader("模型训练流程")
        model = CorrectedXGBoost(target_col='total_load', test_days=test_days)
        model.prepare_data(df_train_total)

        with st.spinner("正在进行春节负荷修正（基于全量历史数据）..."):
            model.correct_spring_festival()
        st.success("春节修正完成。")

        # ---------- 评估：基于修正后数据划分训练/测试集，计算指标 ----------
        st.subheader("测试集性能评估（基于最近 N 天测试集）")
        eval_result = model.evaluate_with_test_split()
        if eval_result is not None:
            y_true = eval_result["true_mat"]
            y_pred = eval_result["pred_mat"]
            test_dates_list = eval_result["dates"]
            hours = eval_result["hours"]
            mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
            rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
            mape_vals = np.abs(y_true.flatten() - y_pred.flatten()) / np.maximum(np.abs(y_true.flatten()), 1e-6)
            mape = np.mean(mape_vals)
            st.write(f"**测试集整体指标：** MAE = {mae:.4f}, RMSE = {rmse:.4f}, MAPE = {mape:.4%}")
            st.write(f"测试日期：{[pd.Timestamp(d).date() for d in test_dates_list]}")
            plot_test_comparison(y_true, y_pred, test_dates_list, hours, title_prefix="测试集总负荷")
        else:
            st.warning("测试集评估未产生结果。")

        # ---------- 训练最终模型（使用全量修正数据）----------
        with st.spinner("正在使用全部历史数据（含测试集）训练最终预测模型..."):
            model.fit_final_model()

        # ---------- 未来预测 ----------
        st.subheader("未来负荷预测")
        if future_weather.empty:
            st.warning("无未来气象数据，无法进行未来预测。")
        else:
            future_df = future_weather[["date", "hour", "temperature"]].copy().sort_values(["date", "hour"])
            future_result = model.predict_future_curve(future_df, return_long=True)

            if future_result["X_load_pred"].size > 0:
                df_total_wide = future_result["df_curve_pred_wide"]
                st.success("未来总负荷预测完成。")
                st.dataframe(df_total_wide.head())

                with st.expander("未来总负荷预测曲线（前3天）"):
                    n_plot = min(3, df_total_wide.shape[0])
                    dates_future = df_total_wide["date"].values
                    load_mat = df_total_wide.iloc[:, 1:].values
                    for i in range(n_plot):
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.plot(hours, load_mat[i], marker="o", label="总负荷预测")
                        ax.set_title(str(pd.Timestamp(dates_future[i]).date()))
                        ax.set_xlabel("小时")
                        ax.set_ylabel("负荷")
                        ax.grid(True, linestyle=":", alpha=0.5)
                        st.pyplot(fig)

                # 导出 Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    workbook = writer.book
                    header_fmt = workbook.add_format({
                        "bold": True,
                        "bg_color": "#4F81BD",
                        "font_color": "white",
                        "border": 1
                    })
                    num_fmt = workbook.add_format({"num_format": "0.0000", "border": 1})

                    df_total_wide.to_excel(writer, sheet_name="未来总负荷预测", index=False)
                    ws = writer.sheets["未来总负荷预测"]
                    for col_num, value in enumerate(df_total_wide.columns.values):
                        ws.write(0, col_num, value, header_fmt)
                        ws.set_column(col_num, col_num, 14, num_fmt)

                output.seek(0)
                st.download_button(
                    label="下载未来预测结果 Excel",
                    data=output,
                    file_name="future_load_forecast_corrected.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("未来预测结果为空，请检查气象数据是否覆盖未来日期。")

        st.success("所有处理流程完成！")