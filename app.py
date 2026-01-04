import streamlit as st
from src.pipeline import main_predict
from src.utils import plot_map
import torch
import geopandas as gpd
import sys, os
from src import config
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import calendar
import json
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

st.set_page_config(
    page_title="Mekong Delta Rainfall Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_resources():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vn_adm1 = gpd.read_file(config.GADM_PATH)
    mekong_provinces = [
        "LongAn", "TiềnGiang", "BếnTre", "TràVinh", "VĩnhLong",
        "ĐồngTháp", "AnGiang", "CầnThơ", "HậuGiang",
        "SócTrăng", "BạcLiêu", "CàMau", "KiênGiang"
    ]
    dbscl = vn_adm1[vn_adm1["NAME_1"].isin(mekong_provinces)]
    return vn_adm1, dbscl, device, mekong_provinces

@st.cache_data
def load_json_data(stage1_model, stage2_model):
    json_path = Path(f"output/monthly_rainfall_{stage1_model}_{stage2_model}.json")
    
    if not json_path.exists():
        st.error(f"File not found: {json_path}")
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    province_mapping = {
        'An Giang': 'An Giang',
        'Bac Lieu': 'Bạc Liêu',
        'Ben Tre': 'Bến Tre',
        'Ca Mau': 'Cà Mau',
        'Can Tho city': 'Cần Thơ',
        'Dong Thap': 'Đồng Tháp',
        'Hau Giang': 'Hậu Giang',
        'Kien Giang': 'Kiên Giang',
        'Long An': 'Long An',
        'Soc Trang': 'Sóc Trăng',
        'Tien Giang': 'Tiền Giang',
        'Tra Vinh': 'Trà Vinh',
        'Vinh Long': 'Vĩnh Long'
    }
    
    df['province_display'] = df['province'].map(province_mapping)
    
    return df

@st.cache_data
def get_available_models():
    output_path = Path("output")
    if not output_path.exists():
        return [], []
    
    files = list(output_path.glob("monthly_rainfall_*.json"))
    
    stage1_models = set()
    stage2_models = set()
    
    for file in files:
        parts = file.stem.replace("monthly_rainfall_", "").split("_")
        if len(parts) >= 2:
            stage1_models.add(parts[0])
            stage2_models.add(parts[1])
    
    return sorted(list(stage1_models)), sorted(list(stage2_models))

st.title("Hệ Thống Dự Báo & Phân Tích Lượng Mưa ĐBSCL")

stage1_available, stage2_available = get_available_models()

if not stage1_available or not stage2_available:
    st.error("Không tìm thấy file dữ liệu JSON trong thư mục output/")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "Dự Đoán Thời Gian Thực", 
    "Thống Kê",
    "Bản Đồ Nhiệt",
    "So Sánh Xu Hướng"
])

with tab1:
    st.header("Dự Đoán Lượng Mưa Theo Ngày")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        day = st.number_input("Ngày", 1, 31, 1)
    with col2:
        month = st.number_input("Tháng", 1, 12, 1)
    with col3:
        year = st.number_input("Năm", min_value=2025, max_value=2027, value=2025)
    
    col4, col5 = st.columns(2)
    with col4:
        model_stage1 = st.selectbox(
            "Stage 1 Model",
            stage1_available,
            key="predict_stage1"
        )
    with col5:
        model_stage2 = st.selectbox(
            "Stage 2 Model",
            stage2_available,
            key="predict_stage2"
        )
    
    if st.button("Bắt Đầu Dự Đoán", type="primary", use_container_width=True):
        status = st.empty()
        status.info("Đang tải tài nguyên và mô hình...")
        vn_adm1, dbscl, device, _ = load_resources()
        
        status.info(f"Đang chạy dự đoán cho ngày {day}/{month}/{year}...")
        pred = main_predict(model_stage1, model_stage2, day, month, year)
        
        status.success("Dự đoán hoàn tất!")
        
        st.subheader(f"Bản Đồ Lượng Mưa Dự Đoán ({day}/{month}/{year})")
        
        fig = plot_map(
            pred,
            title=f"Predicted Rainfall ({day}/{month}/{year})",
            vn_adm1=vn_adm1,
            dbscl=dbscl
        )
        st.pyplot(fig, use_container_width=True)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Lượng Mưa TB", f"{pred.mean().item():.1f} mm")
        with col_b:
            st.metric("Lượng Mưa Lớn Nhất", f"{pred.max().item():.1f} mm")
        with col_c:
            st.metric("Lượng Mưa Nhỏ Nhất", f"{pred.min().item():.1f} mm")
        
        del pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

with tab2:
    st.header("Thống Kê Lượng Mưa Trung Bình")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        stats_stage1 = st.selectbox(
            "Stage 1 Model",
            stage1_available,
            key="stats_stage1"
        )
    with col_s2:
        stats_stage2 = st.selectbox(
            "Stage 2 Model",
            stage2_available,
            key="stats_stage2"
        )
    
    df_data = load_json_data(stats_stage1, stats_stage2)
    
    if df_data is None:
        st.error("Không thể tải dữ liệu. Vui lòng kiểm tra lại.")
    else:
        year_select = st.radio("Chọn Năm", [2025, 2026, 2027], horizontal=True)
        
        df_year = df_data[df_data['year'] == year_select].copy()
    
        if len(df_year) == 0:
            st.warning(f"Không có dữ liệu cho năm {year_select}")
        else:
            pivot_year = df_year.pivot(
                index='province_display', 
                columns='month', 
                values='avg_monthly_rainfall'
            )
            
            colorscale = 'Blues' if year_select == 2025 else ('Greens' if year_select == 2026 else 'Purples')
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=pivot_year.values,
                x=[calendar.month_abbr[i] for i in range(1, 13)],
                y=pivot_year.index,
                colorscale=colorscale,
                text=np.round(pivot_year.values, 1),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="mm")
            ))
            fig_heatmap.update_layout(
                title=f"Lượng Mưa Trung Bình Các Tỉnh ĐBSCL - {year_select}",
                xaxis_title="Tháng",
                yaxis_title="Tỉnh/Thành",
                height=600
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            monthly_avg = df_year.groupby('month')['avg_monthly_rainfall'].mean().reset_index()
            fig_line = px.line(
                monthly_avg, 
                x='month', 
                y='avg_monthly_rainfall',
                markers=True,
                title=f"Xu Hướng Lượng Mưa Trung Bình Theo Tháng - {year_select}"
            )
            fig_line.update_layout(
                xaxis_title="Tháng",
                yaxis_title="Lượng Mưa (mm)",
                height=400
            )
            st.plotly_chart(fig_line, use_container_width=True)
            
            province_total = df_year.groupby('province_display')['avg_monthly_rainfall'].sum().reset_index()
            province_total = province_total.sort_values('avg_monthly_rainfall', ascending=False)
            
            fig_bar = px.bar(
                province_total,
                x='province_display',
                y='avg_monthly_rainfall',
                title=f"Tổng Lượng Mưa Các Tỉnh Năm {year_select}",
                color='avg_monthly_rainfall',
                color_continuous_scale=colorscale
            )
            fig_bar.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    st.header("Bản Đồ Nhiệt Lượng Mưa")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        stats_stage1_map = st.selectbox(
            "Stage 1 Model",
            stage1_available,
            key="stats_stage1_map"
        )
    with col_s2:
        stats_stage2_map = st.selectbox(
            "Stage 2 Model",
            stage2_available,
            key="stats_stage2_map"
        )
    
    df_data = load_json_data(stats_stage1_map, stats_stage2_map)
    
    if df_data is None:
        st.error("Không thể tải dữ liệu. Vui lòng kiểm tra lại.")
    else:        
        col_y, col_m = st.columns(2)
        with col_y:
            year_select = st.radio("Chọn Năm", [2025, 2026, 2027], horizontal=True, key="map_year")
        with col_m:
            month_select = st.slider("Chọn Tháng", 1, 12, 6)
        
        df_selected = df_data[(df_data['year'] == year_select) & (df_data['month'] == month_select)]
        
        if len(df_selected) == 0:
            st.warning(f"Không có dữ liệu cho tháng {month_select}/{year_select}")
        else:
            fig_map = px.bar(
                df_selected,
                x='province_display',
                y='avg_monthly_rainfall',
                title=f"Lượng Mưa Tháng {month_select}/{year_select}",
                color='avg_monthly_rainfall',
                color_continuous_scale='RdYlBu_r',
                height=500
            )
            fig_map.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_map, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Lượng Mưa TB", f"{df_selected['avg_monthly_rainfall'].mean():.2f} mm")
            with col2:
                st.metric("Max", f"{df_selected['avg_monthly_rainfall'].max():.2f} mm")
            with col3:
                st.metric("Min", f"{df_selected['avg_monthly_rainfall'].min():.2f} mm")

with tab4:
    st.header("So Sánh Xu Hướng 2025 vs 2026")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        stats_stage1_compare = st.selectbox(
            "Stage 1 Model",
            stage1_available,
            key="stats_stage1_compare"
        )
    with col_s2:
        stats_stage2_compare = st.selectbox(
            "Stage 2 Model",
            stage2_available,
            key="stats_stage2_compare"
        )
    
    df_data = load_json_data(stats_stage1_compare, stats_stage2_compare)
    
    if df_data is None:
        st.error("Không thể tải dữ liệu. Vui lòng kiểm tra lại.")
    else:
        df_2025 = df_data[df_data['year'] == 2025].copy()
        df_2026 = df_data[df_data['year'] == 2026].copy()
        
        if len(df_2025) == 0 or len(df_2026) == 0:
            st.warning("Không đủ dữ liệu để so sánh")
        else:
            avg_2025 = df_2025.groupby('month')['avg_monthly_rainfall'].mean().reset_index()
            avg_2025['year'] = 2025
            avg_2026 = df_2026.groupby('month')['avg_monthly_rainfall'].mean().reset_index()
            avg_2026['year'] = 2026
            
            df_compare = pd.concat([avg_2025, avg_2026])
            
            fig_compare = px.line(
                df_compare,
                x='month',
                y='avg_monthly_rainfall',
                color='year',
                markers=True,
                title="So Sánh Xu Hướng Lượng Mưa 2025 vs 2026",
                height=500
            )
            fig_compare.update_layout(
                xaxis_title="Tháng",
                yaxis_title="Lượng Mưa Trung Bình (mm)"
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            
            province_compare = st.selectbox(
                "Chọn Tỉnh Để So Sánh Chi Tiết",
                sorted(df_2025['province_display'].unique())
            )
            
            prov_2025 = df_2025[df_2025['province_display'] == province_compare].copy()
            prov_2026 = df_2026[df_2026['province_display'] == province_compare].copy()
            
            prov_2025['year'] = 2025
            prov_2026['year'] = 2026
            prov_compare_df = pd.concat([prov_2025, prov_2026])
            
            fig_prov = px.bar(
                prov_compare_df,
                x='month',
                y='avg_monthly_rainfall',
                color='year',
                barmode='group',
                title=f"So Sánh Lượng Mưa {province_compare} (2025 vs 2026)",
                height=400
            )
            st.plotly_chart(fig_prov, use_container_width=True)
            
            if len(prov_2025) > 0 and len(prov_2026) > 0:
                merged = pd.merge(
                    prov_2025[['month', 'avg_monthly_rainfall']], 
                    prov_2026[['month', 'avg_monthly_rainfall']], 
                    on='month', 
                    suffixes=('_2025', '_2026')
                )
                merged['difference'] = merged['avg_monthly_rainfall_2026'] - merged['avg_monthly_rainfall_2025']
                
                fig_diff = px.bar(
                    merged,
                    x='month',
                    y='difference',
                    title=f"Chênh Lệch Lượng Mưa {province_compare} (2026 - 2025)",
                    color='difference',
                    color_continuous_scale='RdBu',
                    height=400
                )
                fig_diff.update_layout(xaxis_title="Tháng", yaxis_title="Chênh lệch (mm)")
                st.plotly_chart(fig_diff, use_container_width=True)