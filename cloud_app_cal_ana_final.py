# cloud_analysis_app_final.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import io

# --- 1. ส่วนของการตั้งค่าและฟังก์ชัน (Setup & Functions) ---
try:
    font_path = 'THSarabunNew.ttf'
    font_prop = fm.FontProperties(fname=font_path)
except Exception:
    st.warning(f"ไม่พบไฟล์ฟอนต์ '{font_path}' กรุณาวางไฟล์ในโฟลเดอร์เดียวกับโค้ด")
    font_prop = fm.FontProperties() # Fallback to default font

@st.cache_data
def load_and_process_files(uploaded_files, columns_to_keep, time_col):
    """ฟังก์ชันสำหรับโหลดไฟล์ทั้งหมด, ทำความสะอาด, และประมวลผลข้อมูล"""
    if not uploaded_files: return None
    
    list_of_dfs = [pd.read_csv(file, sep='\t', engine='python', skiprows=1, on_bad_lines='skip') for file in uploaded_files]
    df = pd.concat(list_of_dfs, ignore_index=True)
    
    df.columns = df.columns.str.strip()
    
    missing_cols = [col for col in columns_to_keep if col not in df.columns]
    if missing_cols:
        st.error(f"ไฟล์ที่อัปโหลดขาดคอลัมน์ที่ต้องการ: {', '.join(missing_cols)}")
        st.info(f"คอลัมน์ที่พบในไฟล์: {', '.join(df.columns)}")
        return None

    processed_df = df[columns_to_keep].copy()
    processed_df[time_col] = pd.to_datetime(processed_df[time_col].astype(str).str.strip(), errors='coerce')
    processed_df.dropna(subset=[time_col], inplace=True)
    processed_df.set_index(time_col, inplace=True)
    processed_df.index = processed_df.index.floor('s').tz_localize(None)
    
    return processed_df

# --- 2. ส่วนหน้าตาของแอปพลิเคชัน (Streamlit UI) ---

st.set_page_config(layout="wide")
st.title("โปรแกรมจัดการข้อมูลความสูงฐานเมฆ")

# === ส่วนที่ 1: การนำเข้าข้อมูล ===
st.header("1. นำเข้าข้อมูล")
st.write("อัปโหลดชุดข้อมูลความสูงฐานเมฆจากเครื่องตรวจวัดและข้อมูลอากาศเพื่อเริ่มการวิเคราะห์")

col1, col2 = st.columns(2)
with col1:
    instrument_files = st.file_uploader("ไฟล์ข้อมูลความสูงฐานเมฆ", type=['his', 'txt'], accept_multiple_files=True, key="inst_uploader", help="ต้องมีคอลัมน์ BASE1 (FT), BASE2 (FT), AMOUNT1, AMOUNT2")
with col2:
    weather_files = st.file_uploader("ไฟล์ข้อมูลอากาศ (T, Td)", type=['his', 'txt'], accept_multiple_files=True, key="weather_uploader", help="ต้องมีคอลัมน์ CREATEDATE, TEMP (°C), DEWPOINT (°C)")

if st.button("เริ่มการวิเคราะห์", type="primary", use_container_width=True):
    if instrument_files and weather_files:
        with st.spinner('กำลังประมวลผลข้อมูล...'):
            inst_cols_to_keep = ['CREATEDATE', 'BASE1 (FT)', 'BASE2 (FT)', 'AMOUNT1', 'AMOUNT2']
            df_inst = load_and_process_files(instrument_files, inst_cols_to_keep, 'CREATEDATE')
            df_weather = load_and_process_files(weather_files, ['CREATEDATE', 'TEMP (°C)', 'DEWPOINT (°C)'], 'CREATEDATE')

            if df_inst is not None and df_weather is not None:
                # แปลงข้อมูลเป็นตัวเลข
                df_inst['BASE1 (FT)'] = pd.to_numeric(df_inst['BASE1 (FT)'], errors='coerce')
                df_inst['BASE2 (FT)'] = pd.to_numeric(df_inst['BASE2 (FT)'], errors='coerce')
                df_inst.dropna(subset=['BASE1 (FT)', 'BASE2 (FT)'], how='all', inplace=True)

                df_weather['TEMP (°C)'] = pd.to_numeric(df_weather['TEMP (°C)'], errors='coerce')
                df_weather['DEWPOINT (°C)'] = pd.to_numeric(df_weather['DEWPOINT (°C)'], errors='coerce')
                df_weather.dropna(subset=['TEMP (°C)', 'DEWPOINT (°C)'], inplace=True)

                if df_inst.empty:
                    st.error("ไม่พบข้อมูลความสูงฐานเมฆที่ถูกต้องในไฟล์ที่อัปโหลด")
                    st.stop()

                st.markdown("---")
                
                # === ส่วนที่ 2: การสำรวจข้อมูลที่ตรวจวัดได้ ===
                st.header("2. ภาพรวมข้อมูลจากเครื่องตรวจวัด")
                
                st.subheader("2.1 ข้อมูลสถิติพื้นฐาน")
                st.dataframe(df_inst[['BASE1 (FT)', 'BASE2 (FT)']].describe())
                with st.expander("ดูคำอธิบายค่าทางสถิติ"):
                    descriptions = { "ค่าสถิติ": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"], "คำอธิบาย": ["จำนวนข้อมูล", "ค่าเฉลี่ย", "ค่าเบี่ยงเบนมาตรฐาน", "ค่าต่ำสุด", "ควอร์ไทล์ที่ 1", "ค่ากลาง (Median)", "ควอร์ไทล์ที่ 3", "ค่าสูงสุด"] }
                    st.dataframe(pd.DataFrame(descriptions).set_index("ค่าสถิติ"))

                st.subheader("2.2 การกระจายตัวของข้อมูลความสูงฐานเมฆ")
                st.markdown("#### การกระจายตัวของ Base 1")
                hist1_col1, hist1_col2 = st.columns([0.6, 0.4])
                with hist1_col1:
                    fig_hist1 = px.histogram(df_inst.dropna(subset=['BASE1 (FT)']), x="BASE1 (FT)", title="Histogram ของ BASE1 (FT)")
                    st.plotly_chart(fig_hist1, use_container_width=True)
                with hist1_col2:
                    st.write("ข้อมูลความถี่ (Base 1):")
                    counts1, bin_edges1 = np.histogram(df_inst['BASE1 (FT)'].dropna(), bins=20)
                    bin_labels1 = [f"{int(bin_edges1[i])}-{int(bin_edges1[i+1])}" for i in range(len(counts1))]
                    hist_df1 = pd.DataFrame({"ช่วงความสูง (ฟุต)": bin_labels1, "ความถี่ (ครั้ง)": counts1})
                    st.dataframe(hist_df1, use_container_width=True)
                    csv1 = hist_df1.to_csv(index=False).encode('utf-8')
                    st.download_button("ดาวน์โหลดข้อมูลความถี่ (Base 1)", csv1, "histogram_base1_data.csv", "text/csv", use_container_width=True, key="dl_base1")
                
                st.markdown("---")
                st.markdown("#### การกระจายตัวของ Base 2")
                hist2_col1, hist2_col2 = st.columns([0.6, 0.4])
                with hist2_col1:
                    fig_hist2 = px.histogram(df_inst.dropna(subset=['BASE2 (FT)']), x="BASE2 (FT)", color_discrete_sequence=['indianred'], title="Histogram ของ BASE2 (FT)")
                    st.plotly_chart(fig_hist2, use_container_width=True)
                with hist2_col2:
                    st.write("ข้อมูลความถี่ (Base 2):")
                    counts2, bin_edges2 = np.histogram(df_inst['BASE2 (FT)'].dropna(), bins=20)
                    bin_labels2 = [f"{int(bin_edges2[i])}-{int(bin_edges2[i+1])}" for i in range(len(counts2))]
                    hist_df2 = pd.DataFrame({"ช่วงความสูง (ฟุต)": bin_labels2, "ความถี่ (ครั้ง)": counts2})
                    st.dataframe(hist_df2, use_container_width=True)
                    csv2 = hist_df2.to_csv(index=False).encode('utf-8')
                    st.download_button("ดาวน์โหลดข้อมูลความถี่ (Base 2)", csv2, "histogram_base2_data.csv", "text/csv", use_container_width=True, key="dl_base2")

                st.markdown("---")
                
                st.subheader("2.3 การกระจายตัวของปริมาณเมฆ")
                def analyze_amount(df, amount_col_name):
                    valid_categories = ['FEW', 'SCT', 'BKN', 'OVC']
                    cleaned_series = df[amount_col_name].dropna()
                    filtered_series = cleaned_series[cleaned_series.isin(valid_categories)]
                    counts = filtered_series.value_counts().reset_index()
                    counts.columns = ['ประเภทปริมาณเมฆ', 'ความถี่ (ครั้ง)']
                    counts.index = np.arange(1, len(counts) + 1)
                    return counts

                st.markdown("#### ปริมาณเมฆของชั้นที่ 1 (AMOUNT1)")
                amount1_counts = analyze_amount(df_inst, 'AMOUNT1')
                fig1 = px.bar(amount1_counts, x='ประเภทปริมาณเมฆ', y='ความถี่ (ครั้ง)', title='ความถี่ของ AMOUNT1')
                st.plotly_chart(fig1, use_container_width=True)
                st.dataframe(amount1_counts, use_container_width=True)
                csv1 = amount1_counts.to_csv().encode('utf-8')
                st.download_button("ดาวน์โหลดข้อมูล (AMOUNT1)", csv1, "amount1_data.csv", "text/csv", use_container_width=True, key="dl_amount1")
                
                st.markdown("---")
                st.markdown("#### ปริมาณเมฆของชั้นที่ 2 (AMOUNT2)")
                amount2_counts = analyze_amount(df_inst, 'AMOUNT2')
                fig2 = px.bar(amount2_counts, x='ประเภทปริมาณเมฆ', y='ความถี่ (ครั้ง)', title='ความถี่ของ AMOUNT2', color_discrete_sequence=['indianred'])
                st.plotly_chart(fig2, use_container_width=True)
                st.dataframe(amount2_counts, use_container_width=True)
                csv2 = amount2_counts.to_csv().encode('utf-8')
                st.download_button("ดาวน์โหลดข้อมูล (AMOUNT2)", csv2, "amount2_data.csv", "text/csv", use_container_width=True, key="dl_amount2")

                st.markdown("---")
                st.subheader("2.4 รูปแบบความสูงฐานเมฆรายชั่วโมง (Diurnal Pattern)")
                
                df_inst_eda = df_inst.copy()
                df_inst_eda['hour'] = df_inst_eda.index.hour
                fig_diurnal = px.box(
                    df_inst_eda.dropna(subset=['BASE1 (FT)']),
                    x='hour',
                    y='BASE1 (FT)',
                    title='การกระจายตัวของความสูงฐานเมฆ Base 1 ในแต่ละชั่วโมงของวัน',
                    labels={'hour': 'ชั่วโมงของวัน', 'BASE1 (FT)': 'ความสูงฐานเมฆ (ฟุต)'}
                )
                st.plotly_chart(fig_diurnal, use_container_width=True)
                
                st.markdown("---")
                st.subheader("2.5 รูปแบบความสูงฐานเมฆรายเดือน (Seasonal Pattern)")
                
                df_inst_eda['month'] = df_inst_eda.index.month
                fig_seasonal = px.box(
                    df_inst_eda.dropna(subset=['BASE1 (FT)']),
                    x='month',
                    y='BASE1 (FT)',
                    title='การกระจายตัวของความสูงฐานเมฆ Base 1 ในแต่ละเดือน',
                    labels={'month': 'เดือน', 'BASE1 (FT)': 'ความสูงฐานเมฆ (ฟุต)'}
                )
                st.plotly_chart(fig_seasonal, use_container_width=True)
                
                st.markdown("---")
                
                # === ส่วนที่ 3: การเปรียบเทียบด้วยสูตรสากล ===
                st.header("3. การเปรียบเทียบด้วยสูตรสากล (เทียบกับ Base 1)")
                
                st.subheader("สูตรที่ใช้ในการคำนวณ")
                st.latex(r'''\text{ความสูงฐานเมฆ (ฟุต)} \approx 410 \times (T_{°C} - Td_{°C})''')
                st.latex(r'''\text{ความคลาดเคลื่อน (\%)} = \frac{(\text{ค่าที่คำนวณได้} - \text{ค่าที่วัดได้จริง})}{\text{ค่าที่วัดได้จริง}} \times 100''')
                st.markdown("---")

                df_inst_hourly = df_inst[(df_inst.index.minute == 0) & (df_inst.index.second == 0)]
                df_weather_hourly = df_weather[(df_weather.index.minute == 0) & (df_weather.index.second == 0)]
                final_df = pd.merge(df_inst_hourly, df_weather_hourly, left_index=True, right_index=True, how='inner')

                if final_df.empty:
                    st.error("ไม่พบข้อมูล ณ เวลาต้นชั่วโมงที่ตรงกันในไฟล์ทั้งสองชุด")
                else:
                    st.success(f"พบข้อมูลรายชั่วโมงที่ตรงกัน {len(final_df):,} จุด! เริ่มการเปรียบเทียบ...")
                    
                    final_df['cbh_universal_ft'] = (125 * (final_df['TEMP (°C)'] - final_df['DEWPOINT (°C)'])) * 3.28084
                    # ใช้ np.abs() เพื่อให้เป็นค่า tuyệt đối ก่อนหาค่าเฉลี่ย
                    final_df['error_percent'] = np.abs(((final_df['cbh_universal_ft'] - final_df['BASE1 (FT)']) / final_df['BASE1 (FT)'])) * 100
                    avg_error_percent = final_df['error_percent'].mean()
                    
                    st.metric("ค่าเฉลี่ยความคลาดเคลื่อน (Average Absolute Error)", f"{avg_error_percent:.2f} %")
                    
                    fig_compare, ax_compare = plt.subplots(figsize=(8, 8))
                    sns.regplot(x='BASE1 (FT)', y='cbh_universal_ft', data=final_df, ax=ax_compare, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
                    max_val = final_df[['BASE1 (FT)', 'cbh_universal_ft']].max().max()
                    ax_compare.plot([0, max_val], [0, max_val], color='gray', linestyle='--')
                    ax_compare.set_title('เปรียบเทียบค่าจริง vs. ค่าจากสูตรสากล', fontproperties=font_prop, fontsize=16)
                    ax_compare.set_xlabel('ความสูงจากเครื่องตรวจ (ฟุต)', fontproperties=font_prop, fontsize=12)
                    ax_compare.set_ylabel('ความสูงจากการคำนวณ (ฟุต)', fontproperties=font_prop, fontsize=12)
                    ax_compare.grid(True)
                    ax_compare.axis('equal')
                    st.pyplot(fig_compare)
                    
                    st.markdown("---")
                    st.header("4. ดาวน์โหลดผลลัพธ์การเปรียบเทียบ")
                    st.dataframe(final_df.head())
                    csv_data = final_df.to_csv(index=True).encode('utf-8')
                    st.download_button("ดาวน์โหลดข้อมูลผลลัพธ์ทั้งหมดเป็น CSV", csv_data, 'comparison_results.csv', 'text/csv', use_container_width=True)

    else:
        st.info("กรุณาอัปโหลดไฟล์ข้อมูลทั้ง 2 ชุดเพื่อเริ่มการวิเคราะห์")
