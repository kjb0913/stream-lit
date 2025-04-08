import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import plotly.express as px
from streamlit_modal import Modal
from st_aggrid import AgGrid,GridOptionsBuilder

st.set_page_config(page_title="AS 1170.4:2007", page_icon="🏠", layout="wide")

DEFAULTS = {
    "stld_data": {},
    "unit_data": {},
    "story_df": pd.DataFrame(),
    "structure_height": 0.0,
    "api_connected": False,
    "calculated": False,
    "edc": "II",
    "t1_x": 1.0,
    "t1_y": 1.0,
    "period_calculated": False,
}


for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

def get_session_param(param):
    """Get value from session state with error tracking."""
    return st.session_state.get(param)

def set_session_param(param, value):
    """Set value in session state with logging comment."""
    # Tracking session state update: param -> value
    st.session_state[param] = value

def calc_fundamental_period(hn, formula):
    if formula == "T= 0.1375hn^0.75":
        return round(0.1375 * (hn ** 0.75), 2)
    elif formula == "T= 0.09375hn^0.75":
        return round(0.09375 * (hn ** 0.75), 2)
    elif formula == "T= 0.075hn^0.75":
        return round(0.075 * (hn ** 0.75), 2)
    elif formula == "T= 0.0625hn^0.75":
        return round(0.0625 * (hn ** 0.75), 2)
    else:
        raise ValueError("Unknown formula selected.")

    
def calc_ch_t(t1, soil_class):
    """Calculate Ch(T) based on soil class and fundamental period."""
    if soil_class == "Ae":
        return 0.8 + 15.5 * t1 if t1 <= 0.1 else min(0.704 / t1, 2.35) if t1 <= 1.5 else 1.056 / (t1 ** 2)
    elif soil_class == "Be":
        return 1 + 19.4 * t1 if t1 <= 0.1 else min(0.88 / t1, 2.94) if t1 <= 1.5 else 1.32 / (t1 ** 2)
    elif soil_class == "Ce":
        return 1.3 + 23.8 * t1 if t1 <= 0.1 else min(1.25 / t1, 3.68) if t1 <= 1.5 else 1.874 / (t1 ** 2)
    elif soil_class == "De":
        return 1.1 + 25.8 * t1 if t1 <= 0.1 else min(1.98 / t1, 3.68) if t1 <= 1.5 else 1.874 / (t1 ** 2)
    else:
        return 1.1 + 25.8 * t1 if t1 <= 0.1 else min(3.08 / t1, 3.68) if t1 <= 1.5 else 4.62 / (t1 ** 2)


def calc_seismic_coefficient(ch_t, mu_sp, kp, Z, t1):
    return kp * Z * t1 * ch_t / mu_sp

def calc_earthquake_design_category(importance_level, soil_class, kp, Z, hn):
    kZ = kp * Z  # kZ 값 계산
    
    categories = {
        "2": {
            "Ee": [(0.08, 12, "I"), (0.08, 50, "II"), (0.08, float("inf"), "III"),
                    (0.08, 50, "II"), (0.08, float("inf"), "III"), (float("inf"), 25, "II"), (float("inf"), float("inf"), "III")],
            "De": [(0.08, 12, "I"), (0.08, 50, "II"), (0.08, float("inf"), "III"),
                    (0.08, 50, "II"), (0.08, float("inf"), "III"), (float("inf"), 25, "II"), (float("inf"), float("inf"), "III")],
            "Ce": [(0.08, 12, "I"), (0.08, 50, "II"), (0.08, float("inf"), "III"),
                    (0.12, 50, "II"), (0.12, float("inf"), "III"), (float("inf"), 25, "II"), (float("inf"), float("inf"), "III")],
            "Be": [(0.11, 12, "I"), (0.11, 50, "II"), (0.11, float("inf"), "III"),
                    (0.17, 50, "II"), (0.17, float("inf"), "III"), (float("inf"), 25, "II"), (float("inf"), float("inf"), "III")],
            "Ae": [(0.14, 12, "I"), (0.14, 50, "II"), (0.14, float("inf"), "III"),
                    (0.21, 50, "II"), (0.21, float("inf"), "III"), (float("inf"), 25, "II"), (float("inf"), float("inf"), "III")]
        },
        "3": {
            "Ee": [(0.08, 50, "II"), (0.08, float("inf"), "III"), (float("inf"), 25, "II"), (float("inf"), float("inf"), "III")],
            "De": [(0.08, 50, "II"), (0.08, float("inf"), "III"), (float("inf"), 25, "II"), (float("inf"), float("inf"), "III")],
            "Ce": [(0.12, 50, "II"), (0.12, float("inf"), "III"), (float("inf"), 25, "II"), (float("inf"), float("inf"), "III")],
            "Be": [(0.17, 50, "II"), (0.17, float("inf"), "III"), (float("inf"), 25, "II"), (float("inf"), float("inf"), "III")],
            "Ae": [(0.21, 50, "II"), (0.21, float("inf"), "III"), (float("inf"), 25, "II"), (float("inf"), float("inf"), "III")]
        },
        "4": {
            "default": [(float("inf"), 12, "II"), (float("inf"), float("inf"), "III")]
        }
    }
    
    if importance_level == "4":
        for max_kz, max_hn, category in categories[importance_level]["default"]:
            if hn <= max_hn:
                return category
        return "III"
    
    if importance_level not in categories or soil_class not in categories[importance_level]:
        return "Invalid input"
    
    for max_kz, max_hn, category in categories[importance_level][soil_class]:
        if kZ <= max_kz and hn <= max_hn:
            return category
    
    return "III"

def def_k_value(t1):
    if t1 <= 0.5:
        return 1
    elif t1 >= 2.5:
        return 2
    else:
        return 1 + (t1 - 0.5) * (2 - 1) / (2.5 - 0.5)


def calculate_seismic_loads(df, t1, mu_sp, direction, soil_class, kp, Z):
    ch_t = calc_ch_t(t1, soil_class)
    cd = calc_seismic_coefficient(ch_t, mu_sp, kp, Z, t1)
    k = def_k_value(t1)
    df= df.copy() # 원본 데이터 보호
    df[f"Wi*Hi^k for {direction}"] = df["WEIGHT"] * df["ELEV"] ** k
    df[f"Story Force for {direction}"] = df[f"Wi*Hi^k for {direction}"] * df["WEIGHT"].sum() / df[f"Wi*Hi^k for {direction}"].sum() * cd
    df[f"Story Shear for {direction}"] = df[f"Story Force for {direction}"].cumsum().shift(1).fillna(0)
    delta_h = df["ELEV"].shift(1) - df["ELEV"]
    df[f"Overturning Moment for {direction}"] = 0.0
    df.loc[1:, f"Overturning Moment for {direction}"] = df.loc[1:, f"Story Shear for {direction}"] * delta_h[1:]
    return df


def plot_graph(df, column, direction):
    formatted_text = df[column].apply(lambda x: f"{x:.2f}")
    fig = px.bar(
        df,
        x=column,
        y="STORY",
        orientation="h",
        # title=f"{column} ({direction}-Direction)",       
        labels={column: column, "STORY": "Story"},
        height=700,
        color_discrete_sequence=["#3498db"]  # 하늘색
    )
    # 막대 안쪽에 값 표시
    fig.update_traces(
        text=formatted_text,
        textposition='inside',
        textfont=dict(color='white', size=12)  # 대비를 위해 흰색 글씨
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

def build_user_data(df, direction): 
    return [
        {
            "STORY_NAME": row["STORY"],
            "WEIGHT": row["WEIGHT"],
            "ELIV": row["ELEV"],
            "FORCE_X": row[f"Story Force for {direction}"] if direction == "X" else 0,
            "FORCE_Y": row[f"Story Force for {direction}"] if direction == "Y" else 0
        }
        for _, row in df.iterrows()
    ]

def to_excel_with_metadata(metadata, df_x, df_y):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        pd.DataFrame(list(metadata.items()), columns=["Parameter", "Value"]).to_excel(writer, sheet_name="Inputs", index=False)
        df_x.to_excel(writer, sheet_name="Seismic Loads X", index=False)
        df_y.to_excel(writer, sheet_name="Seismic Loads Y", index=False)
    output.seek(0)
    return output

def mark_inputs_dirty():
    st.session_state.calculated = False

if __name__ == "__main__":
    st.sidebar.title("Authentication")
    url = st.sidebar.text_input("Base URL", value="https://moa-engineers.midasit.com:443/gen")
    mapi_key = st.sidebar.text_input("MAPI-KEY", value="", type="password")

    if st.sidebar.button("Login", use_container_width=True):
        headers = {"MAPI-Key": mapi_key}
        try:
            res = requests.get(f"{url}/db/stld", headers=headers)
            if res.status_code == 200:
                st.sidebar.success("🚀 API connected successfully!")
                st.session_state.api_connected = True
                st.session_state.stld_data = res.json().get("STLD", {})
                res2 = requests.post(f"{url}/ope/storyprop", headers=headers, json={"Argument": {}})
                story_data = res2.json().get("STORYPROP", {}).get("DATA", [])
                df = pd.DataFrame(story_data)

                df["WEIGHT"] = df["WEIGHT"].astype(float)
                df["ELEV"] = df["ELEV"].astype(float)
                st.session_state.story_df = df
                st.session_state.structure_height = df["ELEV"].max()
            else:
                st.sidebar.error(f"API Error {res.status_code}: {res.text}")
        except Exception as e:
            st.sidebar.error(f"Connection failed: {e}")

    if st.session_state.api_connected:
        ## 화면 설정
        st.title("[AS 1170.4:2024] Static Seismic Loads Generator")
        story_df = st.session_state.story_df.copy()

        c1, c2 = st.columns([1,2])
        # 왼쪽 화면 설정
        with c1:
            with st.container():
                st.markdown("#### Seismic Design Parameters")
                st.divider()
                
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    st.write("Sub Soil Class")
                with col1_2:
                    soil_class = st.selectbox("Soil Class", ["Ae", "Be", "Ce", "De", "Ee"], key="soil_class",label_visibility="collapsed")
                
                col1_3, col1_4 = st.columns(2)
                with col1_3:
                    st.write("Importance Level")
                with col1_4:
                    importance_level = st.selectbox("Importance Level", ["2", "3", "4"], key="importance_level",label_visibility="collapsed")
                
                col1_5, col1_6 = st.columns(2)
                with col1_5:
                    st.write("Probability Factor (kp)")
                with col1_6:
                    kp = st.number_input("Probability Factor (kp)", min_value=0.0, max_value=1.8, value=0.10, key="kp",label_visibility="collapsed")
                
                col1_7, col1_8 = st.columns(2)
                with col1_7:
                    st.write("Hazard Factor (Z)")
                with col1_8:
                    Z = st.number_input("Hazard Factor (Z)", min_value=0.0, max_value=1.0, value=0.08, key="Z",label_visibility="collapsed")
                
                #Structure Height
                col1_9, col1_10 = st.columns(2)
                with col1_9:
                    st.write("Structure Height (hn)")
                with col1_10:
                    st.number_input("Structure Height (hn)", value=st.session_state.structure_height, disabled=True,label_visibility="collapsed")

                col1_11, col1_12 = st.columns(2)
                with col1_11:
                    st.write("Seismic Design Category")
                with col1_12:
                    st.text_input("Seismic Design Category", value=st.session_state.edc, disabled=True,label_visibility="collapsed")
                
                #방향별 파라미터 입력
                col1_13, col1_14, col1_15 = st.columns([1,0.5,0.5])
                with col1_13:
                    st.write("")
                with col1_14:
                    st.write("X-Direction")
                with col1_15:
                    st.write("Y-Direction")

                #μ/Sp
                col1_16, col1_17, col1_18, col1_19 = st.columns([1, 0.4, 0.4, 0.2])
                with col1_16:
                    st.write("μ/Sp")
                with col1_17:
                    mu_sp_x = float(st.selectbox("μ/Sp (X)", options=["1.3", "2.0","2.6", "4.5", "6.0"], key="mu_sp_x",label_visibility="collapsed"))
                with col1_18:
                    mu_sp_y = float(st.selectbox("μ/Sp (Y)", options=["1.3", "2.0","2.6", "4.5", "6.0"], key="mu_sp_y",label_visibility="collapsed"))
                with col1_19:
                    st.write("")
                
                #Fundamental Period
                col1_20, col1_21, col1_22, col1_23 = st.columns([1, 0.4, 0.4, 0.2])
                with col1_20:
                    st.write("Fundamental Period")
                with col1_21:
                    t1_x = st.number_input("Fundamental Period - X", min_value=0.0, value=st.session_state.t1_x, key="t1_x_input", label_visibility="collapsed")
                with col1_22:
                    t1_y = st.number_input("Fundamental Period - Y", min_value=0.0, value=st.session_state.t1_y, key="t1_y_input", label_visibility="collapsed")
                modal = Modal("Calculate Fundamental Period", key="modal_fundamental_period", padding=20)
                
                with col1_23:               
                    if st.button("🔍", use_container_width=True, type="primary"):
                        modal.open()
                if modal.is_open():
                    with modal.container():
                        # 첫 번째 열: X/Y 방향 경험식 선택
                        col2_1, col2_2 = st.columns(2)
                        with col2_1:
                            st.markdown("#### X- Direction Period")
                            x_formula = st.radio(
                                "X- Direction Period",
                                options=["T= 0.1375hn^0.75", "T= 0.09375hn^0.75", "T= 0.075hn^0.75", "T= 0.0625hn^0.75"],
                                index=0,
                                key="x_direction_period",
                                label_visibility="collapsed"
                            )
                        with col2_2:
                            st.markdown("#### Y- Direction Period")
                            y_formula = st.radio(
                                "Y- Direction Period",
                                options=["T= 0.1375hn^0.75", "T= 0.09375hn^0.75", "T= 0.075hn^0.75", "T= 0.0625hn^0.75"],
                                index=0,
                                key="y_direction_period",
                                label_visibility="collapsed"
                            )

                        hn = float(st.session_state.structure_height)
                        st.info(f"hn = {hn}m")

                        # 미리보기 버튼
                        if st.button("Calculate", key="btn_calc_t_period", use_container_width=True, type="primary"):
                            try:
                                st.session_state.calc_result_x = calc_fundamental_period(hn, x_formula)
                                st.session_state.calc_result_y = calc_fundamental_period(hn, y_formula)
                                st.session_state.period_calculated = True
                            except Exception as e:
                                    st.warning(f"❗ Error Occured: {e}")

                        # 계산 결과 보여주기
                        if st.session_state.get("period_calculated", False):
                            st.info(f"X = {st.session_state.calc_result_x} / Y = {st.session_state.calc_result_y}")

                        # OK / CLOSE 버튼은 한 줄에 배치 (최상위 열에서!)
                        btn_col1, btn_col2, btn_col3 = st.columns([6, 1, 1])
                        with btn_col2:
                            if st.button("OK", key="btn_period_ok", use_container_width=True, type="primary"):
                                try:
                                    calc_x = calc_fundamental_period(hn, x_formula)
                                    calc_y = calc_fundamental_period(hn, y_formula)
                                    st.session_state["t1_x"] = calc_x
                                    st.session_state["t1_y"] = calc_y
                                    st.session_state["period_calculated"] = False
                                    modal.close()
                                except Exception as e:
                                    st.warning(f"❗ Error Occured: {e}")
                        with btn_col3:
                            if st.button("CLOSE", key="btn_period_close", use_container_width=True):
                                st.session_state["period_calculated"] = False
                                modal.close()


                ## 하중 할당 화면 설정
                stld_names = [v["NAME"] for v in st.session_state.stld_data.values()]
                stld_key_map = {v["NAME"]: int(k) for k, v in st.session_state.stld_data.items()}
                st.markdown("#### Load Cases to Apply to Seismic Loads")
                st.divider()
                colA_1, colA_2 = st.columns(2)
                with colA_1:
                    st.write("Load Case for X-Direction")
                with colA_2:
                    load_case_x = st.selectbox("Load Case for X-Direction", stld_names,label_visibility="collapsed")
                
                colB_1, colB_2 = st.columns(2)
                with colB_1:
                    st.write("Load Case for Y-Direction")
                with colB_2:
                    load_case_y = st.selectbox("Load Case for Y-Direction", stld_names,label_visibility="collapsed")
                    
                hn = st.session_state.structure_height
                st.session_state.edc = calc_earthquake_design_category(importance_level, soil_class, kp, Z, hn)    

                if story_df is not None and not story_df.empty:
                    df_x = calculate_seismic_loads(story_df.copy(), t1_x, mu_sp_x, "X", soil_class, kp, Z)
                    df_y = calculate_seismic_loads(story_df.copy(), t1_y, mu_sp_y, "Y", soil_class, kp, Z)
                    st.session_state.df_x = df_x
                    st.session_state.df_y = df_y
                    st.session_state.calculated = True

        # 오른쪽 화면 설정
        with c2:
            with st.container():
                st.markdown("#### Graphs and Tables")
                if st.session_state.calculated:
                    df_x = st.session_state.df_x
                    df_y = st.session_state.df_y

                    tab1, tab2 = st.tabs(["Graph", "Table"])
                    with tab1:
                        col1_1, col1_2 = st.columns(2)
                        with col1_1:
                            st.write("Select Graph Type")
                        with col1_2:
                            graph_type = st.selectbox("Graph Type", ["Story Force", "Story Shear", "Overturning Moment"], key="graph_type",label_visibility="collapsed")
                        
                        col2_1, col2_2 = st.columns(2)
                        with col2_1:
                            plot_graph(df_x, f"{graph_type} for X", "X")
                        with col2_2:
                            plot_graph(df_y, f"{graph_type} for Y", "Y")

                    with tab2:
                        col2_1, col2_2 = st.columns(2)
                        with col2_1:
                            st.write("Select Table Direction")
                        with col2_2:
                            table_dir = st.selectbox("Select Table Direction", options=["X-Direction", "Y-Direction"], index=0, key="table_dir",label_visibility="collapsed")

                        # 테이블 숨길 열 목록
                        hidden_columns = ["LOADED_H", "LOADED_BX", "LOADED_BY", "Wi*Hi^k for X", "Wi*Hi^k for Y"]

                        if table_dir == "X-Direction":
                            display_df = df_x.drop(columns=[col for col in hidden_columns if col in df_x.columns])
                            # st.dataframe(display_df,use_container_width=True)
                            # AgGrid(display_df, use_container_width=True, height=700,fit_columns_on_grid_load=True)
                        else:
                            display_df = df_y.drop(columns=[col for col in hidden_columns if col in df_y.columns])
                            # st.dataframe(display_df,use_container_width=True)
                            # AgGrid(display_df, use_container_width=True, height=700,fit_columns_on_grid_load=True)

                            
                        
                        gb = GridOptionsBuilder.from_dataframe(display_df)

                        for col in display_df.columns:
                            gb.configure_column(col, flex=1,minWidth=150)
                        
                        
                        # 
                        gb.configure_default_column(
                            cellStyle={
                                "fontSize": "14px",
                                "padding": "10px",
                                "lineHeight": "20px",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center"
                            }
                        )

                        # 
                        grid_options = gb.build()
                        grid_options["rowHeight"] = 50  # 원하는 높이로 설정 (기본은 25)

                        # 
                        AgGrid(
                            display_df,
                            gridOptions=grid_options,
                            use_container_width=True,
                            height=700,
                            fit_columns_on_grid_load=False
                        )

                else:
                    st.info("Please fill in parameters to view results.")
                
        metadata = {
            "Sub Soil Class": soil_class,
            "Importance Level": importance_level,
            "Probability Factor (kp)": kp,
            "Hazard Factor (Z)": Z,
            "Structure Height": st.session_state.structure_height,
            "Seismic Design Category": st.session_state.edc,
            "μ/Sp X": mu_sp_x,
            "μ/Sp Y": mu_sp_y,
            "Fundamental Period X": t1_x,
            "Fundamental Period Y": t1_y,
        }

        
        ##이제 방향별로 각각의 load case에 API를 사용하여 하중을 할당시키자
        spacer, applied_button,excel_button = st.columns([5, 1, 1])
        with applied_button:
            if st.button("Apply Seismic Loads", use_container_width=True, type="primary"):
                if load_case_x == load_case_y:
                    st.toast(" Load Case - X and Load Case - Y must be different.", icon="❗")
                else:
                    try:
                        headers = {"MAPI-Key": mapi_key, "Content-Type": "application/json"}
                        post_url = f"{url}/db/seis/"

                        # 첫 번째: X방향 하중 전송
                        body_x = {
                            "Assign": {
                                stld_key_map[load_case_x]: {
                                    "SEIS_LOAD_CODE": "USER TYPE",
                                    "DESC": "",
                                    "SCALE_FACTOR_X": 1,
                                    "SCALE_FACTOR_Y": 0,
                                    "SCALE_FACTOR_R": 0,
                                    "bACCIDENTAL_ECCEN": False,
                                    "bINHERENT_ECCEN": False,
                                    "USER": build_user_data(df_x, "X")
                                }
                            }
                        }

                        res_x = requests.put(post_url, headers=headers, json=body_x)

                        if res_x.status_code == 200:
                            st.toast("X-direction seismic loads successfully applied.", icon="✅")
                        else:
                            try:
                                error_detail = res_x.json().get("message") or res_x.text
                            except Exception:
                                error_detail = res_x.text
                            st.error(f"❌ Failed to apply X-direction loads.\n\n**Status Code:** {res_x.status_code}\n**Details:** {error_detail}")

                        # 두 번째: Y방향 하중 전송
                        body_y = {
                            "Assign": {
                                stld_key_map[load_case_y]: {
                                    "SEIS_LOAD_CODE": "USER TYPE",
                                    "DESC": "",
                                    "SCALE_FACTOR_X": 0,
                                    "SCALE_FACTOR_Y": 1,
                                    "SCALE_FACTOR_R": 0,
                                    "bACCIDENTAL_ECCEN": False,
                                    "bINHERENT_ECCEN": False,
                                    "USER": build_user_data(df_y, "Y")
                                }
                            }
                        }

                        res_y = requests.put(post_url, headers=headers, json=body_y)

                        if res_y.status_code == 200:
                            st.toast("Y-direction seismic loads successfully applied.", icon="✅")
                        else:
                            try:
                                error_detail = res_y.json().get("message") or res_y.text
                            except Exception:
                                error_detail = res_y.text
                            st.error(f"❌ Failed to apply Y-direction loads.\n\n**Status Code:** {res_y.status_code}\n**Details:** {error_detail}")

                    except Exception as e:
                        st.error(f"❗ Unexpected error occurred while sending API request.\n\n**Exception:** {e}")

        with excel_button:
            metadata = {
            "Sub Soil Class": soil_class,
            "Importance Level": importance_level,
            "Probability Factor (kp)": kp,
            "Hazard Factor (Z)": Z,
            "Structure Height": st.session_state.structure_height,
            "Seismic Design Category": st.session_state.edc,
            "μ/Sp X": mu_sp_x,
            "μ/Sp Y": mu_sp_y,
            "Fundamental Period X": t1_x,
            "Fundamental Period Y": t1_y,
             }

            excel_file = to_excel_with_metadata(metadata, df_x, df_y)
            st.download_button(label="Download Excel Data", data=excel_file, file_name="seismic_loads.xlsx", type="primary",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


